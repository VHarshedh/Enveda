"""
patch_05b_spec2fp.py - Surgically updates only cells fa64e0ac and 4ee70ad7 in 05b_spec2fp_finetune.ipynb.
Leaves all other cells and manual customizations untouched.
"""
import ast
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path("05b_spec2fp_finetune.ipynb")

CELL_FA64E0AC_SOURCE = '''raw = model.module if hasattr(model, "module") else model
head_lr, llrd_gamma = 1e-4, 0.80
layer_lrs = {}
for name, _ in raw.named_parameters():
    if any(k in name for k in ("fp_head", "maccs_head", "heavy_atom", "ring_count", "elem_", "norm")):
        layer_lrs[name] = head_lr
    elif "encoder.layers." in name:
        lid = int(name.split("encoder.layers.")[1].split(".")[0])
        layer_lrs[name] = head_lr * (llrd_gamma ** (cfg.N_LAYERS - lid))
    else:
        layer_lrs[name] = head_lr * (llrd_gamma ** (cfg.N_LAYERS + 1))

param_groups = []
for name, p in raw.named_parameters():
    if not p.requires_grad:
        continue
    no_wd = p.ndim < 2 or any(nd in name.lower() for nd in ("bias", "norm", "emb", "cls_token"))
    param_groups.append({"params": [p], "lr": layer_lrs[name], "weight_decay": 0.0 if no_wd else cfg.WEIGHT_DECAY})

optimizer = torch.optim.AdamW(param_groups)
steps_per_epoch = max(len(dl_train), 1)
max_epochs = cfg.EPOCHS
total_steps = max_epochs * steps_per_epoch
warmup_steps = int(total_steps * cfg.WARMUP_FRAC)

def lr_lambda(step):
    if step < warmup_steps:
        return max(0.02, float(step) / max(warmup_steps, 1))
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    min_ratio = cfg.MIN_LR / head_lr
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler("cuda", enabled="cuda" in str(DEVICE))
pw_tensor = torch.from_numpy(pos_weight).to(DEVICE)
print(f"LLRD Ready: Heads={head_lr:.1e} -> Bottom={head_lr*(llrd_gamma**(cfg.N_LAYERS+1)):.1e} ({total_steps} steps)")'''

CELL_4EE70AD7_SOURCE = '''def compute_loss(out, batch, pw_tensor):
    dev = out["fp_logits"].device
    true_fp = batch["morgan_fp"].to(dev)
    true_mac = batch["maccs_fp"].to(dev)
    pw_dev = pw_tensor.to(dev)
    fp_loss = F.binary_cross_entropy_with_logits(
        out["fp_logits"], true_fp * 0.995 + 0.0025, pos_weight=pw_dev, reduction="mean")
    maccs_loss = F.binary_cross_entropy_with_logits(
        out["maccs_logits"], true_mac * 0.995 + 0.0025, reduction="mean")
    heavy_loss = F.smooth_l1_loss(out["heavy_atoms"], batch["heavy_atoms"].to(dev))
    ring_loss = F.smooth_l1_loss(out["ring_count"], batch["ring_count"].to(dev))
    epres_loss = F.binary_cross_entropy_with_logits(
        out["elem_presence"], batch["elem_presence"].to(dev), reduction="mean")
    ecount_loss = F.smooth_l1_loss(out["elem_count"], batch["elem_count"].to(dev))
    bce_multi = fp_loss + 0.5 * maccs_loss + 0.01 * heavy_loss + 0.02 * ring_loss + 0.2 * epres_loss + 0.02 * ecount_loss

    cal_logits = out["fp_logits"] - 0.5 * torch.log(pw_dev.clamp(min=1.0))
    pred_p = torch.sigmoid(cal_logits).float()
    pred_mac = torch.sigmoid(out["maccs_logits"]).float()
    cos_m = 1.0 - F.cosine_similarity(pred_p, true_fp, dim=1, eps=1e-6).mean()
    cos_mac = 1.0 - F.cosine_similarity(pred_mac, true_mac, dim=1, eps=1e-6).mean()
    inter = (pred_p * true_fp).sum(dim=1)
    union = (pred_p * pred_p).sum(dim=1) + true_fp.sum(dim=1) - inter
    tan_loss = (1.0 - (inter + 1e-5) / (union + 1e-5)).mean()

    # In-batch InfoNCE contrastive ranking loss (pushes apart hard negatives)
    p_norm = F.normalize(pred_p, p=2, dim=1, eps=1e-6)
    t_norm = F.normalize(true_fp, p=2, dim=1, eps=1e-6)
    sim_mat = (p_norm @ t_norm.T) / 0.07
    same_mol = (true_fp @ true_fp.T) >= (true_fp.sum(dim=1, keepdim=True) - 0.5)
    eye = torch.eye(len(true_fp), dtype=torch.bool, device=dev)
    sim_mat = sim_mat.masked_fill(same_mol & ~eye, -1e4)
    infonce_loss = F.cross_entropy(sim_mat, torch.arange(len(true_fp), device=dev))

    total = bce_multi + 0.30 * cos_m + 0.10 * cos_mac + 0.20 * tan_loss + 0.10 * infonce_loss
    return total, fp_loss.item()'''


def patch_notebook():
    nb = nbf.read(str(NOTEBOOK_PATH), as_version=4)
    patched_count = 0

    for idx, cell in enumerate(nb.cells):
        cid = cell.get("id", "")
        if cid == "fa64e0ac":
            # Syntax validation before assignment
            ast.parse(CELL_FA64E0AC_SOURCE)
            cell.source = CELL_FA64E0AC_SOURCE
            patched_count += 1
            print(f"Patched cell {idx} (id={cid}): LLRD optimizer ({len(CELL_FA64E0AC_SOURCE.splitlines())} lines)")
        elif cid == "4ee70ad7":
            # Syntax validation before assignment
            ast.parse(CELL_4EE70AD7_SOURCE)
            cell.source = CELL_4EE70AD7_SOURCE
            patched_count += 1
            print(f"Patched cell {idx} (id={cid}): InfoNCE + Dual Cosine/Tanimoto loss ({len(CELL_4EE70AD7_SOURCE.splitlines())} lines)")

    assert patched_count == 2, f"Expected to patch exactly 2 cells, but found {patched_count}"

    # Validate the full notebook schema
    nbf.validate(nb)

    # Validate AST across all code cells
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            clean_src = "\n".join(line for line in c.source.splitlines() if not line.strip().startswith(("!", "%")))
            try:
                ast.parse(clean_src)
            except SyntaxError as e:
                raise SyntaxError(f"AST parse error in cell {i} (id={c.get('id')}): {e}")

    nbf.write(nb, str(NOTEBOOK_PATH))
    print(f"Successfully patched {NOTEBOOK_PATH} (total cells: {len(nb.cells)})")


if __name__ == "__main__":
    patch_notebook()
