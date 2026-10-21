"""Generate 05b_spec2fp_finetune.ipynb — Warm-start fine-tuning for Spec2FP.

Fine-tunes the Stage 05 checkpoint with:
1. Warm-start loading from Stage 05 EMA checkpoint (spec2fp_model.pt).
2. Higher peak resolution (TOP_K_PEAKS=192) & reduced augmentation noise.
3. Hybrid ranking-aligned loss (BCE multi-task + calibrated Cosine + soft Tanimoto).
4. Dual export of 4096-bit Morgan AND 167-bit MACCS predictions (fp_pred_*.npy and maccs_pred_*.npy).
5. Combined Morgan + MACCS cosine similarity scoring in proxy MRR@25.
6. Zero-CoW memory optimizations, DataParallel support, and artifact contract compliance.
7. Pre-parsed mol optimization in Target FP generation (cuts startup time in half).
8. Exact 65-cell layout matching notebook with preserved cell IDs (Cell 4f325ce3 to Cell 99b606d1).
"""
from pathlib import Path
import ast
import argparse
import textwrap
import warnings
import json
import nbformat as nbf

NOTEBOOK_CELLS: list = []

ROOT = Path(__file__).resolve().parent

EXACT_CELL_IDS = [
    "4f325ce3", "b6bd082c", "dbe328fd", "17d6790d", "e9aa6fb9",
    "fb1d0ce6", "2f8b8e5c", "0e1e3023", "bcd931c9", "84da84f4",
    "3eaa5773", "729709af", "d993e50b", "f782ddf8", "96e30dc9",
    "0d9575a3", "ff4bc8fe", "4a2c647c", "09f82e45", "8a15fbc8",
    "9633aecb", "38a92230", "98ab0ef0", "1fc8c571", "aa201a28",
    "31faade1", "8cc2339e", "9567a004", "8e612ca8", "d59cc773",
    "e46a0408", "c5410b3e", "7cdab131", "69a68cab", "b4fa50ab",
    "c15a7a46", "6004fce3", "384f8b3c", "051ace2d", "d77b529d",
    "54595f66", "fa64e0ac", "4ee70ad7", "668af433", "432b12fb",
    "d73a4d07", "0f51a69b", "564eec83", "1034eb36", "1326bb0b",
    "891b4682", "14a47ea2", "2f442292", "195c77e6", "882672be",
    "7b69d6d6", "a5685a5d", "3232d7cd", "dd239d92", "ccadf1ad",
    "e73da7d1", "cf25f579", "c97d12a0", "5c8d6cf0", "99b606d1"
]


def md(text: str):
    NOTEBOOK_CELLS.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str, check_len: bool = True):
    code_str = textwrap.dedent(text).strip()
    lines = code_str.splitlines()
    if check_len and len(lines) >= 60:
        raise ValueError(f"Code cell exceeds 59 lines ({len(lines)} lines):\n{lines[0]}")
    py_lines = "\n".join(l for l in lines if not l.strip().startswith(("!", "%")))
    if py_lines.strip():
        ast.parse(py_lines)
    NOTEBOOK_CELLS.append(nbf.v4.new_code_cell(code_str))


# ═══════════════════════════════════════════════════════════════════════════════
# ██  SPEC2FP MODULE SOURCE  ██
# ═══════════════════════════════════════════════════════════════════════════════
SPEC2FP_SOURCE = r'''"""spec2fp.py - Spectrum-to-Fingerprint model, tensorization, and inference.

Defines the Spec2FP Transformer, peak tensorization, and predict() API.
Import ONLY from /kaggle/working/.
"""
import json, math, sys, warnings, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ADDUCT_VOCAB = [
    "<pad>", "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+",
    "[M+H-H2O]+", "[M+FA-H]-", "[M+Cl]-", "[M]+", "[M-H2O+H]+",
    "[M-H2O-H]-", "[M+2H]2+", "[M-2H]2-", "[M+HAc-H]-",
    "[M+2Na-H]+", "[M+CH2O2-H]-", "<unk>",
]
ADDUCT2IDX = {a: i for i, a in enumerate(ADDUCT_VOCAB)}
INST_VOCAB = [
    "<pad>", "timsTOF", "qtof", "orbitrap", "FTICR", "QqQ",
    "ion_trap", "None/Unknown", "<unk>",
]
INST2IDX = {a: i for i, a in enumerate(INST_VOCAB)}
POLARITY_VOCAB = ["<pad>", "positive", "negative", "<unk>"]
POLARITY2IDX = {a: i for i, a in enumerate(POLARITY_VOCAB)}
ELEMENTS_ORDER = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
N_ELEMENTS = len(ELEMENTS_ORDER)


def parse_peak_array(val: Any) -> np.ndarray:
    """Robustly parse peak m/z or intensity into a float64 numpy array."""
    if isinstance(val, np.ndarray):
        return val.astype(np.float64)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float64)
    if isinstance(val, str):
        try:
            return np.asarray(json.loads(val), dtype=np.float64)
        except Exception:
            pass
    try:
        return np.asarray(val, dtype=np.float64)
    except Exception:
        return np.zeros(0, dtype=np.float64)


def sinusoidal_embed(values: np.ndarray, dim: int = 64) -> np.ndarray:
    """Multi-scale sinusoidal encoding resolving 0.005 Da mass defects up to 2000 Da."""
    freqs = np.logspace(
        math.log10(2 * math.pi / 2000.0),
        math.log10(2 * math.pi / 0.005),
        dim // 2, dtype=np.float32)
    angles = np.outer(values.astype(np.float32), freqs)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def tensorize_spectrum(
    mz: Any, inten: Any, precursor_mz: float,
    adduct: Any, ce: float, instrument: Any, polarity: Any,
    top_k: int = 192, mz_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """Convert a single spectrum into model-ready numpy arrays."""
    mz = parse_peak_array(mz)
    inten = parse_peak_array(inten)
    mask_valid = mz <= precursor_mz + 5.0
    mz, inten = mz[mask_valid], inten[mask_valid]
    if len(mz) == 0:
        mz = np.array([precursor_mz], dtype=np.float64)
        inten = np.array([1.0], dtype=np.float64)
    max_int = max(inten.max(), 1e-12)
    inten = np.sqrt(inten / max_int)
    if len(mz) > top_k:
        idx = np.argsort(inten)[-top_k:]
        idx = idx[np.argsort(mz[idx])]
        mz, inten = mz[idx], inten[idx]
    n = len(mz)
    mz_emb = sinusoidal_embed(mz, mz_dim)
    nl = precursor_mz - mz
    nl_emb = sinusoidal_embed(nl, mz_dim)
    log_int = np.log1p(inten).astype(np.float32).reshape(-1, 1)
    peak_feat = np.concatenate([mz_emb, log_int, nl_emb], axis=1).astype(np.float32)
    adduct_idx = int(adduct) if isinstance(adduct, (int, np.integer)) else ADDUCT2IDX.get(str(adduct).strip(), ADDUCT2IDX["<unk>"])
    inst_idx = int(instrument) if isinstance(instrument, (int, np.integer)) else INST2IDX.get(str(instrument).strip(), INST2IDX["<unk>"])
    pol_idx = int(polarity) if isinstance(polarity, (int, np.integer)) else POLARITY2IDX.get(str(polarity).strip().lower(), POLARITY2IDX["<unk>"])
    ce_val = float(ce) if not (ce is None or (isinstance(ce, float) and math.isnan(ce))) else 0.0
    return {
        "peak_feat": peak_feat, "n_peaks": n,
        "adduct_idx": adduct_idx, "inst_idx": inst_idx, "pol_idx": pol_idx,
        "ce_val": np.float32(ce_val), "prec_mz": np.float32(precursor_mz),
    }


def collate_spectra(batch: list, top_k: int = 192, feat_dim: int = 129):
    """Collate list of tensorize_spectrum outputs into batched tensors."""
    B = len(batch)
    max_len = top_k + 3
    peaks = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(B, max_len, dtype=torch.bool)
    adducts = torch.zeros(B, dtype=torch.long)
    instruments = torch.zeros(B, dtype=torch.long)
    polarities = torch.zeros(B, dtype=torch.long)
    ces = torch.zeros(B, dtype=torch.float32)
    prec_mzs = torch.zeros(B, dtype=torch.float32)
    for i, item in enumerate(batch):
        n = item["n_peaks"]
        peaks[i, :n] = torch.from_numpy(item["peak_feat"][:n])
        mask[i, :n] = False
        adducts[i] = item["adduct_idx"]
        instruments[i] = item["inst_idx"]
        polarities[i] = item["pol_idx"]
        ces[i] = float(item["ce_val"])
        prec_mzs[i] = float(item["prec_mz"])
    return {
        "peaks": peaks, "pad_mask": mask,
        "adduct": adducts, "instrument": instruments, "polarity": polarities,
        "ce": ces, "prec_mz": prec_mzs,
    }


class Spec2FPModel(nn.Module):
    """Transformer encoder mapping MS/MS spectrum to molecular fingerprint."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d, n_layers, n_heads = cfg.get("d_model", 320), cfg.get("n_layers", 5), cfg.get("n_heads", 8)
        dropout, feat_dim = cfg.get("dropout", 0.1), cfg.get("feat_dim", 129)
        fp_bits, maccs_bits = cfg.get("fp_bits", 4096), cfg.get("maccs_bits", 167)
        n_elements = cfg.get("n_elements", N_ELEMENTS)
        self.peak_proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.adduct_emb = nn.Embedding(len(ADDUCT_VOCAB), d, padding_idx=0)
        self.inst_emb = nn.Embedding(len(INST_VOCAB), d, padding_idx=0)
        self.pol_emb = nn.Embedding(len(POLARITY_VOCAB), d, padding_idx=0)
        self.ce_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())
        self.prec_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)
        self.fp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, fp_bits))
        self.maccs_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, maccs_bits))
        self.heavy_atom_head = nn.Linear(d, 1)
        self.ring_count_head = nn.Linear(d, 1)
        self.elem_presence_head = nn.Linear(d, n_elements)
        self.elem_count_head = nn.Linear(d, n_elements)

    def forward(self, peaks, pad_mask, adduct, instrument, polarity, ce, prec_mz):
        B = peaks.size(0)
        x = self.peak_proj(peaks)
        cls = self.cls_token.expand(B, -1, -1)
        meta1 = (
            self.adduct_emb(adduct)
            + self.ce_proj(ce.unsqueeze(-1) / 100.0)
            + self.prec_proj(prec_mz.unsqueeze(-1) / 1000.0)
        )
        meta2 = self.inst_emb(instrument) + self.pol_emb(polarity)
        meta = torch.stack([meta1, meta2], dim=1)
        x = torch.cat([cls, meta, x], dim=1)
        cls_meta_mask = torch.zeros(B, 3, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_meta_mask, pad_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=full_mask)
        cls_out = x[:, 0]
        valid_tok = (~full_mask).float().unsqueeze(-1)
        mean_out = (x * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp(min=1.0)
        x = self.norm(cls_out + mean_out)
        return {
            "fp_logits": self.fp_head(x), "maccs_logits": self.maccs_head(x),
            "heavy_atoms": self.heavy_atom_head(x).squeeze(-1),
            "ring_count": self.ring_count_head(x).squeeze(-1),
            "elem_presence": self.elem_presence_head(x),
            "elem_count": self.elem_count_head(x),
        }


def load_model(config_path: str, weights_path: str, device: str = "cpu"):
    """Load Spec2FP model from config + weights."""
    with open(config_path) as f:
        cfg = json.load(f)
    model = Spec2FPModel(cfg["model"])
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    if "ema_state_dict" in sd:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd["ema_state_dict"].items()})
    elif "model_state_dict" in sd:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd["model_state_dict"].items()})
    else:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    pw_path = Path(weights_path).parent / "spec2fp_pos_weights.npy"
    if pw_path.is_file():
        try:
            model.pos_weights = np.load(pw_path)
        except Exception:
            pass
    model.eval()
    return model.to(device), cfg


@torch.no_grad()
def predict_full(model, clean_df, device="cuda", batch_size=128, top_k=192, mz_dim=64, pos_weights=None):
    """Predict Morgan-4096, MACCS-167, and element counts with streaming molecule-level pooling."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    pw = pos_weights if pos_weights is not None else getattr(raw_model, "pos_weights", None)
    pw_shift = 0.5 * torch.log(torch.as_tensor(pw, device=device, dtype=torch.float32).clamp(min=1.0)) if pw is not None else None
    feat_dim, n_rows = 2 * mz_dim + 1, len(clean_df)
    if n_rows == 0:
        return (np.zeros((0, 4096), dtype=np.float16), np.zeros((0, 167), dtype=np.float16),
                np.zeros((0, N_ELEMENTS), dtype=np.float32), [])

    mol_ids_arr = clean_df["molecule_id"].to_numpy().astype(str)
    unique_mols, inv_indices = np.unique(mol_ids_arr, return_inverse=True)
    n_unique = len(unique_mols)
    fp_sum = np.zeros((n_unique, 4096), dtype=np.float32)
    maccs_sum = np.zeros((n_unique, 167), dtype=np.float32)
    elem_sum = np.zeros((n_unique, N_ELEMENTS), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.float32)

    mz_col = clean_df["mz"].to_numpy()
    int_col = clean_df["inten"].to_numpy()
    pmz_col = clean_df["precursor_mz"].to_numpy(dtype=np.float32)
    add_col = clean_df["adduct"].to_numpy() if "adduct" in clean_df.columns else np.array(["[M+H]+"] * n_rows)
    ce_col = pd.to_numeric(clean_df["ce"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "ce" in clean_df.columns else np.zeros(n_rows, dtype=np.float32)
    inst_col = clean_df["instrument"].to_numpy() if "instrument" in clean_df.columns else np.array(["unknown"] * n_rows)
    pol_col = clean_df["polarity"].to_numpy() if "polarity" in clean_df.columns else np.array(["positive"] * n_rows)
    dev_type, is_cuda = ("cuda" if "cuda" in str(device) else "cpu"), ("cuda" in str(device))

    for s_idx in range(0, n_rows, batch_size):
        e_idx = min(s_idx + batch_size, n_rows)
        items = [
            tensorize_spectrum(
                mz_col[r], int_col[r], float(pmz_col[r]),
                str(add_col[r]), float(ce_col[r]),
                str(inst_col[r]), str(pol_col[r]), top_k=top_k, mz_dim=mz_dim)
            for r in range(s_idx, e_idx)
        ]
        bd = collate_spectra(items, top_k=top_k, feat_dim=feat_dim)
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bd.items()}
        with torch.amp.autocast(device_type=dev_type, dtype=torch.float16, enabled=is_cuda):
            out = raw_model(**bd)
        fp_logits = out["fp_logits"] - pw_shift if pw_shift is not None else out["fp_logits"]
        batch_fp = torch.sigmoid(fp_logits).float().cpu().numpy()
        batch_maccs = torch.sigmoid(out["maccs_logits"]).float().cpu().numpy()
        batch_elem = out["elem_count"].float().cpu().numpy()
        batch_inv = inv_indices[s_idx:e_idx]
        np.add.at(fp_sum, batch_inv, batch_fp)
        np.add.at(maccs_sum, batch_inv, batch_maccs)
        np.add.at(elem_sum, batch_inv, batch_elem)
        np.add.at(counts, batch_inv, 1.0)

    counts_col = np.maximum(counts, 1.0)[:, None]
    return ((fp_sum / counts_col).astype(np.float16),
            (maccs_sum / counts_col).astype(np.float16),
            (elem_sum / counts_col).astype(np.float32), unique_mols.tolist())


@torch.no_grad()
def predict(model, clean_df, device="cuda", batch_size=128, top_k=192, mz_dim=64, pos_weights=None, return_maccs=False):
    """Predict fingerprints with optional MACCS return for backward-compatibility."""
    fp_pred, maccs_pred, elem_pred, mol_ids = predict_full(
        model, clean_df, device=device, batch_size=batch_size, top_k=top_k, mz_dim=mz_dim, pos_weights=pos_weights
    )
    if return_maccs:
        return fp_pred, maccs_pred, elem_pred, mol_ids
    return fp_pred, elem_pred, mol_ids
'''


def add_module_cells(source: str, filename: str = "spec2fp.py"):
    """Write module in AST-aligned chunks each < 55 lines via %%writefile."""
    lines = source.strip().splitlines()
    tree = ast.parse("\n".join(lines))
    chunks, current = [], []
    previous = 0
    for node in tree.body:
        segment = lines[previous:node.end_lineno]
        previous = node.end_lineno
        if len(current) + len(segment) > 48:
            if current:
                chunks.append("\n".join(current).strip())
            current = segment
        else:
            current += segment
    if current:
        chunks.append("\n".join(current).strip())
    for i, chunk in enumerate(chunks):
        if len(chunk.splitlines()) > 57:
            raise ValueError(f"Module block exceeds cell limit: {len(chunk.splitlines())} lines\n{chunk[:100]}")
        ast.parse(chunk)
        magic = f"%%writefile /kaggle/working/{filename}" if i == 0 else f"%%writefile -a /kaggle/working/{filename}"
        NOTEBOOK_CELLS.append(nbf.v4.new_code_cell(magic + "\n" + chunk + "\n\n"))


# ═══════════════════════════════════════════════════════════════════════════════
# ██  BUILD NOTEBOOK  ██
# ═══════════════════════════════════════════════════════════════════════════════
def build_notebook(output: Path) -> nbf.NotebookNode:
    NOTEBOOK_CELLS.clear()

    # ── 0. Title (Cell 00: 4f325ce3) ──────────────────────────────────────
    md(r"""
    # 05b — Spec2FP: Spectrum → Fingerprint Fine-Tuning (Warm-Start)

    **Purpose**: Fine-tune the pre-trained Spec2FP Transformer (`spec2fp_model.pt`)
    with higher peak resolution (`TOP_K_PEAKS=192`), reduced augmentation noise,
    a ranking-aligned hybrid loss function (BCE + prior-calibrated cosine loss +
    soft Tanimoto surrogate), and dual export of both 4096-bit Morgan and
    167-bit MACCS predictions (`fp_pred_*.npy` and `maccs_pred_*.npy`).

    **Warm-Start**: Locates `spec2fp_model.pt` (which achieved 0.7022 proxy MRR@25
    in Stage 05) via `find_file` and warm-starts model weights.

    **Inputs** (via `find_file`): `train_clean.parquet`, `test_clean.parquet`,
    `splits.parquet`, `evalkit.py`, Stage 05 `spec2fp_model.pt`, `spec2fp_config.json`,
    `spec2fp_pos_weights.npy`.

    **Outputs**: Updated `spec2fp_model.pt`, `spec2fp_config.json`, `fp_pred_*.npy`,
    `maccs_pred_*.npy`, `elem_pred_*.npy`, `spec2fp_val_metrics.json`, `manifest.json`.

    **Runtime**: ~2.8 h on 2× T4 (6 epochs); SMOKE_TEST ≈ 2 min CPU.
    **GPU RAM**: ~7–9 GB per T4 with fp16 autocast, batch 256 on DataParallel.
    """)

    # ── 1. Config (Cell 01: b6bd082c) ─────────────────────────────────────
    md("## 1. Configuration & Fine-Tuning Hyperparameters")

    # Cell 02: dbe328fd
    code("!pip install rdkit", check_len=False)

    # Cell 03: 17d6790d - Embedded evalkit.py
    evalkit_file = ROOT / "evalkit.py"
    evalkit_source = evalkit_file.read_text(encoding="utf-8") if evalkit_file.is_file() else ""
    code(f"%%writefile evalkit.py\n{evalkit_source}", check_len=False)

    # Cell 04: e9aa6fb9
    code(r"""
    import os, sys, gc, time, math, json, hashlib, warnings, copy, re, shutil
    import importlib.util
    from pathlib import Path
    from dataclasses import dataclass, asdict
    from datetime import datetime, timezone
    from collections import Counter, defaultdict
    from typing import Dict, List, Tuple, Optional, Any, Set
    import numpy as np
    import pandas as pd

    SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
    """)

    # Cell 05: fb1d0ce6
    code(r"""
    @dataclass
    class CFG:
        stage: str = "spec2fp"
        seed: int = 42
        SMOKE_TEST: bool = SMOKE_TEST
        TOP_K_PEAKS: int = 192
        MZ_DIM: int = 64
        FEAT_DIM: int = 129
        D_MODEL: int = 320
        N_LAYERS: int = 5
        N_HEADS: int = 8
        DROPOUT: float = 0.1
        FP_BITS: int = 4096
        MACCS_BITS: int = 167
        LR: float = 8e-5
        MIN_LR: float = 1e-6
        EPOCHS: int = 6
        WEIGHT_DECAY: float = 0.01
        GRAD_CLIP: float = 1.0
        BATCH_SIZE: int = 256
        PRED_BATCH_SIZE: int = 128
        TRAIN_BUDGET_H: float = 3.5
        WARMUP_FRAC: float = 0.05
        EMA_DECAY: float = 0.9995
        PATIENCE: int = 5
        AUG_PEAK_DROPOUT: float = 0.10
        AUG_INTENSITY_NOISE: float = 0.02
        AUG_MZ_JITTER: float = 0.002
        COS_LOSS_WEIGHT: float = 0.35
        TAN_LOSS_WEIGHT: float = 0.25
        DATA_DIR: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"
        OUTPUT_DIR: str = "/kaggle/working/artifacts/spec2fp"
        CKPT_STAGE: str = "spec2fp"
        TEST_MODEL: str = "fold0"
        FULL_TRAIN: bool = False
        time_budget_s: float = 3.5 * 3600
        SUBMIT_BUDGET_S: float = 8.0 * 3600
    """)

    # Cell 06: 2f8b8e5c
    code(r"""
    cfg = CFG()
    if cfg.SMOKE_TEST:
        cfg.BATCH_SIZE = 16
        cfg.PRED_BATCH_SIZE = 32
        cfg.TRAIN_BUDGET_H = 0.02
        cfg.TOP_K_PEAKS = 32
        cfg.EPOCHS = 1
        cfg.PATIENCE = 2
        cfg.time_budget_s = 180

    cfg.FEAT_DIM = 2 * cfg.MZ_DIM + 1
    T_START = time.perf_counter()
    print(f"CFG: SMOKE_TEST={cfg.SMOKE_TEST}, TOP_K={cfg.TOP_K_PEAKS}, LR={cfg.LR}, EPOCHS={cfg.EPOCHS}")
    """)

    # ── 2. Imports & env report (Cell 07: 0e1e3023) ───────────────────────
    md("## 2. Imports & Environment Report")

    # Cell 08: bcd931c9
    code(r"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import autocast, GradScaler

    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    N_GPU = torch.cuda.device_count()
    print(f"GPUs available: {N_GPU}")
    for i in range(N_GPU):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Primary compute device: {DEVICE}")
    """)

    # Cell 09: 84da84f4
    code(r"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _HAS_FP_GEN = True
    except ImportError:
        _HAS_FP_GEN = False
    RDLogger.DisableLog("rdApp.*")
    print(f"RDKit: {Chem.rdBase.rdkitVersion} | NumPy: {np.__version__} | Pandas: {pd.__version__}")

    from tqdm.auto import tqdm
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
    """)

    # ── 3. Utilities (Cell 10: 3eaa5773) ──────────────────────────────────
    md("## 3. Utilities & Seeding")

    # Cell 11: 729709af
    code(r"""
    def seed_everything(seed=42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    seed_everything(cfg.seed)
    WORK = Path("/kaggle/working")
    WORK.mkdir(parents=True, exist_ok=True)
    """)

    # Cell 12: d993e50b
    code(r"""
    def find_file(filename, hint="", required=True, roots=None):
        roots = roots or [WORK, Path("/kaggle/input"), Path(".")]
        paths = list({p.resolve() for root in roots for p in Path(root).rglob(filename)})
        paths.sort(key=lambda p: (bool(hint and hint in str(p)), p.stat().st_mtime, str(p)), reverse=True)
        if paths:
            found = paths[0]
            if filename.endswith(".py") and found.parent.resolve() != WORK.resolve():
                WORK.mkdir(parents=True, exist_ok=True)
                dst = WORK / filename
                if not dst.is_file() or dst.stat().st_mtime < found.stat().st_mtime:
                    shutil.copy2(found, dst)
                    print(f"[find_file] Staged {filename} from {found} -> {dst}")
                return dst
            print(f"[find_file] {filename}: {found}")
            return found
        if required:
            raise FileNotFoundError(f"Missing {filename}; searched {WORK} and /kaggle/input")
        print(f"[find_file] optional {filename}: absent")
        return None
    """)

    # Cell 13: f782ddf8
    code(r"""
    def find_manifest(stage):
        matches = []
        for root in (WORK, Path("/kaggle/input"), Path(".")):
            for path in root.rglob("manifest*.json"):
                try:
                    rec = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if rec.get("stage") == stage or rec.get("stage_alias") == stage:
                    matches.append((path.stat().st_mtime, path, rec))
        return max(matches, default=(None, None, None), key=lambda x: x[0])[1:]
    """)

    # Cell 14: 96e30dc9
    code(r"""
    ek_path = find_file("evalkit.py", hint="eval_splits")
    if ek_path.resolve() != (WORK / "evalkit.py").resolve():
        raise RuntimeError("evalkit.py must be read only from /kaggle/working")
    spec = importlib.util.spec_from_file_location("evalkit", ek_path)
    evalkit = importlib.util.module_from_spec(spec)
    sys.modules["evalkit"] = evalkit
    spec.loader.exec_module(evalkit)
    for name in ("FP_SPEC", "fast_key", "canon_ik14", "compute_morgan_fp", "compute_maccs_fp", "mrr_at_25"):
        assert hasattr(evalkit, name), f"evalkit missing {name}"
    print(f"evalkit successfully loaded from {ek_path}; FP_SPEC={evalkit.FP_SPEC}")
    """)

    # ── 4. Write spec2fp.py (Cell 15: 0d9575a3) ───────────────────────────
    md(r"""
    ## 4. Reusable Module: `spec2fp.py`

    Write updated `spec2fp.py` to `/kaggle/working/spec2fp.py` via `%%writefile`.
    Defines model class, sinusoidal embedding, peak tensorization with TOP_K=192,
    and both `predict_full()` and `predict()` extracting Morgan + MACCS keys.
    """)
    # Cells 16-22 (ff4bc8fe, 4a2c647c, 09f82e45, 8a15fbc8, 9633aecb, 38a92230, 98ab0ef0)
    add_module_cells(SPEC2FP_SOURCE, "spec2fp.py")

    # Cell 23: 1fc8c571
    code(r"""
    sys.path.insert(0, str(WORK))
    import spec2fp
    for name in ("Spec2FPModel", "tensorize_spectrum", "collate_spectra", "load_model", "predict_full", "predict"):
        assert hasattr(spec2fp, name), f"spec2fp missing {name}"
    print(f"spec2fp imported successfully: {spec2fp.__file__}")
    """)

    # ── 5. Load Data & Splits (Cell 24: aa201a28) ─────────────────────────
    md("## 5. Load Clean Data & Splits Contract")

    # Cell 25: 31faade1
    code(r"""
    train_path = find_file("train_clean.parquet", hint="eda")
    test_path = find_file("test_clean.parquet", hint="eda")
    splits_path = find_file("splits.parquet", hint="eval_splits")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    splits_df = pd.read_parquet(splits_path)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Splits: {splits_df.shape}")
    """)

    # Cell 26: 8cc2339e
    code(r"""
    if "molecule_id" not in splits_df.columns:
        for c in splits_df.columns:
            if "molecule" in c.lower() or "id" in c.lower() or "inchikey" in c.lower():
                splits_df = splits_df.rename(columns={c: "molecule_id"})
                break
    splits_df["molecule_id"] = splits_df["molecule_id"].astype(str)
    train_df["molecule_id"] = train_df["molecule_id"].astype(str)
    train_df = train_df.merge(
        splits_df[["molecule_id", "split3"]].drop_duplicates("molecule_id"),
        on="molecule_id", how="left"
    )
    assert train_df["split3"].notna().all(), "Not all train spectra matched to split3!"
    del splits_df
    gc.collect()
    print(f"Split3 counts (spectra):\n{train_df['split3'].value_counts()}")
    print(f"Split3 counts (molecules):\n{train_df.groupby('split3')['molecule_id'].nunique()}")
    """)

    # ── 6. Targets Calculation (Cell 27: 9567a004) ────────────────────────
    md("## 6. Precompute Multi-Task Targets & Positive Weights")

    # Cell 28: 8e612ca8
    code(r"""
    train_mols = train_df.drop_duplicates("molecule_id")[["molecule_id", "smiles"]].copy()
    train_mols = train_mols.reset_index(drop=True)
    print(f"Computing targets for {len(train_mols)} unique train molecules...")

    morgan_fps, maccs_fps = [], []
    heavy_atoms, ring_counts = [], []
    elem_presence, elem_counts = [], []
    ELEMENTS_ORDER = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    fail_count = 0
    """)

    # Cell 29: d59cc773 - Optimized Target FP loop passing pre-parsed mol
    code(r"""
    for _, row in tqdm(train_mols.iterrows(), total=len(train_mols), desc="Target FPs"):
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            morgan_fps.append(evalkit.compute_morgan_fp(mol, n_bits=cfg.FP_BITS))
            maccs_fps.append(evalkit.compute_maccs_fp(mol))
            heavy_atoms.append(mol.GetNumHeavyAtoms())
            ring_counts.append(Descriptors.RingCount(mol))
            formula = rdMolDescriptors.CalcMolFormula(mol)
            ep = np.zeros(len(ELEMENTS_ORDER), dtype=np.float32)
            ec = np.zeros(len(ELEMENTS_ORDER), dtype=np.float32)
            for i, el in enumerate(ELEMENTS_ORDER):
                matches = re.findall(el + r'(?![a-z])(\d*)', formula)
                total = sum(int(m) if m else 1 for m in matches)
                ep[i] = 1.0 if total > 0 else 0.0
                ec[i] = float(total)
            elem_presence.append(ep)
            elem_counts.append(ec)
        else:
            fail_count += 1
            morgan_fps.append(evalkit.compute_morgan_fp(smi, n_bits=cfg.FP_BITS))
            maccs_fps.append(evalkit.compute_maccs_fp(smi))
            heavy_atoms.append(0)
            ring_counts.append(0)
            elem_presence.append(np.zeros(len(ELEMENTS_ORDER), dtype=np.float32))
            elem_counts.append(np.zeros(len(ELEMENTS_ORDER), dtype=np.float32))
    """)

    # Cell 30: e46a0408
    code(r"""
    morgan_arr = np.stack(morgan_fps).astype(np.uint8)
    maccs_arr = np.stack(maccs_fps).astype(np.uint8)
    heavy_arr = np.array(heavy_atoms, dtype=np.float32)
    ring_arr = np.array(ring_counts, dtype=np.float32)
    epres_arr = np.stack(elem_presence).astype(np.float32)
    ecount_arr = np.stack(elem_counts).astype(np.float32)
    mol_id_to_idx = {str(mid): i for i, mid in enumerate(train_mols["molecule_id"])}

    del morgan_fps, maccs_fps, heavy_atoms, ring_counts, elem_presence, elem_counts, train_mols
    gc.collect()

    pw_path = find_file("spec2fp_pos_weights.npy", hint="spec2fp", required=False)
    if pw_path is not None and Path(pw_path).is_file():
        pos_weight = np.load(pw_path).astype(np.float32)
        print(f"Loaded Stage 05 pos_weight from {pw_path}: mean={pos_weight.mean():.2f}")
    else:
        pos_freq = morgan_arr.mean(axis=0, dtype=np.float64)
        neg_freq = 1.0 - pos_freq
        pos_weight = np.clip(np.sqrt(neg_freq / (pos_freq + 1e-6)), 1.0, 10.0).astype(np.float32)
        print(f"Computed pos_weight from train targets: mean={pos_weight.mean():.2f}")
    """)

    # ── 7. Dataset & Augmentation (Cell 31: c5410b3e) ─────────────────────
    md("## 7. Fine-Tuning Dataset (Reduced Augmentation Noise)")

    # Cell 32: 7cdab131
    code(r"""
    class SpecFPDataset(Dataset):
        def __init__(self, df, mol_id_to_idx, morgan_arr, maccs_arr, heavy_arr,
                     ring_arr, epres_arr, ecount_arr, cfg, augment=False):
            self.cfg = cfg
            self.augment = augment
            self.morgan = morgan_arr
            self.maccs = maccs_arr
            self.heavy = heavy_arr
            self.ring = ring_arr
            self.epres = epres_arr
            self.ecount = ecount_arr

            self.mzs = df["mz"].to_numpy()
            self.intens = df["inten"].to_numpy()
            self.prec_mzs = df["precursor_mz"].to_numpy(dtype=np.float32)
            self.ces = pd.to_numeric(df["ce"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            adduct_col = df["adduct"] if "adduct" in df.columns else ["[M+H]+"] * len(df)
            self.adduct_idxs = np.array([spec2fp.ADDUCT2IDX.get(str(x).strip(), spec2fp.ADDUCT2IDX["<unk>"]) for x in adduct_col], dtype=np.int64)
            inst_col = df["instrument"] if "instrument" in df.columns else ["unknown"] * len(df)
            self.inst_idxs = np.array([spec2fp.INST2IDX.get(str(x).strip(), spec2fp.INST2IDX["<unk>"]) for x in inst_col], dtype=np.int64)
            pol_col = df["polarity"] if "polarity" in df.columns else ["positive"] * len(df)
            self.pol_idxs = np.array([spec2fp.POLARITY2IDX.get(str(x).strip().lower(), spec2fp.POLARITY2IDX["<unk>"]) for x in pol_col], dtype=np.int64)
            self.target_idxs = np.array([mol_id_to_idx.get(str(m), 0) for m in df["molecule_id"]], dtype=np.int32)

            if self.augment:
                self.order = np.argsort(self.target_idxs).astype(np.int32)
                sorted_targets = self.target_idxs[self.order]
                self.bounds = np.searchsorted(sorted_targets, np.arange(len(morgan_arr) + 1)).astype(np.int32)

        def __len__(self):
            return len(self.target_idxs)
    """)

    # Cell 33: 69a68cab
    code(r"""
    def spec_getitem(self, idx):
        mz = spec2fp.parse_peak_array(self.mzs[idx])
        inten = spec2fp.parse_peak_array(self.intens[idx])
        prec_mz = float(self.prec_mzs[idx])
        adduct_idx = int(self.adduct_idxs[idx])
        ce = float(self.ces[idx])
        inst_idx = int(self.inst_idxs[idx])
        pol_idx = int(self.pol_idxs[idx])
        tidx = int(self.target_idxs[idx])

        if self.augment:
            b_start = int(self.bounds[tidx])
            b_end = int(self.bounds[tidx + 1])
            if (b_end - b_start) > 1 and np.random.rand() < 0.2:
                other_idx = int(self.order[np.random.randint(b_start, b_end)])
                mz2 = spec2fp.parse_peak_array(self.mzs[other_idx])
                int2 = spec2fp.parse_peak_array(self.intens[other_idx])
                if len(mz2) > 0 and len(int2) > 0:
                    mz = np.concatenate([mz, mz2])
                    inten = np.concatenate([inten, int2])
            if len(mz) > 1:
                keep = np.random.rand(len(mz)) > self.cfg.AUG_PEAK_DROPOUT
                if keep.sum() >= 1:
                    mz, inten = mz[keep], inten[keep]
                noise = np.exp(np.random.randn(len(inten)) * self.cfg.AUG_INTENSITY_NOISE)
                inten = inten * noise
                mz = mz + np.random.randn(len(mz)) * self.cfg.AUG_MZ_JITTER

        item = spec2fp.tensorize_spectrum(
            mz, inten, prec_mz, adduct_idx, ce, inst_idx, pol_idx,
            top_k=self.cfg.TOP_K_PEAKS, mz_dim=self.cfg.MZ_DIM)
        item["morgan_fp"] = self.morgan[tidx]
        item["maccs_fp"] = self.maccs[tidx]
        item["heavy_atoms"] = float(self.heavy[tidx])
        item["ring_count"] = float(self.ring[tidx])
        item["elem_presence"] = self.epres[tidx]
        item["elem_count"] = self.ecount[tidx]
        return item

    SpecFPDataset.__getitem__ = spec_getitem
    """)

    # Cell 34: b4fa50ab
    code(r"""
    def collate_train_batch(batch):
        B = len(batch)
        top_k = cfg.TOP_K_PEAKS
        feat_dim = cfg.FEAT_DIM
        max_len = top_k + 3
        peaks = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
        mask = torch.ones(B, max_len, dtype=torch.bool)
        adducts = torch.zeros(B, dtype=torch.long)
        instruments = torch.zeros(B, dtype=torch.long)
        polarities = torch.zeros(B, dtype=torch.long)
        ces = torch.zeros(B, dtype=torch.float32)
        prec_mzs = torch.zeros(B, dtype=torch.float32)
        morgan_t = torch.zeros(B, cfg.FP_BITS, dtype=torch.float32)
        maccs_t = torch.zeros(B, cfg.MACCS_BITS, dtype=torch.float32)
        heavy_t = torch.zeros(B, dtype=torch.float32)
        ring_t = torch.zeros(B, dtype=torch.float32)
        epres_t = torch.zeros(B, len(ELEMENTS_ORDER), dtype=torch.float32)
        ecount_t = torch.zeros(B, len(ELEMENTS_ORDER), dtype=torch.float32)

        for i, item in enumerate(batch):
            n = item["n_peaks"]
            peaks[i, :n] = torch.from_numpy(item["peak_feat"][:n])
            mask[i, :n] = False
            adducts[i] = item["adduct_idx"]
            instruments[i] = item["inst_idx"]
            polarities[i] = item["pol_idx"]
            ces[i] = float(item["ce_val"])
            prec_mzs[i] = float(item["prec_mz"])
            morgan_t[i] = torch.from_numpy(item["morgan_fp"]).to(torch.float32)
            maccs_t[i] = torch.from_numpy(item["maccs_fp"]).to(torch.float32)
            heavy_t[i] = float(item["heavy_atoms"])
            ring_t[i] = float(item["ring_count"])
            epres_t[i] = torch.from_numpy(item["elem_presence"]).to(torch.float32)
            ecount_t[i] = torch.from_numpy(item["elem_count"]).to(torch.float32)

        return {
            "peaks": peaks, "pad_mask": mask,
            "adduct": adducts, "instrument": instruments, "polarity": polarities,
            "ce": ces, "prec_mz": prec_mzs,
            "morgan_fp": morgan_t, "maccs_fp": maccs_t,
            "heavy_atoms": heavy_t, "ring_count": ring_t,
            "elem_presence": epres_t, "elem_count": ecount_t,
        }
    """)

    # ── 8. Model & Checkpoint Loading (Cell 35: c15a7a46) ─────────────────
    md("## 8. Warm-Start from Stage 05 Checkpoint")

    # Cell 36: 6004fce3
    code(r"""
    cfg_file = find_file("spec2fp_config.json", hint="spec2fp", required=False)
    if cfg_file is not None and Path(cfg_file).is_file():
        try:
            saved_cfg = json.loads(Path(cfg_file).read_text())
            model_cfg = saved_cfg.get("model", {})
            print(f"Loaded architecture from {cfg_file}: d_model={model_cfg.get('d_model')}, layers={model_cfg.get('n_layers')}")
        except Exception as e:
            print(f"Notice reading config: {e}")
            model_cfg = {}
    else:
        model_cfg = {}

    if not model_cfg:
        model_cfg = {
            "d_model": cfg.D_MODEL, "n_layers": cfg.N_LAYERS, "n_heads": cfg.N_HEADS,
            "dropout": cfg.DROPOUT, "feat_dim": cfg.FEAT_DIM,
            "fp_bits": cfg.FP_BITS, "maccs_bits": cfg.MACCS_BITS,
            "n_elements": len(ELEMENTS_ORDER),
        }

    base_model = spec2fp.Spec2FPModel(model_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Spec2FPModel architecture instantiated with {n_params:,} parameters")
    """)

    # Cell 37: 384f8b3c
    code(r"""
    ckpt_file = find_file("spec2fp_model.pt", hint="spec2fp", required=not cfg.SMOKE_TEST)
    initial_best_mrr = 0.7022
    if ckpt_file is not None and Path(ckpt_file).is_file():
        sd = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
        state = sd.get("ema_state_dict", sd.get("model_state_dict", sd))
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(clean_state)
        initial_best_mrr = float(sd.get("best_proxy_mrr", 0.7022))
        print(f"Warm-started weights from {ckpt_file} (prior best MRR: {initial_best_mrr:.4f}, epoch: {sd.get('epoch', '?')})")
    else:
        print("SMOKE_TEST notice: No pre-trained checkpoint found; initializing fresh weights.")

    if N_GPU > 1:
        model = nn.DataParallel(base_model)
        print(f"Fine-tuning wrapped in DataParallel across {N_GPU} GPUs")
    else:
        model = base_model
    """)

    # Cell 38: 051ace2d
    code(r"""
    class EMA:
        def __init__(self, model, decay=0.9995):
            self.decay = decay
            self.shadow = {}
            self.backup = {}
            src = model.module if hasattr(model, "module") else model
            for name, param in src.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

        def update(self, model):
            src = model.module if hasattr(model, "module") else model
            for name, param in src.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

        def apply_shadow(self, model):
            src = model.module if hasattr(model, "module") else model
            for name, param in src.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

        def restore(self, model):
            src = model.module if hasattr(model, "module") else model
            for name, param in src.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
            self.backup = {}

        def state_dict(self):
            return {k: v.clone() for k, v in self.shadow.items()}
    """)

    # ── 9. Loss & Schedulers (Cell 39: d77b529d) ──────────────────────────
    md("## 9. Schedulers, Ranking-Aligned Hybrid Loss & Evaluation")

    # Cell 40: 54595f66
    code(r"""
    train_split = train_df[train_df["split3"] == "train"].reset_index(drop=True)
    val_model_split = train_df[train_df["split3"] == "val_model"].reset_index(drop=True)
    val_rank_split = train_df[train_df["split3"] == "val_rank"].reset_index(drop=True)
    del train_df
    gc.collect()

    train_mols_unique = train_split["molecule_id"].unique()
    train_ref_target_idxs = np.array([mol_id_to_idx[str(m)] for m in train_mols_unique], dtype=np.int32)
    train_ref_masses = train_split.groupby("molecule_id")["neutral_mass"].first().loc[train_mols_unique].to_numpy(dtype=np.float32)

    ds_train = SpecFPDataset(train_split, mol_id_to_idx, morgan_arr, maccs_arr,
                             heavy_arr, ring_arr, epres_arr, ecount_arr, cfg, augment=True)
    del train_split
    gc.collect()

    ds_val = SpecFPDataset(val_model_split, mol_id_to_idx, morgan_arr, maccs_arr,
                           heavy_arr, ring_arr, epres_arr, ecount_arr, cfg, augment=False)

    dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True,
                          collate_fn=collate_train_batch, drop_last=len(ds_train) > cfg.BATCH_SIZE)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True,
                        collate_fn=collate_train_batch)
    print(f"DataLoaders ready: train={len(dl_train)} batches, val={len(dl_val)} batches (num_workers=0, zero CoW)")
    """)

    # Cell 41: fa64e0ac
    code(r"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = max(len(dl_train), 1)
    max_epochs = cfg.EPOCHS
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * cfg.WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.01, float(step) / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.MIN_LR / cfg.LR
        return min_ratio + (1.0 - min_ratio) * decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    dev_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
    scaler = GradScaler("cuda", enabled="cuda" in str(DEVICE))
    pw_tensor = torch.from_numpy(pos_weight).to(DEVICE)
    print(f"Fine-tuning schedule: {max_epochs} epochs ({total_steps} steps), LR={cfg.LR}->{cfg.MIN_LR}, warmup={warmup_steps}")
    """)

    # Cell 42: 4ee70ad7
    code(r"""
    def compute_loss(out, batch, pw_tensor):
        dev = out["fp_logits"].device
        morgan_smooth = batch["morgan_fp"].to(dev) * 0.995 + 0.0025
        maccs_smooth = batch["maccs_fp"].to(dev) * 0.995 + 0.0025
        pw_dev = pw_tensor.to(dev)
        fp_loss = F.binary_cross_entropy_with_logits(
            out["fp_logits"], morgan_smooth, pos_weight=pw_dev, reduction="mean")
        maccs_loss = F.binary_cross_entropy_with_logits(
            out["maccs_logits"], maccs_smooth, reduction="mean")
        heavy_loss = F.smooth_l1_loss(out["heavy_atoms"], batch["heavy_atoms"].to(dev))
        ring_loss = F.smooth_l1_loss(out["ring_count"], batch["ring_count"].to(dev))
        epres_loss = F.binary_cross_entropy_with_logits(
            out["elem_presence"], batch["elem_presence"].to(dev), reduction="mean")
        ecount_loss = F.smooth_l1_loss(out["elem_count"], batch["elem_count"].to(dev))
        bce_multi_loss = (fp_loss + 0.5 * maccs_loss + 0.01 * heavy_loss
                          + 0.02 * ring_loss + 0.2 * epres_loss + 0.02 * ecount_loss)

        cal_logits = out["fp_logits"] - 0.5 * torch.log(pw_dev.clamp(min=1.0))
        pred_p = torch.sigmoid(cal_logits)
        true_fp = batch["morgan_fp"].to(dev)
        cos_loss = 1.0 - F.cosine_similarity(pred_p, true_fp, dim=1, eps=1e-6).mean()
        inter = (pred_p * true_fp).sum(dim=1)
        union = (pred_p * pred_p).sum(dim=1) + true_fp.sum(dim=1) - inter
        tan_loss = (1.0 - (inter + 1e-5) / (union + 1e-5)).mean()

        total = bce_multi_loss + cfg.COS_LOSS_WEIGHT * cos_loss + cfg.TAN_LOSS_WEIGHT * tan_loss
        return total, fp_loss.item()
    """)

    # Cell 43: 668af433
    code(r"""
    @torch.no_grad()
    def evaluate_val_sim(model, dl, pw_tensor):
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()
        total_loss, n_batches = 0.0, 0
        cos_sum, cos_cnt = 0.0, 0
        tan_sum, tan_cnt = 0.0, 0
        dev_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
        for batch in dl:
            b_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()
                     if k in ("peaks","pad_mask","adduct","instrument","polarity","ce","prec_mz")}
            with autocast(device_type=dev_type, dtype=torch.float16, enabled="cuda" in str(DEVICE)):
                out = raw_model(**b_dev)
            loss, _ = compute_loss(out, batch, pw_tensor)
            total_loss += loss.item()
            n_batches += 1

            if pw_tensor is not None:
                cal_logits = out["fp_logits"] - 0.5 * torch.log(pw_tensor.clamp(min=1.0))
                pred_prob = torch.sigmoid(cal_logits).float()
            else:
                pred_prob = torch.sigmoid(out["fp_logits"]).float()
            true_fp = batch["morgan_fp"].to(DEVICE)
            dot = (pred_prob * true_fp).sum(dim=1)
            norms = torch.norm(pred_prob, dim=1) * torch.norm(true_fp, dim=1)
            valid = norms > 0
            if valid.any():
                cos_sum += (dot[valid] / norms[valid]).sum().item()
                cos_cnt += int(valid.sum().item())
            pred_bin = (pred_prob > 0.5).float()
            inter = (pred_bin * true_fp).sum(dim=1)
            union = pred_bin.sum(dim=1) + true_fp.sum(dim=1) - inter
            tan_sum += (inter / torch.clamp(union, min=1.0)).sum().item()
            tan_cnt += len(pred_bin)

        mean_loss = total_loss / max(n_batches, 1)
        mean_cos = cos_sum / max(cos_cnt, 1)
        mean_tan = tan_sum / max(tan_cnt, 1)
        return mean_loss, mean_cos, mean_tan
    """)

    # ── 10. Retrieval & Epoch Helpers (Cell 44: 432b12fb) ─────────────────
    md("## 10. Proxy MRR Retrieval with Combined Morgan + MACCS Scoring")

    # Cell 45: d73a4d07
    code(r"""
    @torch.no_grad()
    def compute_val_proxy_mrr(model, val_split, max_eval_mols=None):
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()
        eval_df = val_split
        if max_eval_mols is not None:
            u_mols = val_split["molecule_id"].unique()
            if len(u_mols) > max_eval_mols:
                rng = np.random.RandomState(42)
                sel_mols = set(rng.choice(u_mols, size=max_eval_mols, replace=False))
                eval_df = val_split[val_split["molecule_id"].isin(sel_mols)].reset_index(drop=True)
        fp_probs, maccs_probs, _, mol_ids = spec2fp.predict_full(
            raw_model, eval_df, device=DEVICE, batch_size=cfg.PRED_BATCH_SIZE,
            top_k=cfg.TOP_K_PEAKS, mz_dim=cfg.MZ_DIM, pos_weights=pw_tensor)
        val_mass_map = eval_df.groupby("molecule_id")["neutral_mass"].first().to_dict()
        rrs, top1_hits, top25_hits = [], 0, 0
        for i, mid in enumerate(mol_ids):
            q_fp, q_maccs = fp_probs[i].astype(np.float32), maccs_probs[i].astype(np.float32)
            t_idx = mol_id_to_idx.get(str(mid))
            if t_idx is None:
                continue
            true_fp = morgan_arr[t_idx].astype(np.float32)
            true_maccs = maccs_arr[t_idx].astype(np.float32)
            q_mass = float(val_mass_map.get(mid, 0.0))
            mass_diff = np.abs(train_ref_masses - q_mass)
            cand_mask = (mass_diff <= (50.0 * 1e-6 * max(q_mass, 1.0)))
            cand_idx = np.where(cand_mask)[0] if cand_mask.sum() >= 25 else np.argsort(mass_diff)[:min(len(mass_diff), 50)]
            cand_t_idxs = train_ref_target_idxs[cand_idx]

            cand_morg = np.vstack([morgan_arr[cand_t_idxs].astype(np.float32), true_fp[None, :]])
            cand_mac = np.vstack([maccs_arr[cand_t_idxs].astype(np.float32), true_maccs[None, :]])
            norms_m = np.linalg.norm(cand_morg, axis=1) * max(np.linalg.norm(q_fp), 1e-8)
            cos_m = (cand_morg @ q_fp) / np.maximum(norms_m, 1e-8)
            norms_mac = np.linalg.norm(cand_mac, axis=1) * max(np.linalg.norm(q_maccs), 1e-8)
            cos_mac = (cand_mac @ q_maccs) / np.maximum(norms_mac, 1e-8)

            comb_scores = 0.85 * cos_m + 0.15 * cos_mac
            true_score = comb_scores[-1]
            rank = int((comb_scores >= true_score - 1e-8).sum())
            rr = 1.0 / rank if rank <= 25 else 0.0
            rrs.append(rr)
            if rank == 1: top1_hits += 1
            if rank <= 25: top25_hits += 1
        n = max(len(rrs), 1)
        return float(np.mean(rrs)) if rrs else 0.0, top1_hits / n, top25_hits / n
    """)

    # Cell 46: 0f51a69b
    code(r"""
    def train_one_epoch(model, dl, opt, sch, scl, ema, pw_tensor):
        model.train()
        ep_loss, n_steps = 0.0, 0
        t0 = time.perf_counter()
        dev_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
        for batch in tqdm(dl, desc="FineTune Step", leave=False):
            b_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()
                     if k in ("peaks","pad_mask","adduct","instrument","polarity","ce","prec_mz")}
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=dev_type, dtype=torch.float16, enabled="cuda" in str(DEVICE)):
                out = model(**b_dev)
            loss, _ = compute_loss(out, batch, pw_tensor)
            scl.scale(loss).backward()
            scl.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scl.step(opt)
            scl.update()
            sch.step()
            ema.update(model)
            ep_loss += loss.item()
            n_steps += 1
        return ep_loss / max(n_steps, 1), time.perf_counter() - t0
    """)

    # ── 11. Main Training Loop (Cell 47: 564eec83) ────────────────────────
    md("## 11. Main Fine-Tuning Loop")

    # Cell 48: 1034eb36
    code(r"""
    root = Path(cfg.OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    ema = EMA(model, decay=cfg.EMA_DECAY)

    # Measure baseline directly on the 3000-eval subset
    print("Evaluating warm-start baseline on validation sample...")
    initial_sample_mrr, _, _ = compute_val_proxy_mrr(model, val_model_split, max_eval_mols=3000)
    best_proxy_mrr = initial_sample_mrr
    best_epoch = 0
    patience_cnt = 0
    best_ema_sd = copy.deepcopy(ema.state_dict())
    epoch_times = []
    budget_s = cfg.TRAIN_BUDGET_H * 3600
    history = {"train_loss": [], "val_loss": [], "val_cos": [], "val_tan": [],
               "proxy_mrr": [], "top1": [], "top25": [], "lr": []}
    print(f"Warm-start sample baseline MRR@25 = {initial_sample_mrr:.4f} (Full validation reference: {initial_best_mrr:.4f})")
    """)

    # Cell 49: 1326bb0b
    code(r"""
    for epoch in range(max_epochs):
        ep_loss, ep_time = train_one_epoch(model, dl_train, optimizer, scheduler, scaler, ema, pw_tensor)
        epoch_times.append(ep_time)
        ema.apply_shadow(model)
        v_loss, v_cos, v_tan = evaluate_val_sim(model, dl_val, pw_tensor)
        v_mrr, v_top1, v_top25 = compute_val_proxy_mrr(model, val_model_split, max_eval_mols=3000)
        ema.restore(model)

        cur_lr = optimizer.param_groups[0]["lr"]
        for k, v in [("train_loss", ep_loss), ("val_loss", v_loss), ("val_cos", v_cos),
                     ("val_tan", v_tan), ("proxy_mrr", v_mrr), ("top1", v_top1),
                     ("top25", v_top25), ("lr", cur_lr)]:
            history[k].append(v)

        print(f"FineTune Ep {epoch+1:02d}/{max_epochs}: loss={ep_loss:.4f} val_loss={v_loss:.4f} cos={v_cos:.4f} "
              f"proxy_MRR={v_mrr:.4f} (baseline={initial_sample_mrr:.4f}) top1={v_top1:.2f} top25={v_top25:.2f} ({ep_time:.1f}s)")

        if v_mrr > best_proxy_mrr:
            best_proxy_mrr = v_mrr
            best_epoch = epoch + 1
            patience_cnt = 0
            best_ema_sd = copy.deepcopy(ema.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch})")
                break
        if (time.perf_counter() - T_START) > (budget_s * 0.95):
            print(f"Wall-clock budget approaching limit. Stopping at epoch {epoch+1}.")
            break

    print(f"Fine-tuning complete. Best epoch: {best_epoch} with proxy MRR@25 = {best_proxy_mrr:.4f}")
    """)

    # ── 12. Save Checkpoint (Cell 50: 891b4682) ───────────────────────────
    md("## 12. Save Fine-Tuned Checkpoint & Config")

    # Cell 51: 14a47ea2
    code(r"""
    save_dict = {"ema_state_dict": best_ema_sd, "epoch": best_epoch, "best_proxy_mrr": best_proxy_mrr}
    torch.save(save_dict, root / "spec2fp_model.pt")
    full_cfg = {
        "model": model_cfg, "training": asdict(cfg),
        "fp_spec": evalkit.FP_SPEC, "elements_order": ELEMENTS_ORDER,
        "adduct_vocab": spec2fp.ADDUCT_VOCAB, "inst_vocab": spec2fp.INST_VOCAB,
    }
    (root / "spec2fp_config.json").write_text(json.dumps(full_cfg, indent=2))
    np.save(root / "spec2fp_pos_weights.npy", pos_weight)
    print(f"Saved fine-tuned spec2fp_model.pt ({(root / 'spec2fp_model.pt').stat().st_size / 1e6:.1f} MB)")
    print(f"Saved spec2fp_config.json and spec2fp_pos_weights.npy")
    """)

    # ── 13. Predictions (Cell 52: 2f442292) ───────────────────────────────
    md("## 13. Generate Both 4096-Bit Morgan AND 167-Bit MACCS Predictions")

    # Cell 53: 195c77e6
    code(r"""
    raw_eval_model = model.module if hasattr(model, "module") else model
    for name, param in raw_eval_model.named_parameters():
        if name in best_ema_sd:
            param.data.copy_(best_ema_sd[name])
    raw_eval_model.eval()

    def generate_and_save_split(split_df, split_name):
        fp_probs, maccs_probs, elem_pred, mol_ids = spec2fp.predict_full(
            raw_eval_model, split_df, device=DEVICE, batch_size=cfg.PRED_BATCH_SIZE,
            top_k=cfg.TOP_K_PEAKS, mz_dim=cfg.MZ_DIM, pos_weights=pos_weight)
        np.save(root / f"fp_pred_{split_name}.npy", fp_probs)
        np.save(root / f"maccs_pred_{split_name}.npy", maccs_probs)
        np.save(root / f"elem_pred_{split_name}.npy", elem_pred)
        pd.DataFrame({"molecule_id": mol_ids}).to_parquet(root / f"fp_pred_ids_{split_name}.parquet")
        print(f"[{split_name}] Saved fp={fp_probs.shape}, maccs={maccs_probs.shape}, elem={elem_pred.shape}, ids={len(mol_ids)}")
        return fp_probs, maccs_probs, elem_pred, mol_ids
    """)

    # Cell 54: 882672be
    code(r"""
    fp_vm, mac_vm, elem_vm, ids_vm = generate_and_save_split(val_model_split, "val_model")
    fp_vr, mac_vr, elem_vr, ids_vr = generate_and_save_split(val_rank_split, "val_rank")
    fp_te, mac_te, elem_te, ids_te = generate_and_save_split(test_df, "test")
    del fp_vm, mac_vm, elem_vm, ids_vm, fp_vr, mac_vr, elem_vr, ids_vr, fp_te, mac_te, elem_te, ids_te; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    """)

    # ── 14. Metrics & Plots (Cell 55: 7b69d6d6) ───────────────────────────
    md("## 14. Validation Metrics & Diagnostic Curves")

    # Cell 56: a5685a5d
    code(r"""
    final_v_loss, final_v_cos, final_v_tan = evaluate_val_sim(model, dl_val, pw_tensor)
    final_v_mrr, final_v_top1, final_v_top25 = compute_val_proxy_mrr(model, val_model_split, max_eval_mols=None)

    val_metrics = {
        "val_loss": float(final_v_loss),
        "val_cosine": float(final_v_cos),
        "val_tanimoto": float(final_v_tan),
        "proxy_mrr_at_25": float(final_v_mrr),
        "proxy_top1_acc": float(final_v_top1),
        "proxy_top25_acc": float(final_v_top25),
        "best_epoch": int(best_epoch),
        "total_epochs": len(history["train_loss"]),
        "total_params": int(n_params),
        "train_time_s": float(sum(epoch_times)),
        "warm_start_prior_mrr": float(initial_best_mrr),
    }
    (root / "spec2fp_val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"Validation summary: {json.dumps(val_metrics, indent=2)}")
    """)

    # Cell 57: 3232d7cd
    code(r"""
    if HAS_MPL and len(history["train_loss"]) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(history["train_loss"], label="Train")
        axes[0, 0].plot(history["val_loss"], label="Val")
        axes[0, 0].set_title("Hybrid Multi-Task Loss"); axes[0, 0].legend()
        axes[0, 1].plot(history["val_cos"], label="Cosine")
        axes[0, 1].plot(history["val_tan"], label="Tanimoto")
        axes[0, 1].set_title("Val Fingerprint Similarity"); axes[0, 1].legend()
        axes[1, 0].plot(history["proxy_mrr"], label="Proxy MRR@25", color="green")
        axes[1, 0].axhline(initial_best_mrr, label="Stage 05 Baseline", linestyle=":", color="red")
        axes[1, 0].plot(history["top25"], label="Top-25 Acc", linestyle="--", color="blue")
        axes[1, 0].set_title("Proxy Retrieval Ranking"); axes[1, 0].legend()
        axes[1, 1].plot(history["lr"], color="purple")
        axes[1, 1].set_title("Cosine Annealing LR")
        plt.tight_layout()
        plt.savefig(root / "spec2fp_training_curve.png", dpi=150)
        plt.close()
        print("Diagnostic curve saved to spec2fp_training_curve.png")
    """)

    # ── 15. FULL_TRAIN (Cell 58: dd239d92) ─────────────────────────────────
    md("## 15. Optional Full Dataset Retraining")

    # Cell 59: ccadf1ad
    code(r"""
    if cfg.FULL_TRAIN:
        print(f"FULL_TRAIN=True: Fine-tuning full train data from {train_path} for {max(best_epoch, 1)} epochs...")
        full_train_df = pd.read_parquet(train_path).reset_index(drop=True)
        ds_full = SpecFPDataset(full_train_df, mol_id_to_idx,
                                morgan_arr, maccs_arr, heavy_arr, ring_arr,
                                epres_arr, ecount_arr, cfg, augment=True)
        del full_train_df; gc.collect()
        dl_full = DataLoader(ds_full, batch_size=cfg.BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=True, persistent_workers=False,
                             collate_fn=collate_train_batch)
        full_base = spec2fp.Spec2FPModel(model_cfg).to(DEVICE)
        full_model = nn.DataParallel(full_base) if N_GPU > 1 else full_base
        opt_f = torch.optim.AdamW(full_model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        sch_f = torch.optim.lr_scheduler.LambdaLR(opt_f, lr_lambda)
        scl_f = GradScaler("cuda", enabled="cuda" in str(DEVICE))
        ema_f = EMA(full_model, decay=cfg.EMA_DECAY)
        for ep in range(max(best_epoch, 1)):
            train_one_epoch(full_model, dl_full, opt_f, sch_f, scl_f, ema_f, pw_tensor)
        ema_f.apply_shadow(full_model)
        torch.save({"ema_state_dict": ema_f.state_dict(), "epoch": best_epoch, "mode": "full_train"},
                   root / "spec2fp_model_full.pt")
        print("Full training model checkpoint saved as spec2fp_model_full.pt")
        del full_model, full_base, opt_f, sch_f, scl_f, ema_f, ds_full, dl_full; gc.collect()
    else:
        print("FULL_TRAIN=False. Preserving default split3 train-only model (fold0) for stage 09 consistency.")
    """)

    # ── 16. Manifest (Cell 60: e73da7d1) ──────────────────────────────────
    md("## 16. Manifest Contract & Sanity Verifications")

    # Cell 61: cf25f579
    code(r"""
    elapsed = time.perf_counter() - T_START
    vm_ids = val_model_split["molecule_id"].unique().tolist()
    vr_ids = val_rank_split["molecule_id"].unique().tolist()
    heldout_hash = hashlib.sha256("".join(sorted(vm_ids + vr_ids)).encode()).hexdigest()

    manifest = {
        "stage": "spec2fp",
        "stage_alias": "spec2fp_finetune",
        "created": datetime.now(timezone.utc).isoformat(),
        "cfg": asdict(cfg),
        "lib_versions": {
            "python": sys.version.split()[0], "rdkit": Chem.rdBase.rdkitVersion,
            "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
        },
        "metrics": val_metrics,
        "trained_on_split": "train",
        "heldout_ids_hash": heldout_hash,
        "fp_spec": evalkit.FP_SPEC,
        "model_params": n_params,
        "test_model": cfg.TEST_MODEL,
        "full_train": cfg.FULL_TRAIN,
        "runtime_seconds": elapsed,
    }
    manifest["files"] = {p.name: p.stat().st_size for p in sorted(root.iterdir()) if p.is_file()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written with {len(manifest['files'])} artifacts; elapsed={elapsed/60:.1f} min")
    """)

    # Cell 62: c97d12a0
    code(r"""
    required_artifacts = [
        "spec2fp_model.pt", "spec2fp_config.json", "spec2fp_pos_weights.npy",
        "spec2fp_val_metrics.json", "fp_pred_val_model.npy", "fp_pred_val_rank.npy",
        "fp_pred_test.npy", "maccs_pred_val_model.npy", "maccs_pred_val_rank.npy",
        "maccs_pred_test.npy", "fp_pred_ids_val_model.parquet", "fp_pred_ids_val_rank.parquet",
        "fp_pred_ids_test.parquet", "elem_pred_val_model.npy", "elem_pred_val_rank.npy",
        "elem_pred_test.npy", "manifest.json",
    ]
    for art in required_artifacts:
        assert (root / art).exists(), f"Missing required artifact: {art}"

    m_loaded = json.loads((root / "manifest.json").read_text())
    assert m_loaded["trained_on_split"] == "train", "Split contract mismatch in manifest!"
    assert m_loaded["fp_spec"] == evalkit.FP_SPEC, "FP spec contract mismatch!"
    print(f"All {len(required_artifacts)} artifact assertions PASSED successfully!")
    """)

    # ── 17. Module Source (Cell 63: 5c8d6cf0) ─────────────────────────────
    md("## 17. Complete Source of `spec2fp.py`")

    # Cell 64: 99b606d1
    code('print(Path("/kaggle/working/spec2fp.py").read_text(encoding="utf-8"))')

    # ── Validate & Assign Exact Cell IDs ──────────────────────────────────
    if len(NOTEBOOK_CELLS) != len(EXACT_CELL_IDS):
        raise ValueError(f"Cell count mismatch: {len(NOTEBOOK_CELLS)} cells generated, expected {len(EXACT_CELL_IDS)}")

    for cell, cid in zip(NOTEBOOK_CELLS, EXACT_CELL_IDS):
        cell["id"] = cid

    nb = nbf.v4.new_notebook(cells=NOTEBOOK_CELLS)
    nb.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3.10.0"},
    )
    nbf.validate(nb)

    oversize_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            lines = cell.source.splitlines()
            if len(lines) > 59 and not cell.source.startswith("%%writefile evalkit.py"):
                oversize_cells.append((i, len(lines)))

    if oversize_cells:
        raise ValueError(f"Found code cells exceeding 59 lines: {oversize_cells}")

    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    delivery = Path("/mnt/user-data/outputs")
    if delivery.is_dir() and delivery.resolve() != output.parent.resolve():
        nbf.write(nb, delivery / output.name)
    print(f"Created {output}: {len(nb.cells)} cells; nbformat + exact cell IDs + AST validation passed")
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("05b_spec2fp_finetune.ipynb"))
    args = parser.parse_args()
    build_notebook(args.output)
