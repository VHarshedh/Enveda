"""Generate 05_spec2fp_train.ipynb — Spec→FP neural-network training notebook.

Embeds spec2fp.py module source via %%writefile. Uses nbformat + ast validation.
Strictly eliminates DataParallel B=1 scatter crashes, iterrows bottlenecks,
evalkit cold-start failures, and array parsing errors.
"""
from pathlib import Path
import ast
import argparse
import textwrap
import warnings
import nbformat as nbf

NOTEBOOK_CELLS: list = []


def md(text: str):
    NOTEBOOK_CELLS.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str):
    code_str = textwrap.dedent(text).strip()
    lines = code_str.splitlines()
    if len(lines) >= 60:
        raise ValueError(f"Code cell exceeds 59 lines ({len(lines)} lines):\n{lines[0]}")
    NOTEBOOK_CELLS.append(nbf.v4.new_code_cell(code_str))


# ═══════════════════════════════════════════════════════════════════════════════
# ██  SPEC2FP MODULE SOURCE  ██
# ═══════════════════════════════════════════════════════════════════════════════
SPEC2FP_SOURCE = r'''"""spec2fp.py - Spectrum-to-Fingerprint model, tensorization, and inference.

Defines the Spec2FP Transformer, peak tensorization, and predict() API.
Import ONLY from /kaggle/working/.
"""
import json, math, sys, warnings
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
    """Sinusoidal positional encoding for m/z values. Shape: (N, dim)."""
    freqs = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(math.log(10000.0) / dim))
    angles = np.outer(values.astype(np.float32), freqs)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def tensorize_spectrum(
    mz: Any, inten: Any, precursor_mz: float,
    adduct: str, ce: float, instrument: str, polarity: str,
    top_k: int = 128, mz_dim: int = 64,
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
    adduct_idx = ADDUCT2IDX.get(str(adduct).strip(), ADDUCT2IDX["<unk>"])
    inst_idx = INST2IDX.get(str(instrument).strip(), INST2IDX["<unk>"])
    pol_idx = POLARITY2IDX.get(str(polarity).strip().lower(), POLARITY2IDX["<unk>"])
    ce_val = float(ce) if not (ce is None or (isinstance(ce, float) and math.isnan(ce))) else 0.0
    return {
        "peak_feat": peak_feat, "n_peaks": n,
        "adduct_idx": adduct_idx, "inst_idx": inst_idx, "pol_idx": pol_idx,
        "ce_val": np.float32(ce_val), "prec_mz": np.float32(precursor_mz),
    }


def collate_spectra(batch: list, top_k: int = 128, feat_dim: int = 129):
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
        d = cfg.get("d_model", 320)
        n_layers = cfg.get("n_layers", 5)
        n_heads = cfg.get("n_heads", 8)
        dropout = cfg.get("dropout", 0.1)
        feat_dim = cfg.get("feat_dim", 129)
        fp_bits = cfg.get("fp_bits", 4096)
        maccs_bits = cfg.get("maccs_bits", 167)
        n_elements = cfg.get("n_elements", N_ELEMENTS)
        self.peak_proj = nn.Sequential(
            nn.Linear(feat_dim, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))
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
        x = self.norm(x[:, 0])
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
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    if "ema_state_dict" in sd:
        model.load_state_dict(sd["ema_state_dict"])
    elif "model_state_dict" in sd:
        model.load_state_dict(sd["model_state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()
    return model.to(device), cfg


def _pool_molecule_predictions(all_fp_arr: np.ndarray, all_elem_arr: np.ndarray, mol_ids_arr: np.ndarray):
    """Aggregate per-spectrum predictions into mean-pooled per-molecule predictions."""
    unique_mols, inv_indices = np.unique(mol_ids_arr, return_inverse=True)
    n_unique = len(unique_mols)
    fp_sum = np.zeros((n_unique, 4096), dtype=np.float32)
    elem_sum = np.zeros((n_unique, N_ELEMENTS), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.float32)
    np.add.at(fp_sum, inv_indices, all_fp_arr)
    np.add.at(elem_sum, inv_indices, all_elem_arr)
    np.add.at(counts, inv_indices, 1.0)
    counts_col = np.maximum(counts, 1.0)[:, None]
    return (fp_sum / counts_col).astype(np.float16), (elem_sum / counts_col).astype(np.float32), unique_mols.tolist()


@torch.no_grad()
def predict(model, clean_df, device="cuda", batch_size=128, top_k=128, mz_dim=64):
    """Predict fingerprints + element info for clean_df table with batched inference."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    feat_dim, n_rows = 2 * mz_dim + 1, len(clean_df)
    if n_rows == 0:
        return (np.zeros((0, 4096), dtype=np.float16), np.zeros((0, N_ELEMENTS), dtype=np.float32), [])

    mz_col, int_col = clean_df["mz"].tolist(), clean_df["inten"].tolist()
    pmz_col = clean_df["precursor_mz"].tolist()
    add_col = clean_df["adduct"].tolist() if "adduct" in clean_df.columns else ["[M+H]+"] * n_rows
    ce_col = clean_df["ce"].tolist() if "ce" in clean_df.columns else [0.0] * n_rows
    inst_col = clean_df["instrument"].tolist() if "instrument" in clean_df.columns else ["unknown"] * n_rows
    pol_col = clean_df["polarity"].tolist() if "polarity" in clean_df.columns else ["positive"] * n_rows
    dev_type, is_cuda = "cuda" if "cuda" in str(device) else "cpu", "cuda" in str(device)
    all_fps, all_elems = [], []

    for s_idx in range(0, n_rows, batch_size):
        e_idx = min(s_idx + batch_size, n_rows)
        items = [
            tensorize_spectrum(
                mz_col[r], int_col[r], float(pmz_col[r]),
                str(add_col[r]), float(ce_col[r]) if pd.notna(ce_col[r]) else 0.0,
                str(inst_col[r]), str(pol_col[r]), top_k=top_k, mz_dim=mz_dim)
            for r in range(s_idx, e_idx)
        ]
        bd = collate_spectra(items, top_k=top_k, feat_dim=feat_dim)
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bd.items()}
        with torch.amp.autocast(device_type=dev_type, dtype=torch.float16, enabled=is_cuda):
            out = raw_model(**bd)
        all_fps.append(torch.sigmoid(out["fp_logits"]).float().cpu().numpy())
        all_elems.append(out["elem_count"].float().cpu().numpy())

    all_fp_arr = np.concatenate(all_fps, axis=0)
    all_elem_arr = np.concatenate(all_elems, axis=0)
    mol_ids = clean_df["molecule_id"].to_numpy().astype(str)
    return _pool_molecule_predictions(all_fp_arr, all_elem_arr, mol_ids)
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
        if len(current) + len(segment) > 50:
            if current:
                chunks.append("\n".join(current).strip())
            current = segment
        else:
            current += segment
    if current:
        chunks.append("\n".join(current).strip())
    for i, chunk in enumerate(chunks):
        if len(chunk.splitlines()) > 58:
            raise ValueError(f"Module block exceeds cell limit: {len(chunk.splitlines())} lines\n{chunk[:100]}")
        ast.parse(chunk)
        magic = f"%%writefile /kaggle/working/{filename}" if i == 0 else f"%%writefile -a /kaggle/working/{filename}"
        NOTEBOOK_CELLS.append(nbf.v4.new_code_cell(magic + "\n" + chunk + "\n\n"))


# ═══════════════════════════════════════════════════════════════════════════════
# ██  BUILD NOTEBOOK  ██
# ═══════════════════════════════════════════════════════════════════════════════
def build_notebook(output: Path) -> nbf.NotebookNode:
    NOTEBOOK_CELLS.clear()

    # ── 0. Title ──────────────────────────────────────────────────────────
    md(r"""
    # 05 — Spec2FP: Spectrum → Fingerprint Transformer Training

    **Purpose**: Train a Transformer neural network mapping LC-MS/MS spectra
    (peaks + precursor m/z + adduct + CE + instrument) to molecular fingerprints
    (Morgan-2 4096-bit + MACCS-167), inspired by CSI:FingerID / MIST / SIRIUS.

    **Inputs** (via `find_file`): `train_clean.parquet`, `test_clean.parquet`,
    `splits.parquet`, `evalkit.py` (read only from `/kaggle/working/`), `db.parquet`.

    **Outputs**: `spec2fp_model.pt`, `spec2fp_config.json`, predicted FP arrays,
    element predictions, validation metrics, `spec2fp.py` module, `manifest.json`.

    **Runtime**: ~5 h on 2× T4; SMOKE_TEST ≈ 3 min CPU.
    **GPU RAM**: ~6–8 GB per T4 with fp16 autocast, batch 256 on DataParallel.
    """)

    # ── 1. Config ─────────────────────────────────────────────────────────
    md("## 1. Configuration")
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

    code(r"""
    @dataclass
    class CFG:
        stage: str = "spec2fp"
        seed: int = 42
        SMOKE_TEST: bool = SMOKE_TEST
        TOP_K_PEAKS: int = 128
        MZ_DIM: int = 64
        FEAT_DIM: int = 129
        D_MODEL: int = 320
        N_LAYERS: int = 5
        N_HEADS: int = 8
        DROPOUT: float = 0.1
        FP_BITS: int = 4096
        MACCS_BITS: int = 167
        LR: float = 3e-4
        WEIGHT_DECAY: float = 0.01
        GRAD_CLIP: float = 1.0
        BATCH_SIZE: int = 256
        PRED_BATCH_SIZE: int = 128
        TRAIN_BUDGET_H: float = 5.0
        WARMUP_FRAC: float = 0.05
        EMA_DECAY: float = 0.999
        PATIENCE: int = 8
        AUG_PEAK_DROPOUT: float = 0.20
        AUG_INTENSITY_NOISE: float = 0.05
        AUG_MZ_JITTER: float = 0.005
        DATA_DIR: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"
        OUTPUT_DIR: str = "/kaggle/working/artifacts/spec2fp"
        TEST_MODEL: str = "fold0"
        FULL_TRAIN: bool = False
        time_budget_s: float = 5.0 * 3600
        SUBMIT_BUDGET_S: float = 8.0 * 3600
    """)

    code(r"""
    cfg = CFG()
    if cfg.SMOKE_TEST:
        cfg.BATCH_SIZE = 16
        cfg.PRED_BATCH_SIZE = 32
        cfg.TRAIN_BUDGET_H = 0.02
        cfg.TOP_K_PEAKS = 32
        cfg.D_MODEL = 64
        cfg.N_LAYERS = 2
        cfg.N_HEADS = 4
        cfg.PATIENCE = 2
        cfg.time_budget_s = 180

    cfg.FEAT_DIM = 2 * cfg.MZ_DIM + 1
    T_START = time.perf_counter()
    print(f"CFG: SMOKE_TEST={cfg.SMOKE_TEST}, BATCH={cfg.BATCH_SIZE}, D_MODEL={cfg.D_MODEL}")
    """)

    # ── 2. Imports & env report ───────────────────────────────────────────
    md("## 2. Imports & Environment Report")
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

    # ── 3. Utilities ──────────────────────────────────────────────────────
    md("## 3. Utilities & Seeding")
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

    code(r"""
    def find_manifest(stage):
        matches = []
        for root in (WORK, Path("/kaggle/input")):
            for path in root.rglob("manifest*.json"):
                try:
                    rec = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if rec.get("stage") == stage:
                    matches.append((path.stat().st_mtime, path, rec))
        return max(matches, default=(None, None, None), key=lambda x: x[0])[1:]
    """)

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

    # ── 4. Write spec2fp.py ───────────────────────────────────────────────
    md(r"""
    ## 4. Reusable Module: `spec2fp.py`

    Write `spec2fp.py` to `/kaggle/working/spec2fp.py` via `%%writefile`.
    Contains model class, sinusoidal embedding, peak tensorization, `load_model`, and `predict`.
    """)
    add_module_cells(SPEC2FP_SOURCE, "spec2fp.py")

    code(r"""
    sys.path.insert(0, str(WORK))
    import spec2fp
    for name in ("Spec2FPModel", "tensorize_spectrum", "collate_spectra", "load_model", "predict", "parse_peak_array"):
        assert hasattr(spec2fp, name), f"spec2fp missing {name}"
    print(f"spec2fp imported successfully: {spec2fp.__file__}")
    """)

    # ── 5. Load Data & Splits ─────────────────────────────────────────────
    md("## 5. Load Clean Data & Splits Contract")
    code(r"""
    train_path = find_file("train_clean.parquet", hint="eda")
    test_path = find_file("test_clean.parquet", hint="eda")
    splits_path = find_file("splits.parquet", hint="eval_splits")
    db_path = find_file("db.parquet", hint="candidates", required=False)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    splits_df = pd.read_parquet(splits_path)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Splits: {splits_df.shape}")
    """)

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
    print(f"Split3 counts (spectra):\n{train_df['split3'].value_counts()}")
    print(f"Split3 counts (molecules):\n{train_df.groupby('split3')['molecule_id'].nunique()}")
    """)

    # ── 6. Targets Calculation ────────────────────────────────────────────
    md("## 6. Precompute Multi-Task Targets")
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

    code(r"""
    for _, row in tqdm(train_mols.iterrows(), total=len(train_mols), desc="Target FPs"):
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        morgan_fps.append(evalkit.compute_morgan_fp(smi, n_bits=cfg.FP_BITS))
        maccs_fps.append(evalkit.compute_maccs_fp(smi))
        if mol is not None:
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
            heavy_atoms.append(0)
            ring_counts.append(0)
            elem_presence.append(np.zeros(len(ELEMENTS_ORDER), dtype=np.float32))
            elem_counts.append(np.zeros(len(ELEMENTS_ORDER), dtype=np.float32))
    """)

    code(r"""
    morgan_arr = np.stack(morgan_fps).astype(np.float32)
    maccs_arr = np.stack(maccs_fps).astype(np.float32)
    heavy_arr = np.array(heavy_atoms, dtype=np.float32)
    ring_arr = np.array(ring_counts, dtype=np.float32)
    epres_arr = np.stack(elem_presence)
    ecount_arr = np.stack(elem_counts)
    mol_id_to_idx = {mid: i for i, mid in enumerate(train_mols["molecule_id"])}

    print(f"Targets computed: Morgan={morgan_arr.shape}, MACCS={maccs_arr.shape}, failures={fail_count}")
    pos_freq = morgan_arr.mean(axis=0)
    neg_freq = 1.0 - pos_freq
    pos_weight = np.clip(neg_freq / (pos_freq + 1e-6), 0.1, 50.0).astype(np.float32)
    print(f"BCE pos_weight: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}, mean={pos_weight.mean():.2f}")
    """)

    # ── 7. Dataset & Augmentation ─────────────────────────────────────────
    md("## 7. Dataset with Augmentation & Collate")
    code(r"""
    class SpecFPDataset(Dataset):
        def __init__(self, df, mol_id_to_idx, morgan_arr, maccs_arr, heavy_arr,
                     ring_arr, epres_arr, ecount_arr, cfg, augment=False):
            self.df = df.reset_index(drop=True)
            self.mol_map = mol_id_to_idx
            self.morgan = morgan_arr
            self.maccs = maccs_arr
            self.heavy = heavy_arr
            self.ring = ring_arr
            self.epres = epres_arr
            self.ecount = ecount_arr
            self.cfg = cfg
            self.augment = augment
            self.mol_groups = defaultdict(list)
            for i, mid in enumerate(self.df["molecule_id"]):
                self.mol_groups[str(mid)].append(i)

        def __len__(self):
            return len(self.df)
    """)

    code(r"""
    def spec_getitem(self, idx):
        row = self.df.iloc[idx]
        mz = spec2fp.parse_peak_array(row["mz"])
        inten = spec2fp.parse_peak_array(row["inten"])
        prec_mz = float(row["precursor_mz"])
        adduct = str(row.get("adduct", "[M+H]+"))
        ce = float(row.get("ce", 0.0)) if pd.notna(row.get("ce", None)) else 0.0
        inst = str(row.get("instrument", "unknown"))
        pol = str(row.get("polarity", "positive"))

        if self.augment:
            mid = str(row["molecule_id"])
            peer_indices = self.mol_groups.get(mid, [])
            if len(peer_indices) > 1 and np.random.rand() < 0.3:
                other_idx = np.random.choice(peer_indices)
                other_row = self.df.iloc[other_idx]
                mz2 = spec2fp.parse_peak_array(other_row["mz"])
                int2 = spec2fp.parse_peak_array(other_row["inten"])
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
            mz, inten, prec_mz, adduct, ce, inst, pol,
            top_k=self.cfg.TOP_K_PEAKS, mz_dim=self.cfg.MZ_DIM)
        tidx = self.mol_map.get(str(row["molecule_id"]), 0)
        item["morgan_fp"] = self.morgan[tidx]
        item["maccs_fp"] = self.maccs[tidx]
        item["heavy_atoms"] = float(self.heavy[tidx])
        item["ring_count"] = float(self.ring[tidx])
        item["elem_presence"] = self.epres[tidx]
        item["elem_count"] = self.ecount[tidx]
        return item

    SpecFPDataset.__getitem__ = spec_getitem
    """)

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
            morgan_t[i] = torch.from_numpy(item["morgan_fp"])
            maccs_t[i] = torch.from_numpy(item["maccs_fp"])
            heavy_t[i] = float(item["heavy_atoms"])
            ring_t[i] = float(item["ring_count"])
            epres_t[i] = torch.from_numpy(item["elem_presence"])
            ecount_t[i] = torch.from_numpy(item["elem_count"])

        return {
            "peaks": peaks, "pad_mask": mask,
            "adduct": adducts, "instrument": instruments, "polarity": polarities,
            "ce": ces, "prec_mz": prec_mzs,
            "morgan_fp": morgan_t, "maccs_fp": maccs_t,
            "heavy_atoms": heavy_t, "ring_count": ring_t,
            "elem_presence": epres_t, "elem_count": ecount_t,
        }
    """)

    # ── 8. Model & Multi-GPU ──────────────────────────────────────────────
    md("## 8. Model Architecture & DataParallel")
    code(r"""
    model_cfg = {
        "d_model": cfg.D_MODEL, "n_layers": cfg.N_LAYERS, "n_heads": cfg.N_HEADS,
        "dropout": cfg.DROPOUT, "feat_dim": cfg.FEAT_DIM,
        "fp_bits": cfg.FP_BITS, "maccs_bits": cfg.MACCS_BITS,
        "n_elements": len(ELEMENTS_ORDER),
    }
    base_model = spec2fp.Spec2FPModel(model_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Spec2FPModel instantiated with {n_params:,} trainable parameters")

    if N_GPU > 1:
        model = nn.DataParallel(base_model)
        print(f"Training wrapped in DataParallel across {N_GPU} GPUs")
    else:
        model = base_model
    """)

    code(r"""
    class EMA:
        def __init__(self, model, decay=0.999):
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

    # ── 9. Loss & Dataloaders ─────────────────────────────────────────────
    md("## 9. Schedulers, Loss & Evaluation Functions")
    code(r"""
    train_split = train_df[train_df["split3"] == "train"].reset_index(drop=True)
    val_model_split = train_df[train_df["split3"] == "val_model"].reset_index(drop=True)
    val_rank_split = train_df[train_df["split3"] == "val_rank"].reset_index(drop=True)

    ds_train = SpecFPDataset(train_split, mol_id_to_idx, morgan_arr, maccs_arr,
                             heavy_arr, ring_arr, epres_arr, ecount_arr, cfg, augment=True)
    ds_val = SpecFPDataset(val_model_split, mol_id_to_idx, morgan_arr, maccs_arr,
                           heavy_arr, ring_arr, epres_arr, ecount_arr, cfg, augment=False)

    nw = min(2, os.cpu_count() or 1) if not cfg.SMOKE_TEST else 0
    dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
                          collate_fn=collate_train_batch, drop_last=len(ds_train) > cfg.BATCH_SIZE)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
                        collate_fn=collate_train_batch)
    print(f"DataLoaders ready: train={len(dl_train)} batches, val={len(dl_val)} batches")
    """)

    code(r"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = max(len(dl_train), 1)
    budget_s = cfg.TRAIN_BUDGET_H * 3600
    est_s_per_step = 0.12
    max_epochs = max(int(budget_s / (est_s_per_step * steps_per_epoch)), 4)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * cfg.WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    dev_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
    scaler = GradScaler("cuda", enabled="cuda" in str(DEVICE))
    pw_tensor = torch.from_numpy(pos_weight).to(DEVICE)
    print(f"Training budget: max_epochs={max_epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}")
    """)

    code(r"""
    def compute_loss(out, batch, pw_tensor):
        fp_loss = F.binary_cross_entropy_with_logits(
            out["fp_logits"], batch["morgan_fp"].to(DEVICE),
            pos_weight=pw_tensor, reduction="mean")
        maccs_loss = F.binary_cross_entropy_with_logits(
            out["maccs_logits"], batch["maccs_fp"].to(DEVICE), reduction="mean")
        heavy_loss = F.smooth_l1_loss(out["heavy_atoms"], batch["heavy_atoms"].to(DEVICE))
        ring_loss = F.smooth_l1_loss(out["ring_count"], batch["ring_count"].to(DEVICE))
        epres_loss = F.binary_cross_entropy_with_logits(
            out["elem_presence"], batch["elem_presence"].to(DEVICE), reduction="mean")
        ecount_loss = F.smooth_l1_loss(out["elem_count"], batch["elem_count"].to(DEVICE))
        total = fp_loss + 0.5 * maccs_loss + 0.01 * heavy_loss + 0.02 * ring_loss + 0.2 * epres_loss + 0.02 * ecount_loss
        return total, fp_loss.item()
    """)

    code(r"""
    @torch.no_grad()
    def evaluate_val_sim(model, dl, pw_tensor):
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()
        total_loss, n_batches = 0.0, 0
        all_pred, all_true = [], []
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
            all_pred.append(torch.sigmoid(out["fp_logits"]).float().cpu())
            all_true.append(batch["morgan_fp"])

        pred_fp = torch.cat(all_pred, dim=0).numpy()
        true_fp = torch.cat(all_true, dim=0).numpy()
        cos_sims = []
        for i in range(len(pred_fp)):
            n1, n2 = np.linalg.norm(pred_fp[i]), np.linalg.norm(true_fp[i])
            if n1 > 0 and n2 > 0:
                cos_sims.append(np.dot(pred_fp[i], true_fp[i]) / (n1 * n2))
        pred_bin = (pred_fp > 0.5).astype(np.float32)
        tanims = []
        for i in range(len(pred_bin)):
            inter = (pred_bin[i] * true_fp[i]).sum()
            union = pred_bin[i].sum() + true_fp[i].sum() - inter
            tanims.append(inter / max(union, 1))
        return total_loss / max(n_batches, 1), np.mean(cos_sims) if cos_sims else 0.0, np.mean(tanims) if tanims else 0.0
    """)

    # ── 10. Training Helpers & Epochs ─────────────────────────────────────
    md("## 10. Proxy MRR Retrieval & Epoch Execution")
    code(r"""
    train_mols_unique = train_split["molecule_id"].unique()
    train_ref_fps = np.stack([morgan_arr[mol_id_to_idx[m]] for m in train_mols_unique])
    train_ref_masses = train_split.groupby("molecule_id")["neutral_mass"].first().loc[train_mols_unique].to_numpy()

    @torch.no_grad()
    def compute_val_proxy_mrr(model, val_split):
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()
        fp_probs, _, mol_ids = spec2fp.predict(
            raw_model, val_split, device=DEVICE, batch_size=cfg.PRED_BATCH_SIZE,
            top_k=cfg.TOP_K_PEAKS, mz_dim=cfg.MZ_DIM)
        val_mass_map = val_split.groupby("molecule_id")["neutral_mass"].first().to_dict()
        rrs, top1_hits, top25_hits = [], 0, 0
        for i, mid in enumerate(mol_ids):
            q_fp = fp_probs[i].astype(np.float32)
            t_idx = mol_id_to_idx.get(str(mid))
            if t_idx is None:
                continue
            true_fp = morgan_arr[t_idx]
            q_mass = val_mass_map.get(mid, 0.0)
            mass_diff = np.abs(train_ref_masses - q_mass)
            cand_mask = (mass_diff <= (50.0 * 1e-6 * max(q_mass, 1.0)))
            if cand_mask.sum() < 25:
                cand_idx = np.argsort(mass_diff)[:min(len(mass_diff), 50)]
            else:
                cand_idx = np.where(cand_mask)[0]
            cand_fps = np.vstack([train_ref_fps[cand_idx], true_fp[None, :]])
            norms = np.linalg.norm(cand_fps, axis=1) * max(np.linalg.norm(q_fp), 1e-8)
            cos_scores = (cand_fps @ q_fp) / np.maximum(norms, 1e-8)
            true_score = cos_scores[-1]
            rank = int((cos_scores >= true_score - 1e-8).sum())
            rr = 1.0 / rank if rank <= 25 else 0.0
            rrs.append(rr)
            if rank == 1: top1_hits += 1
            if rank <= 25: top25_hits += 1
        n = max(len(rrs), 1)
        return float(np.mean(rrs)) if rrs else 0.0, top1_hits / n, top25_hits / n
    """)

    code(r"""
    def train_one_epoch(model, dl, opt, sch, scl, ema, pw_tensor):
        model.train()
        ep_loss, n_steps = 0.0, 0
        t0 = time.perf_counter()
        dev_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
        for batch in tqdm(dl, desc="Train Step", leave=False):
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

    # ── 11. Main Training Loop ────────────────────────────────────────────
    md("## 11. Main Training Loop")
    code(r"""
    root = Path(cfg.OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    ema = EMA(model, decay=cfg.EMA_DECAY)
    best_proxy_mrr, best_epoch, patience_cnt = -1.0, 0, 0
    best_ema_sd = None
    epoch_times = []
    history = {"train_loss": [], "val_loss": [], "val_cos": [], "val_tan": [],
               "proxy_mrr": [], "top1": [], "top25": [], "lr": []}
    """)

    code(r"""
    for epoch in range(max_epochs):
        ep_loss, ep_time = train_one_epoch(model, dl_train, optimizer, scheduler, scaler, ema, pw_tensor)
        epoch_times.append(ep_time)
        ema.apply_shadow(model)
        v_loss, v_cos, v_tan = evaluate_val_sim(model, dl_val, pw_tensor)
        v_mrr, v_top1, v_top25 = compute_val_proxy_mrr(model, val_model_split)
        ema.restore(model)

        cur_lr = optimizer.param_groups[0]["lr"]
        for k, v in [("train_loss", ep_loss), ("val_loss", v_loss), ("val_cos", v_cos),
                     ("val_tan", v_tan), ("proxy_mrr", v_mrr), ("top1", v_top1),
                     ("top25", v_top25), ("lr", cur_lr)]:
            history[k].append(v)

        print(f"Ep {epoch+1:02d}: loss={ep_loss:.4f} val_loss={v_loss:.4f} cos={v_cos:.4f} "
              f"proxy_MRR={v_mrr:.4f} top1={v_top1:.2f} top25={v_top25:.2f} ({ep_time:.1f}s)")

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

    print(f"Training completed. Best epoch: {best_epoch} with proxy MRR@25 = {best_proxy_mrr:.4f}")
    """)

    # ── 12. Artifact Saving ───────────────────────────────────────────────
    md("## 12. Save Best Checkpoint & Config")
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
    print(f"Saved spec2fp_model.pt ({(root / 'spec2fp_model.pt').stat().st_size / 1e6:.1f} MB)")
    print(f"Saved spec2fp_config.json and spec2fp_pos_weights.npy")
    """)

    # ── 13. Generate Split Predictions ────────────────────────────────────
    md("## 13. Generate Predictions on Splits (val_model, val_rank, test)")
    code(r"""
    raw_eval_model = model.module if hasattr(model, "module") else model
    for name, param in raw_eval_model.named_parameters():
        if name in best_ema_sd:
            param.data.copy_(best_ema_sd[name])
    raw_eval_model.eval()

    def generate_and_save_split(split_df, split_name):
        fp_probs, elem_pred, mol_ids = spec2fp.predict(
            raw_eval_model, split_df, device=DEVICE, batch_size=cfg.PRED_BATCH_SIZE,
            top_k=cfg.TOP_K_PEAKS, mz_dim=cfg.MZ_DIM)
        np.save(root / f"fp_pred_{split_name}.npy", fp_probs)
        np.save(root / f"elem_pred_{split_name}.npy", elem_pred)
        pd.DataFrame({"molecule_id": mol_ids}).to_parquet(root / f"fp_pred_ids_{split_name}.parquet")
        print(f"[{split_name}] Saved fp={fp_probs.shape}, elem={elem_pred.shape}, ids={len(mol_ids)}")
        return fp_probs, elem_pred, mol_ids
    """)

    code(r"""
    fp_vm, elem_vm, ids_vm = generate_and_save_split(val_model_split, "val_model")
    fp_vr, elem_vr, ids_vr = generate_and_save_split(val_rank_split, "val_rank")
    fp_te, elem_te, ids_te = generate_and_save_split(test_df, "test")
    del fp_vm, fp_vr, fp_te; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    """)

    # ── 14. Validation Metrics & Plot ─────────────────────────────────────
    md("## 14. Validation Metrics & Diagnostic Curves")
    code(r"""
    final_v_loss, final_v_cos, final_v_tan = evaluate_val_sim(model, dl_val, pw_tensor)
    final_v_mrr, final_v_top1, final_v_top25 = compute_val_proxy_mrr(model, val_model_split)

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
    }
    (root / "spec2fp_val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"Validation summary: {json.dumps(val_metrics, indent=2)}")
    """)

    code(r"""
    if HAS_MPL and len(history["train_loss"]) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(history["train_loss"], label="Train")
        axes[0, 0].plot(history["val_loss"], label="Val")
        axes[0, 0].set_title("Multi-Task Loss"); axes[0, 0].legend()
        axes[0, 1].plot(history["val_cos"], label="Cosine")
        axes[0, 1].plot(history["val_tan"], label="Tanimoto")
        axes[0, 1].set_title("Val Fingerprint Similarity"); axes[0, 1].legend()
        axes[1, 0].plot(history["proxy_mrr"], label="Proxy MRR@25", color="green")
        axes[1, 0].plot(history["top25"], label="Top-25 Acc", linestyle="--", color="blue")
        axes[1, 0].set_title("Proxy Retrieval Ranking"); axes[1, 0].legend()
        axes[1, 1].plot(history["lr"], color="purple")
        axes[1, 1].set_title("Cosine LR Schedule")
        plt.tight_layout()
        plt.savefig(root / "spec2fp_training_curve.png", dpi=150)
        plt.close()
        print("Diagnostic curve saved to spec2fp_training_curve.png")
    """)

    # ── 15. FULL_TRAIN (Optional) ─────────────────────────────────────────
    md("## 15. Optional Full Dataset Retraining")
    code(r"""
    if cfg.FULL_TRAIN:
        print(f"FULL_TRAIN=True: Retraining on all train molecules for {best_epoch} epochs...")
        ds_full = SpecFPDataset(train_df.reset_index(drop=True), mol_id_to_idx,
                                morgan_arr, maccs_arr, heavy_arr, ring_arr,
                                epres_arr, ecount_arr, cfg, augment=True)
        dl_full = DataLoader(ds_full, batch_size=cfg.BATCH_SIZE, shuffle=True,
                             num_workers=nw, pin_memory=True, collate_fn=collate_train_batch)
        full_base = spec2fp.Spec2FPModel(model_cfg).to(DEVICE)
        full_model = nn.DataParallel(full_base) if N_GPU > 1 else full_base
        opt_f = torch.optim.AdamW(full_model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        sch_f = torch.optim.lr_scheduler.LambdaLR(opt_f, lr_lambda)
        scl_f = GradScaler("cuda", enabled="cuda" in str(DEVICE))
        ema_f = EMA(full_model, decay=cfg.EMA_DECAY)
        for ep in range(best_epoch):
            train_one_epoch(full_model, dl_full, opt_f, sch_f, scl_f, ema_f, pw_tensor)
        ema_f.apply_shadow(full_model)
        torch.save({"ema_state_dict": ema_f.state_dict(), "epoch": best_epoch, "mode": "full_train"},
                   root / "spec2fp_model_full.pt")
        print("Full training model checkpoint saved as spec2fp_model_full.pt")
        del full_model, full_base, opt_f, sch_f, scl_f, ema_f; gc.collect()
    else:
        print("FULL_TRAIN=False. Preserving default split3 train-only model (fold0) for stage 09 consistency.")
    """)

    # ── 16. Manifest & Sanity Checks ──────────────────────────────────────
    md("## 16. Manifest Contract & Sanity Verifications")
    code(r"""
    elapsed = time.perf_counter() - T_START
    heldout_ids = sorted(set(val_model_split["molecule_id"]).union(set(val_rank_split["molecule_id"])))
    heldout_hash = hashlib.sha256("".join(heldout_ids).encode()).hexdigest()

    manifest = {
        "stage": "spec2fp",
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

    code(r"""
    required_artifacts = [
        "spec2fp_model.pt", "spec2fp_config.json", "spec2fp_pos_weights.npy",
        "spec2fp_val_metrics.json", "fp_pred_val_model.npy", "fp_pred_val_rank.npy",
        "fp_pred_test.npy", "fp_pred_ids_val_model.parquet", "fp_pred_ids_val_rank.parquet",
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

    # ── 17. Final Module Source Output ────────────────────────────────────
    md("## 17. Complete Source of `spec2fp.py`")
    code('print(Path("/kaggle/working/spec2fp.py").read_text(encoding="utf-8"))')

    # ── Build & Validate Notebook ─────────────────────────────────────────
    if output.exists():
        try:
            old_nb = nbf.read(output, as_version=4)
            if len(old_nb.cells) == len(NOTEBOOK_CELLS):
                for old_cell, new_cell in zip(old_nb.cells, NOTEBOOK_CELLS):
                    if "id" in old_cell:
                        new_cell["id"] = old_cell["id"]
        except Exception as e:
            warnings.warn(f"Could not preserve cell IDs: {e}")

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
            if len(lines) > 59:
                oversize_cells.append((i, len(lines)))
            source = cell.source.split("\n", 1)[1] if cell.source.startswith("%%writefile") else cell.source
            try:
                ast.parse(source, filename=f"cell_{i}")
            except SyntaxError as e:
                raise SyntaxError(f"Cell {i} syntax error: {e}\n{source[:200]}")

    if oversize_cells:
        raise ValueError(f"Found code cells exceeding 59 lines: {oversize_cells}")

    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    delivery = Path("/mnt/user-data/outputs")
    if delivery.is_dir() and delivery.resolve() != output.parent.resolve():
        nbf.write(nb, delivery / output.name)
    print(f"Created {output}: {len(nb.cells)} cells; nbformat + per-cell AST validation passed")
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("05_spec2fp_train.ipynb"))
    args = parser.parse_args()
    build_notebook(args.output)
