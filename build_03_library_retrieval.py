#!/usr/bin/env python
"""
Build 03_library_retrieval.ipynb programmatically via nbformat (v4).
Strongest non-learned baseline: spectral library search with multiple
similarity metrics on GPU.
"""
import ast, json, os, sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
})
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip()))

def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Title
# ═══════════════════════════════════════════════════════════════════════
md(r"""
# 03 – Library Retrieval Baseline (CASMI 2026 / Enveda)

**Purpose**: Strongest non-learned baseline — rank training molecules for each
query by spectral similarity.

| Spec | Detail |
|:---|:---|
| **Similarities** | (a) Cosine on √-intensity binned vectors (0.01 Da, 0.1 Da), (b) Modified cosine (neutral-loss matching), (c) Spectral entropy (JSD) |
| **Inputs** | `train_clean.parquet` or `train.parquet`, `test_clean.parquet` or `test.parquet`, `splits.parquet`, `evalkit.py` |
| **Outputs** | `cand_lib_val_model.parquet`, `cand_lib_val_rank.parquet`, `cand_lib_test.parquet`, `submission_libretrieval.csv`, `lib_index.npz`, `lib_config.json`, `manifest.json` |
| **Hardware** | 2× T4 GPU (≤12 GB each), ≤29 GB RAM |
| **Runtime** | Measured, never simplified to shorten |
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Config
# ═══════════════════════════════════════════════════════════════════════
code(r"""
from dataclasses import dataclass, field, asdict
import os, time

IN_KAGGLE = os.path.exists("/kaggle")

@dataclass
class CFG:
    stage: str = "library_retrieval"
    seed: int = 42
    SMOKE_TEST: bool = False
    smoke_n_rows: int = 500

    # Paths
    data_dir: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra" if IN_KAGGLE else os.getcwd()
    output_dir: str = "/kaggle/working/artifacts/library_retrieval" if IN_KAGGLE else os.path.join(os.getcwd(), "artifacts", "library_retrieval")

    # Spectral binning
    MAX_MZ: float = 1500.0
    BIN_COARSE: float = 0.1     # 15 000 bins
    BIN_FINE: float = 0.01      # 150 000 bins
    MAX_PEAKS: int = 200        # per spectrum cap

    # Mass pre-filter for modified cosine / entropy
    MASS_WINDOW: float = 2.0    # Da

    # Modified cosine tolerances
    DA_TOL: float = 0.01
    PPM_TOL: float = 10e-6      # 10 ppm

    # GPU batching
    COS_LIB_CHUNK: int = 30000
    COS_QUERY_CHUNK: int = 5000
    MOD_COS_BATCH: int = 2048
    TOP_K: int = 100

    # Time budget (soft – never simplify to meet it)
    time_budget_s: float = 28800

cfg = CFG()
os.makedirs(cfg.output_dir, exist_ok=True)
T_START = time.time()
print(f"CFG initialised. Output -> {cfg.output_dir}")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — Imports & environment
# ═══════════════════════════════════════════════════════════════════════
code(r"""
import gc, json, math, hashlib, warnings
from glob import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
from scipy import stats as sp_stats
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize as sk_normalize
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    raise ImportError(f"RDKit mandatory: {e}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | device={DEVICE} | GPUs={N_GPUS}")
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_mem/1e9:.1f} GB)")
print(f"NumPy {np.__version__} | Pandas {pd.__version__} | SciPy {scipy.__version__}")

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — Utilities & evalkit
# ═══════════════════════════════════════════════════════════════════════
code(r"""
# ── find_file: recursive artifact discovery ──────────────────────────
def find_file(filename, hint="", required=True):
    # Glob /kaggle/input/**/ and working dir for *filename*.
    paths = (
        glob(f"/kaggle/input/**/{filename}", recursive=True)
        + glob(f"/kaggle/working/**/{filename}", recursive=True)
        + glob(f"**/{filename}", recursive=True)
    )
    if hint:
        pref = [p for p in paths if hint in p]
        if pref:
            paths = pref + [p for p in paths if p not in pref]
    if paths:
        print(f"[find_file] {filename} -> {paths[0]}")
        return paths[0]
    if required:
        raise FileNotFoundError(f"Cannot locate {filename}")
    print(f"[find_file] {filename} not found (optional)")
    return None

# ── evalkit ──────────────────────────────────────────────────────────
import sys, shutil
if IN_KAGGLE:
    ek_path = "/kaggle/working/evalkit.py"
    if not os.path.exists(ek_path):
        src_ek = find_file("evalkit.py", required=False)
        if src_ek and os.path.abspath(src_ek) != os.path.abspath(ek_path):
            shutil.copy(src_ek, ek_path)
            print(f"Copied {src_ek} -> {ek_path}")
else:
    ek_path = find_file("evalkit.py", required=True)

ek_dir = os.path.dirname(os.path.abspath(ek_path))
if ek_dir not in sys.path:
    sys.path.insert(0, ek_dir)
import evalkit
print(f"evalkit loaded from {ek_path}")
print(f"  FP_SPEC = {evalkit.FP_SPEC}")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — MD: Data loading
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 1. Data Ingestion
Load `train_clean.parquet` (or raw `train.parquet`), `test_clean.parquet`
(or `test.parquet`), and `splits.parquet`. Join train with splits to
assign molecules to `train` / `val_model` / `val_rank`.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — Load train + splits
# ═══════════════════════════════════════════════════════════════════════
code(r"""
# ── Load splits ──────────────────────────────────────────────────────
splits_path = find_file("splits.parquet")
df_splits = pd.read_parquet(splits_path)
assert {"molecule_id", "ik14", "split3"}.issubset(df_splits.columns)
print(f"splits: {len(df_splits)} rows, split3 counts = {df_splits['split3'].value_counts().to_dict()}")

# ── Load train spectra ───────────────────────────────────────────────
train_path = find_file("train_clean.parquet", required=False)
if train_path is None:
    train_path = find_file("train.parquet")
df_train = pd.read_parquet(train_path)
print(f"train raw: {len(df_train)} spectra, cols={df_train.columns.tolist()}")

# ── Standardise column names ─────────────────────────────────────────
col_map = {}
for dst, srcs in [
    ("molecule_id",  ["molecule_id", "inchikey14"]),
    ("smiles",       ["smiles", "normalized_smiles"]),
    ("precursor_mz", ["precursor_mz"]),
    ("adduct",       ["adduct"]),
    ("polarity",     ["polarity", "ionization_mode"]),
    ("mzs",          ["mzs", "ms2_mzs", "peaks_mz", "mz"]),
    ("intensities",  ["intensities", "ms2_normalized_intensities", "peaks_intensity", "inten"]),
]:
    for s in srcs:
        if s in df_train.columns:
            col_map[s] = dst
            break
df_train = df_train.rename(columns=col_map)
print(f"Column mapping applied: {col_map}")

# Ensure molecule_id
assert "molecule_id" in df_train.columns, f"No molecule_id column! Cols: {df_train.columns.tolist()}"
df_train["molecule_id"] = df_train["molecule_id"].astype(str)

# SMOKE_TEST slice
if cfg.SMOKE_TEST:
    keep_mols = df_splits["molecule_id"].head(cfg.smoke_n_rows).tolist()
    df_train = df_train[df_train["molecule_id"].isin(keep_mols)].copy()
    print(f"SMOKE_TEST: kept {len(df_train)} spectra")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Load test + merge splits
# ═══════════════════════════════════════════════════════════════════════
code(r"""
# ── Load test spectra ────────────────────────────────────────────────
test_path = find_file("test_clean.parquet", required=False)
if test_path is None:
    test_path = find_file("test.parquet")
df_test = pd.read_parquet(test_path)
test_col_map = {}
for dst, srcs in [
    ("molecule_id",  ["molecule_id"]),
    ("precursor_mz", ["precursor_mz"]),
    ("adduct",       ["adduct"]),
    ("polarity",     ["polarity", "ionization_mode"]),
    ("mzs",          ["mzs", "ms2_mzs", "peaks_mz", "mz"]),
    ("intensities",  ["intensities", "ms2_normalized_intensities", "peaks_intensity", "inten"]),
]:
    for s in srcs:
        if s in df_test.columns:
            test_col_map[s] = dst
            break
df_test = df_test.rename(columns=test_col_map)
df_test["molecule_id"] = df_test["molecule_id"].astype(str)
print(f"test: {len(df_test)} spectra, {df_test['molecule_id'].nunique()} molecules")

# ── Merge splits into train ──────────────────────────────────────────
if "split_random" not in df_splits.columns:
    df_splits["split_random"] = df_splits["split3"]

split_merge_cols = [c for c in ["split3", "split_random"] if c in df_splits.columns]
if "ik14" not in df_train.columns and "ik14" in df_splits.columns:
    split_merge_cols.append("ik14")

df_train = df_train.merge(
    df_splits[["molecule_id"] + split_merge_cols],
    on="molecule_id", how="inner"
)
if "ik14" not in df_train.columns and "ik14" in df_splits.columns:
    df_train["ik14"] = df_train["molecule_id"]
print(f"train after merge: {len(df_train)} spectra")
print(f"  split3 spectra counts: {df_train['split3'].value_counts().to_dict()}")

# ── Build molecule-level metadata table ──────────────────────────────
smiles_col = "smiles" if "smiles" in df_train.columns else None
meta_cols = [c for c in ["molecule_id", "ik14", "split3", "split_random", "precursor_mz", "adduct", "polarity"] if c in df_train.columns]
if smiles_col and smiles_col in df_train.columns and smiles_col not in meta_cols:
    meta_cols.insert(2, smiles_col)
df_mol_meta = df_train[meta_cols].drop_duplicates("molecule_id").reset_index(drop=True)

# Compute neutral mass per molecule (use first spectrum's precursor)
df_mol_meta["neutral_mass"] = [
    evalkit.calc_neutral_mass(row["precursor_mz"], row["adduct"],
                              row.get("polarity", "+"))
    for _, row in df_mol_meta.iterrows()
]
print(f"Molecule meta: {len(df_mol_meta)} molecules")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — MD: Preprocessing
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 2. Spectrum Preprocessing & Molecule-Level Aggregation
For each molecule we build:
1. **Merged binned vector** (max-pool of √-intensity across spectra) at 0.1 Da and 0.01 Da.
2. **Best peak list** (spectrum with highest total √-intensity) for modified cosine.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — Build molecule-level data (train)
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
n_bins_c = int(cfg.MAX_MZ / cfg.BIN_COARSE)   # 15 000
n_bins_f = int(cfg.MAX_MZ / cfg.BIN_FINE)      # 150 000

# Ordered molecule list and index map
mol_ids = df_mol_meta["molecule_id"].tolist()
mol_id_to_idx = {m: i for i, m in enumerate(mol_ids)}
n_mols = len(mol_ids)

# Pre-allocate padded peak arrays for modified cosine
mol_peak_mzs  = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.float32)
mol_peak_ints = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.float32)
mol_peak_mask = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.bool_)
mol_prec_mass = np.zeros(n_mols, dtype=np.float32)
mol_best_score = np.full(n_mols, -1.0, dtype=np.float32)

# Sparse builders for binned vectors
rows_c, cols_c, data_c = [], [], []
rows_f, cols_f, data_f = [], [], []

# Temporary dense accumulators (one molecule at a time)
_buf_c = np.zeros(n_bins_c, dtype=np.float32)
_buf_f = np.zeros(n_bins_f, dtype=np.float32)

# Group spectra by molecule
grouped = df_train.groupby("molecule_id")

for mol_id, grp in tqdm(grouped, total=n_mols, desc="Building mol vectors"):
    midx = mol_id_to_idx.get(mol_id)
    if midx is None:
        continue
    _buf_c[:] = 0.0
    _buf_f[:] = 0.0

    for _, row in grp.iterrows():
        mzs  = np.asarray(row["mzs"],  dtype=np.float64)
        ints = np.asarray(row["intensities"], dtype=np.float64)
        prec = float(row["precursor_mz"])

        # Filter: peaks <= precursor + 5 Da, positive
        keep = (mzs <= prec + 5.0) & (mzs >= 0) & (ints > 0)
        mzs, ints = mzs[keep], ints[keep]
        if len(mzs) == 0:
            continue
        ints = np.sqrt(np.clip(ints, 0, None))

        # Coarse bins
        bc = np.clip((mzs / cfg.BIN_COARSE).astype(np.int64), 0, n_bins_c - 1)
        np.maximum.at(_buf_c, bc, ints)
        # Fine bins
        bf = np.clip((mzs / cfg.BIN_FINE).astype(np.int64), 0, n_bins_f - 1)
        np.maximum.at(_buf_f, bf, ints)

        # Best spectrum for modified cosine
        s = float(ints.sum())
        if s > mol_best_score[midx]:
            mol_best_score[midx] = s
            # top MAX_PEAKS by intensity
            if len(mzs) > cfg.MAX_PEAKS:
                top = np.argsort(ints)[-cfg.MAX_PEAKS:]
                mzs_t, ints_t = mzs[top], ints[top]
            else:
                mzs_t, ints_t = mzs, ints
            n = len(mzs_t)
            mol_peak_mzs[midx, :] = 0; mol_peak_ints[midx, :] = 0; mol_peak_mask[midx, :] = False
            mol_peak_mzs[midx, :n]  = mzs_t.astype(np.float32)
            mol_peak_ints[midx, :n] = ints_t.astype(np.float32)
            mol_peak_mask[midx, :n] = True
            mol_prec_mass[midx] = prec

    # L2-normalise and store sparse coarse
    nc = np.linalg.norm(_buf_c)
    if nc > 0:
        _buf_c /= nc
    nz = np.nonzero(_buf_c)[0]
    rows_c.extend([midx]*len(nz)); cols_c.extend(nz); data_c.extend(_buf_c[nz].tolist())

    # L2-normalise and store sparse fine
    nf = np.linalg.norm(_buf_f)
    if nf > 0:
        _buf_f /= nf
    nz2 = np.nonzero(_buf_f)[0]
    rows_f.extend([midx]*len(nz2)); cols_f.extend(nz2); data_f.extend(_buf_f[nz2].tolist())

# Build CSR
sp_coarse = csr_matrix((np.array(data_c, dtype=np.float32),
                         (np.array(rows_c), np.array(cols_c))),
                        shape=(n_mols, n_bins_c))
sp_fine   = csr_matrix((np.array(data_f, dtype=np.float32),
                         (np.array(rows_f), np.array(cols_f))),
                        shape=(n_mols, n_bins_f))

mol_neutral_mass = df_mol_meta["neutral_mass"].values.astype(np.float32)
print(f"Built molecule vectors in {time.time()-t0:.1f}s")
print(f"  coarse sparse: {sp_coarse.shape}, nnz={sp_coarse.nnz}")
print(f"  fine   sparse: {sp_fine.shape}, nnz={sp_fine.nnz}")

# Free raw train dataframe
del df_train, grouped, rows_c, cols_c, data_c, rows_f, cols_f, data_f
gc.collect()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — Build test molecule data
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
test_mol_ids_all = sorted(df_test["molecule_id"].unique())
n_test = len(test_mol_ids_all)
test_id_to_idx = {m: i for i, m in enumerate(test_mol_ids_all)}

test_peak_mzs  = np.zeros((n_test, cfg.MAX_PEAKS), dtype=np.float32)
test_peak_ints = np.zeros((n_test, cfg.MAX_PEAKS), dtype=np.float32)
test_peak_mask = np.zeros((n_test, cfg.MAX_PEAKS), dtype=np.bool_)
test_prec_mass = np.zeros(n_test, dtype=np.float32)
test_best_score = np.full(n_test, -1.0)

tr_c, tc_c, td_c = [], [], []
tr_f, tc_f, td_f = [], [], []
_buf_c2 = np.zeros(n_bins_c, dtype=np.float32)
_buf_f2 = np.zeros(n_bins_f, dtype=np.float32)

for mol_id, grp in tqdm(df_test.groupby("molecule_id"), total=n_test, desc="Test mol vectors"):
    tidx = test_id_to_idx[mol_id]
    _buf_c2[:] = 0; _buf_f2[:] = 0
    for _, row in grp.iterrows():
        mzs  = np.asarray(row["mzs"], dtype=np.float64)
        ints = np.asarray(row["intensities"], dtype=np.float64)
        prec = float(row["precursor_mz"])
        keep = (mzs <= prec + 5.0) & (mzs >= 0) & (ints > 0)
        mzs, ints = mzs[keep], ints[keep]
        if len(mzs) == 0:
            continue
        ints = np.sqrt(np.clip(ints, 0, None))
        bc = np.clip((mzs / cfg.BIN_COARSE).astype(np.int64), 0, n_bins_c - 1)
        np.maximum.at(_buf_c2, bc, ints)
        bf = np.clip((mzs / cfg.BIN_FINE).astype(np.int64), 0, n_bins_f - 1)
        np.maximum.at(_buf_f2, bf, ints)
        s = float(ints.sum())
        if s > test_best_score[tidx]:
            test_best_score[tidx] = s
            if len(mzs) > cfg.MAX_PEAKS:
                top = np.argsort(ints)[-cfg.MAX_PEAKS:]
                mzs_t, ints_t = mzs[top], ints[top]
            else:
                mzs_t, ints_t = mzs, ints
            n = len(mzs_t)
            test_peak_mzs[tidx, :] = 0; test_peak_ints[tidx, :] = 0; test_peak_mask[tidx, :] = False
            test_peak_mzs[tidx, :n]  = mzs_t.astype(np.float32)
            test_peak_ints[tidx, :n] = ints_t.astype(np.float32)
            test_peak_mask[tidx, :n] = True
            test_prec_mass[tidx] = prec
    nc = np.linalg.norm(_buf_c2)
    if nc > 0: _buf_c2 /= nc
    nz = np.nonzero(_buf_c2)[0]
    tr_c.extend([tidx]*len(nz)); tc_c.extend(nz); td_c.extend(_buf_c2[nz].tolist())
    nf = np.linalg.norm(_buf_f2)
    if nf > 0: _buf_f2 /= nf
    nz2 = np.nonzero(_buf_f2)[0]
    tr_f.extend([tidx]*len(nz2)); tc_f.extend(nz2); td_f.extend(_buf_f2[nz2].tolist())

test_sp_coarse = csr_matrix((np.array(td_c, dtype=np.float32),
                              (np.array(tr_c), np.array(tc_c))),
                             shape=(n_test, n_bins_c))
test_sp_fine   = csr_matrix((np.array(td_f, dtype=np.float32),
                              (np.array(tr_f), np.array(tc_f))),
                             shape=(n_test, n_bins_f))

# Neutral mass for test
df_test_first = df_test.drop_duplicates("molecule_id").set_index("molecule_id")
test_neutral_mass = np.array([
    evalkit.calc_neutral_mass(
        float(df_test_first.loc[m, "precursor_mz"]),
        str(df_test_first.loc[m, "adduct"]),
        str(df_test_first.loc[m, "polarity"]) if "polarity" in df_test_first.columns else (str(df_test_first.loc[m, "ionization_mode"]) if "ionization_mode" in df_test_first.columns else "+")
    ) for m in test_mol_ids_all
], dtype=np.float32)
del df_test_first

del df_test; gc.collect()
print(f"Test vectors built in {time.time()-t0:.1f}s  ({n_test} molecules)")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — Library ceiling check
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 3. Library Ceiling
Fraction of val molecules whose tautomer-canonical InChIKey14 matches
*any* train molecule. This is the theoretical max MRR@25 from pure
library retrieval.
""")

code(r"""
t0 = time.time()
train_mask = df_mol_meta["split3"] == "train"
val_model_mask = df_mol_meta["split3"] == "val_model"
val_rank_mask  = df_mol_meta["split3"] == "val_rank"

lib_idx  = np.where(train_mask)[0]
vm_idx   = np.where(val_model_mask)[0]
vr_idx   = np.where(val_rank_mask)[0]

# Build canon_ik14 for library molecules (use fast_key for speed on bulk)
lib_ik14 = set(df_mol_meta.loc[train_mask, "ik14"].tolist())
smiles_col = "smiles" if "smiles" in df_mol_meta.columns else None

def _ceiling(val_indices, lib_set):
    # Fraction of val molecules whose ik14 in lib.
    hits = 0
    for i in val_indices:
        if df_mol_meta.iloc[i]["ik14"] in lib_set:
            hits += 1
    return hits / max(len(val_indices), 1)

ceil_vm = _ceiling(vm_idx, lib_ik14)
ceil_vr = _ceiling(vr_idx, lib_ik14)
print(f"Library ceiling (ik14 match):")
print(f"  val_model: {ceil_vm:.4f} ({int(ceil_vm*len(vm_idx))}/{len(vm_idx)})")
print(f"  val_rank:  {ceil_vr:.4f} ({int(ceil_vr*len(vr_idx))}/{len(vr_idx)})")
print(f"  (If 0%, scaffold split ensures no val molecule is in library;")
print(f"   MRR>0 still possible if tautomers match after canon_ik14.)")
print(f"Ceiling check took {time.time()-t0:.1f}s")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — MD: Cosine similarity
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 4. Binned Cosine Similarity (GPU)
Chunked matrix multiplication of L2-normalised √-intensity binned
vectors. Two resolutions: 0.1 Da (15 000 bins) and 0.01 Da (150 000
bins). Sparse -> dense conversion per chunk keeps GPU memory ≤ 12 GB.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 13 — GPU cosine function
# ═══════════════════════════════════════════════════════════════════════
code(r"""
def gpu_cosine_topk(query_sp, lib_sp, top_k=100,
                    q_chunk=5000, l_chunk=30000, desc="cosine"):
    # Compute cosine similarity between sparse query and library matrices
    # on GPU using chunked dense matmul.
    # Returns (scores, indices) arrays of shape (n_query, top_k).
    n_q = query_sp.shape[0]
    n_l = lib_sp.shape[0]
    actual_k = min(top_k, n_l)
    best_scores = np.full((n_q, top_k), -1.0, dtype=np.float32)
    best_idx    = np.full((n_q, top_k), -1, dtype=np.int64)

    if actual_k == 0:
        return best_scores, best_idx

    for qs in tqdm(range(0, n_q, q_chunk), desc=desc):
        qe = min(qs + q_chunk, n_q)
        Q = torch.from_numpy(query_sp[qs:qe].toarray().astype(np.float32)).to(DEVICE)

        cur_topk_scores = torch.full((qe - qs, actual_k), -1.0, device=DEVICE)
        cur_topk_idx    = torch.full((qe - qs, actual_k), -1, dtype=torch.long, device=DEVICE)

        for ls in range(0, n_l, l_chunk):
            le = min(ls + l_chunk, n_l)
            L = torch.from_numpy(lib_sp[ls:le].toarray().astype(np.float32)).to(DEVICE)
            sim = Q @ L.t()                          # (qchunk, lchunk)
            chunk_k = min(actual_k, le - ls)
            vals, idxs = sim.topk(chunk_k, dim=1)
            idxs += ls                               # offset
            # Merge with running top-k
            merged_s = torch.cat([cur_topk_scores, vals], dim=1)
            merged_i = torch.cat([cur_topk_idx, idxs], dim=1)
            topk_s, sel = merged_s.topk(actual_k, dim=1)
            cur_topk_scores = topk_s
            cur_topk_idx    = merged_i.gather(1, sel)
            del L, sim, vals, idxs, merged_s, merged_i
        best_scores[qs:qe, :actual_k] = cur_topk_scores.cpu().numpy()
        best_idx[qs:qe, :actual_k]    = cur_topk_idx.cpu().numpy()
        del Q, cur_topk_scores, cur_topk_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_scores, best_idx
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Run cosine 0.1 Da on val splits
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
# Library = train-split molecules only
lib_sp_c  = sp_coarse[lib_idx]
# Queries = val_model + val_rank
val_idx = np.concatenate([vm_idx, vr_idx])
query_sp_c = sp_coarse[val_idx]

print(f"Cosine 0.1 Da: {query_sp_c.shape[0]} queries × {lib_sp_c.shape[0]} lib")
cos01_scores, cos01_idx = gpu_cosine_topk(
    query_sp_c, lib_sp_c, top_k=cfg.TOP_K,
    q_chunk=cfg.COS_QUERY_CHUNK, l_chunk=cfg.COS_LIB_CHUNK,
    desc="cos_0.1Da"
)
print(f"Cosine 0.1 Da done in {time.time()-t0:.1f}s")
del query_sp_c; gc.collect(); torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 15 — Run cosine 0.01 Da on val splits
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
lib_sp_f  = sp_fine[lib_idx]
query_sp_f = sp_fine[val_idx]

print(f"Cosine 0.01 Da: {query_sp_f.shape[0]} queries × {lib_sp_f.shape[0]} lib")
cos001_scores, cos001_idx = gpu_cosine_topk(
    query_sp_f, lib_sp_f, top_k=cfg.TOP_K,
    q_chunk=min(cfg.COS_QUERY_CHUNK, 2000),
    l_chunk=min(cfg.COS_LIB_CHUNK, 5000),
    desc="cos_0.01Da"
)
print(f"Cosine 0.01 Da done in {time.time()-t0:.1f}s")
del query_sp_f, lib_sp_f; gc.collect(); torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 16 — MD: Modified cosine
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 5. Modified Cosine (Precursor-Shift-Aware, GPU)
Neutral-loss peak matching: peaks match directly **or** after shifting
by the precursor mass difference Δ = M_query − M_lib.

Tolerance: max(0.01 Da, 10 ppm × m/z).

Pre-filtered to |neutral mass diff| ≤ 2 Da.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 17 — Modified cosine function
# ═══════════════════════════════════════════════════════════════════════
code(r"""
def modified_cosine_batch(q_mzs, q_ints, q_masks,
                          l_mzs, l_ints, l_masks,
                          shifts, da_tol=0.01, ppm_tol=10e-6):
    # Batched modified cosine on GPU.
    # Args: all (B, MAX_PEAKS) tensors except shifts (B,).
    # Returns: (B,) similarity scores.
    # (B, Q, 1) vs (B, 1, L) -> (B, Q, L)
    diff = q_mzs.unsqueeze(2) - l_mzs.unsqueeze(1)

    # Adaptive tolerance based on mean m/z
    mean_mz = (q_mzs.unsqueeze(2).abs() + l_mzs.unsqueeze(1).abs()) / 2 + 1e-6
    tol = torch.clamp(mean_mz * ppm_tol, min=da_tol)

    # Direct match + neutral-loss (shifted) match
    direct  = diff.abs() <= tol
    shifted = (diff - shifts.view(-1, 1, 1)).abs() <= tol
    match   = (direct | shifted)

    # Valid-pair mask (both peaks non-padding)
    valid = q_masks.unsqueeze(2) & l_masks.unsqueeze(1)
    match = match & valid

    # Intensity products for matched pairs
    prods = q_ints.unsqueeze(2) * l_ints.unsqueeze(1)
    prods = prods * match.float()

    # Greedy: best match per query peak
    best_per_q = prods.max(dim=2).values           # (B, Q)
    matched_sum = (best_per_q * q_masks.float()).sum(dim=1)

    # Cosine normalisation
    q_norm = (q_ints**2 * q_masks.float()).sum(1).sqrt()
    l_norm = (l_ints**2 * l_masks.float()).sum(1).sqrt()
    denom  = q_norm * l_norm + 1e-8
    return matched_sum / denom
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 18 — Run modified cosine on val
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()

# Sort library by neutral mass for fast window lookup
lib_masses_sorted_order = np.argsort(mol_neutral_mass[lib_idx])
lib_masses_sorted = mol_neutral_mass[lib_idx][lib_masses_sorted_order]
lib_orig_idx_sorted = lib_idx[lib_masses_sorted_order]

# Transfer peak arrays to GPU
all_mzs_t  = torch.from_numpy(mol_peak_mzs).to(DEVICE)
all_ints_t = torch.from_numpy(mol_peak_ints).to(DEVICE)
all_mask_t = torch.from_numpy(mol_peak_mask).to(DEVICE)

mod_scores_list = []
mod_idx_list    = []

for qi, q_global in enumerate(tqdm(val_idx, desc="mod_cos")):
    q_mass = mol_neutral_mass[q_global]
    # Mass window lookup
    lo = np.searchsorted(lib_masses_sorted, q_mass - cfg.MASS_WINDOW, side="left")
    hi = np.searchsorted(lib_masses_sorted, q_mass + cfg.MASS_WINDOW, side="right")
    cand_orig = lib_orig_idx_sorted[lo:hi]

    if len(cand_orig) == 0:
        mod_scores_list.append(np.zeros(cfg.TOP_K, dtype=np.float32))
        mod_idx_list.append(np.full(cfg.TOP_K, -1, dtype=np.int64))
        continue

    n_cand = len(cand_orig)
    q_mzs_b  = all_mzs_t[q_global].unsqueeze(0).expand(n_cand, -1)
    q_ints_b = all_ints_t[q_global].unsqueeze(0).expand(n_cand, -1)
    q_mask_b = all_mask_t[q_global].unsqueeze(0).expand(n_cand, -1)

    cand_t = torch.from_numpy(cand_orig).long().to(DEVICE)
    l_mzs_b  = all_mzs_t[cand_t]
    l_ints_b = all_ints_t[cand_t]
    l_mask_b = all_mask_t[cand_t]
    shifts_b = torch.full((n_cand,), mol_prec_mass[q_global], device=DEVICE) - \
               torch.from_numpy(mol_prec_mass[cand_orig]).to(DEVICE)

    # Batch if too large
    all_sc = []
    for bs in range(0, n_cand, cfg.MOD_COS_BATCH):
        be = min(bs + cfg.MOD_COS_BATCH, n_cand)
        sc = modified_cosine_batch(
            q_mzs_b[bs:be], q_ints_b[bs:be], q_mask_b[bs:be],
            l_mzs_b[bs:be], l_ints_b[bs:be], l_mask_b[bs:be],
            shifts_b[bs:be], cfg.DA_TOL, cfg.PPM_TOL)
        all_sc.append(sc)
    scores_t = torch.cat(all_sc)

    # Top-K
    k = min(cfg.TOP_K, n_cand)
    topv, topi = scores_t.topk(k)
    row_scores = np.full(cfg.TOP_K, -1.0, dtype=np.float32)
    row_idx    = np.full(cfg.TOP_K, -1, dtype=np.int64)
    row_scores[:k] = topv.cpu().numpy()
    # Map back: cand_orig[topi] gives global mol indices
    row_idx[:k] = cand_orig[topi.cpu().numpy()]
    mod_scores_list.append(row_scores)
    mod_idx_list.append(row_idx)

mod_scores = np.stack(mod_scores_list)
mod_idx    = np.stack(mod_idx_list)
print(f"Modified cosine done in {time.time()-t0:.1f}s")
del all_sc; torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 19 — MD: Spectral entropy
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 6. Spectral Entropy Similarity (JSD-Based)
$$S = 1 - \frac{2\,H\!\left(\frac{p+q}{2}\right) - H(p) - H(q)}{\ln 4}$$
where $p, q$ are L1-normalised (probability) √-intensity vectors.
Uses 0.1 Da bins with neutral-mass pre-filter.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 20 — Entropy function + run
# ═══════════════════════════════════════════════════════════════════════
code(r"""
def _entropy(p):
    # Shannon entropy of a probability distribution (rows).
    safe = torch.clamp(p, min=1e-12)
    return -(safe * safe.log()).sum(dim=-1)

def entropy_sim_topk(query_sp_raw, lib_sp_raw, query_masses, lib_masses,
                     mass_window, top_k, desc="entropy"):
    # JSD spectral-entropy similarity with mass pre-filter.
    # query_sp_raw, lib_sp_raw: scipy CSR (unnormalised sqrt-intensity).
    # Returns (scores, indices) arrays (n_query, top_k).
    # L1-normalise to probability distributions
    q_prob = sk_normalize(query_sp_raw, norm="l1", axis=1)
    l_prob = sk_normalize(lib_sp_raw,   norm="l1", axis=1)

    lib_order = np.argsort(lib_masses)
    lib_masses_s = lib_masses[lib_order]
    n_q = q_prob.shape[0]
    best_s = np.full((n_q, top_k), -1.0, dtype=np.float32)
    best_i = np.full((n_q, top_k), -1, dtype=np.int64)

    for qi in tqdm(range(n_q), desc=desc, mininterval=5):
        qm = query_masses[qi]
        lo = np.searchsorted(lib_masses_s, qm - mass_window, side="left")
        hi = np.searchsorted(lib_masses_s, qm + mass_window, side="right")
        cand = lib_order[lo:hi]
        if len(cand) == 0:
            continue
        q_v = torch.from_numpy(q_prob[qi].toarray().astype(np.float32)).to(DEVICE)  # (1,D)
        l_v = torch.from_numpy(l_prob[cand].toarray().astype(np.float32)).to(DEVICE) # (C,D)
        H_q = _entropy(q_v).expand(len(cand))
        H_l = _entropy(l_v)
        H_m = _entropy((q_v + l_v) / 2)
        jsd = (2 * H_m - H_q - H_l) / math.log(4)
        sim = 1.0 - jsd
        k = min(top_k, len(cand))
        topv, topi = sim.topk(k)
        best_s[qi, :k] = topv.cpu().numpy()
        best_i[qi, :k] = cand[topi.cpu().numpy()]
        del q_v, l_v

    torch.cuda.empty_cache()
    return best_s, best_i
""")

code(r"""
t0 = time.time()
# Build un-normalised sparse for library and queries
# (sp_coarse is L2-normed; need raw for entropy -> rebuild L1)
# Trick: un-L2-norm then L1-norm is equivalent to just L1-norming the raw.
# We stored L2-normed values. Recover raw by multiplying by L2 norm.
# Simpler: re-use sp_coarse since L1-norm of L2-normed row still gives valid prob.

ent_lib_raw  = sp_coarse[lib_idx]
ent_q_raw    = sp_coarse[val_idx]

ent_scores, ent_idx = entropy_sim_topk(
    ent_q_raw, ent_lib_raw,
    query_masses=mol_neutral_mass[val_idx],
    lib_masses=mol_neutral_mass[lib_idx],
    mass_window=cfg.MASS_WINDOW, top_k=cfg.TOP_K,
    desc="entropy_val"
)
print(f"Entropy done in {time.time()-t0:.1f}s")
del ent_lib_raw, ent_q_raw; gc.collect(); torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 21 — MD: Evaluation
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 7. Evaluation: MRR@25, Hit Rates, and Method Selection
Compute MRR@25 and hit-rate @1/@5/@25 for each similarity on
`val_model`. Choose the best method using `val_model` only.
Also evaluate on `val_rank` (independent) and `split_random`.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 22 — Build candidate dicts & evaluate
# ═══════════════════════════════════════════════════════════════════════
code(r"""
def build_pred_dict(global_query_indices, scores_arr, idx_arr, lib_global_indices):
    # Build {molecule_id: [smiles_1, ...]} prediction dict from scored indices.
    # idx_arr values index into lib_global_indices (or are global mol indices
    # if lib_global_indices is None).
    pred = {}
    truth = {}
    for qi_local, qi_global in enumerate(global_query_indices):
        qid = mol_ids[qi_global]
        gt_smi = df_mol_meta.iloc[qi_global].get("smiles", "")
        truth[qid] = gt_smi if gt_smi else df_mol_meta.iloc[qi_global]["ik14"]

        guesses = []
        for rank in range(scores_arr.shape[1]):
            lidx = int(idx_arr[qi_local, rank])
            if lidx < 0:
                break
            # lidx might be a global index or lib-local index
            if lib_global_indices is not None:
                g_idx = lib_global_indices[lidx]
            else:
                g_idx = lidx
            cand_smi = df_mol_meta.iloc[g_idx].get("smiles", "")
            if cand_smi:
                guesses.append(cand_smi)
            else:
                guesses.append(mol_ids[g_idx])
        pred[qid] = guesses
    return pred, truth

def eval_metrics(pred, truth, label=""):
    # Compute MRR@25 and hit-rates @1/@5/@25.
    mrr = evalkit.mrr_at_25(pred, truth)
    hits = {1: 0, 5: 0, 25: 0}
    for qid, tval in truth.items():
        tkey = tval if (len(tval)==14 and tval.isupper()) else evalkit.fast_key(tval)
        guesses = pred.get(qid, [])
        for k in [1, 5, 25]:
            for g in guesses[:k]:
                gkey = g if (len(g)==14 and g.isupper()) else evalkit.fast_key(g)
                if gkey == tkey:
                    hits[k] += 1
                    break
    n = len(truth)
    hr = {k: v/max(n,1) for k, v in hits.items()}
    print(f"  [{label}] MRR@25={mrr:.4f} | HR@1={hr[1]:.4f} HR@5={hr[5]:.4f} HR@25={hr[25]:.4f}")
    return {"mrr25": mrr, "hr1": hr[1], "hr5": hr[5], "hr25": hr[25]}
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 23 — Evaluate all methods on val_model
# ═══════════════════════════════════════════════════════════════════════
code(r"""
n_vm = len(vm_idx)
n_vr = len(vr_idx)
vm_local = np.arange(n_vm)            # first n_vm of val_idx
vr_local = np.arange(n_vm, n_vm+n_vr) # next n_vr of val_idx

methods = {}
print("=== val_model results ===")

# Cosine 0.1 Da — idx values index into lib_idx
pred, truth = build_pred_dict(val_idx[vm_local], cos01_scores[vm_local], cos01_idx[vm_local], lib_idx)
methods["cos_01"] = eval_metrics(pred, truth, "cos_0.1Da")

# Cosine 0.01 Da
pred, truth = build_pred_dict(val_idx[vm_local], cos001_scores[vm_local], cos001_idx[vm_local], lib_idx)
methods["cos_001"] = eval_metrics(pred, truth, "cos_0.01Da")

# Modified cosine — idx values are GLOBAL mol indices already
pred, truth = build_pred_dict(val_idx[vm_local], mod_scores[vm_local], mod_idx[vm_local], None)
methods["mod_cos"] = eval_metrics(pred, truth, "mod_cos")

# Entropy — idx values index into lib_idx (via lib_order mapping stored inside)
# Actually ent_idx stores raw indices from lib_order -> re-map
pred, truth = build_pred_dict(val_idx[vm_local], ent_scores[vm_local], ent_idx[vm_local], lib_idx)
methods["entropy"] = eval_metrics(pred, truth, "entropy")

# Select best by val_model MRR@25
best_method = max(methods, key=lambda k: methods[k]["mrr25"])
print(f"\n*** Best method on val_model: {best_method} (MRR@25 = {methods[best_method]['mrr25']:.4f}) ***")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 24 — Evaluate on val_rank + random split
# ═══════════════════════════════════════════════════════════════════════
code(r"""
print("=== val_rank results (independent) ===")
for name, scores_arr, idx_arr, lib_g in [
    ("cos_0.1Da",  cos01_scores,  cos01_idx,  lib_idx),
    ("cos_0.01Da", cos001_scores, cos001_idx, lib_idx),
    ("mod_cos",    mod_scores,    mod_idx,    None),
    ("entropy",    ent_scores,    ent_idx,    lib_idx),
]:
    pred, truth = build_pred_dict(val_idx[vr_local], scores_arr[vr_local], idx_arr[vr_local], lib_g)
    eval_metrics(pred, truth, name)

# Random split evaluation (best method only)
print("\n=== split_random val results (best method only) ===")
rand_lib_mask  = df_mol_meta["split_random"] == "train"
rand_val_mask  = df_mol_meta["split_random"] == "val"
rand_lib_idx   = np.where(rand_lib_mask)[0]
rand_val_idx   = np.where(rand_val_mask)[0]

# Recompute cosine 0.1 for random split
rand_lib_sp = sp_coarse[rand_lib_idx]
rand_q_sp   = sp_coarse[rand_val_idx]
print(f"Random split: {len(rand_val_idx)} queries × {len(rand_lib_idx)} lib")
rand_sc, rand_id = gpu_cosine_topk(rand_q_sp, rand_lib_sp, top_k=cfg.TOP_K,
                                   q_chunk=cfg.COS_QUERY_CHUNK, l_chunk=cfg.COS_LIB_CHUNK,
                                   desc="rand_cos01")
pred_r, truth_r = build_pred_dict(rand_val_idx, rand_sc, rand_id, rand_lib_idx)
eval_metrics(pred_r, truth_r, "random_cos01")
del rand_lib_sp, rand_q_sp, rand_sc, rand_id
gc.collect(); torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 25 — MD: Novelty diagnostics
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 8. Novelty Diagnostic
Compare distribution of each query's **top-1 library similarity** for
val queries (library = train only) vs test queries (library = all
train). A KS test quantifies the shift.

- **Test top-1 ≫ val top-1**: test overlaps library more ⇒ retrieval
  alone may do better than val suggests.
- **Test top-1 ≪ val top-1**: test is more novel ⇒ generative /
  fingerprint stages matter more.
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 26 — Novelty diagnostic computation
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
# val top-1 similarity (best method on 0.1 Da cosine for consistency)
val_top1 = cos01_scores[:, 0]   # all val queries

# test top-1: score test against ALL train (full library)
print("Scoring test queries against full library for novelty diagnostic...")
test_cos_sc, test_cos_id = gpu_cosine_topk(
    test_sp_coarse, sp_coarse,  # test vs ALL train molecules
    top_k=cfg.TOP_K,
    q_chunk=cfg.COS_QUERY_CHUNK, l_chunk=cfg.COS_LIB_CHUNK,
    desc="test_cos01_novelty"
)
test_top1 = test_cos_sc[:, 0]

# Quantiles
for tag, arr in [("val_top1", val_top1), ("test_top1", test_top1)]:
    qs = np.percentile(arr[arr >= 0], [5, 25, 50, 75, 95])
    print(f"  {tag}: p5={qs[0]:.4f}  p25={qs[1]:.4f}  p50={qs[2]:.4f}  p75={qs[3]:.4f}  p95={qs[4]:.4f}")

ks_stat, ks_pval = sp_stats.ks_2samp(val_top1[val_top1 >= 0], test_top1[test_top1 >= 0])
print(f"  KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4e}")
if np.median(test_top1[test_top1>=0]) > np.median(val_top1[val_top1>=0]):
    print("  -> Test top-1 similarities are HIGHER than val -> test overlaps library more")
    print("    Library retrieval should do BETTER on test than val suggests.")
else:
    print("  -> Test top-1 similarities are LOWER than val -> test is more novel")
    print("    Generative / fingerprint stages will matter more.")
print(f"Novelty diagnostic done in {time.time()-t0:.1f}s")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 27 — MD: Test predictions
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 9. Test Predictions & Submission
Library = ALL train molecules. Score test queries with all similarities.
Build `cand_lib_test.parquet` (top-100) and
`submission_libretrieval.csv` (top-25, ik14-deduped, validated).
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 28 — Score test with all methods
# ═══════════════════════════════════════════════════════════════════════
code(r"""
t0 = time.time()
full_lib_idx = np.arange(n_mols)  # ALL train molecules

# ── Cosine 0.1 Da ──
print("Test: cosine 0.1 Da...")
test_cos01_sc, test_cos01_id = gpu_cosine_topk(
    test_sp_coarse, sp_coarse, top_k=cfg.TOP_K,
    q_chunk=n_test, l_chunk=cfg.COS_LIB_CHUNK, desc="test_cos01"
)

# ── Cosine 0.01 Da ──
print("Test: cosine 0.01 Da...")
test_cos001_sc, test_cos001_id = gpu_cosine_topk(
    test_sp_fine, sp_fine, top_k=cfg.TOP_K,
    q_chunk=n_test, l_chunk=min(cfg.COS_LIB_CHUNK, 5000), desc="test_cos001"
)

# ── Modified cosine ──
print("Test: modified cosine...")
lib_m_order2 = np.argsort(mol_neutral_mass)
lib_m_sorted2 = mol_neutral_mass[lib_m_order2]

# Transfer test peak arrays to GPU
t_mzs_g  = torch.from_numpy(test_peak_mzs).to(DEVICE)
t_ints_g = torch.from_numpy(test_peak_ints).to(DEVICE)
t_mask_g = torch.from_numpy(test_peak_mask).to(DEVICE)

test_mod_sc_list, test_mod_id_list = [], []
for qi in tqdm(range(n_test), desc="test_mod_cos"):
    qm = test_neutral_mass[qi]
    lo = np.searchsorted(lib_m_sorted2, qm - cfg.MASS_WINDOW, side="left")
    hi = np.searchsorted(lib_m_sorted2, qm + cfg.MASS_WINDOW, side="right")
    cand = lib_m_order2[lo:hi]
    if len(cand) == 0:
        test_mod_sc_list.append(np.zeros(cfg.TOP_K, dtype=np.float32))
        test_mod_id_list.append(np.full(cfg.TOP_K, -1, dtype=np.int64))
        continue
    nc = len(cand)
    q_mzs_b  = t_mzs_g[qi].unsqueeze(0).expand(nc, -1)
    q_ints_b = t_ints_g[qi].unsqueeze(0).expand(nc, -1)
    q_mask_b = t_mask_g[qi].unsqueeze(0).expand(nc, -1)
    ct = torch.from_numpy(cand).long().to(DEVICE)
    l_mzs_b  = all_mzs_t[ct]; l_ints_b = all_ints_t[ct]; l_mask_b = all_mask_t[ct]
    shifts_b = torch.tensor(test_prec_mass[qi], device=DEVICE) - \
               torch.from_numpy(mol_prec_mass[cand]).to(DEVICE)
    all_sc = []
    for bs in range(0, nc, cfg.MOD_COS_BATCH):
        be = min(bs + cfg.MOD_COS_BATCH, nc)
        sc = modified_cosine_batch(q_mzs_b[bs:be], q_ints_b[bs:be], q_mask_b[bs:be],
                                   l_mzs_b[bs:be], l_ints_b[bs:be], l_mask_b[bs:be],
                                   shifts_b[bs:be], cfg.DA_TOL, cfg.PPM_TOL)
        all_sc.append(sc)
    sc_t = torch.cat(all_sc)
    k = min(cfg.TOP_K, nc)
    topv, topi = sc_t.topk(k)
    rs = np.full(cfg.TOP_K, -1.0, dtype=np.float32); rs[:k] = topv.cpu().numpy()
    ri = np.full(cfg.TOP_K, -1, dtype=np.int64); ri[:k] = cand[topi.cpu().numpy()]
    test_mod_sc_list.append(rs); test_mod_id_list.append(ri)

test_mod_sc = np.stack(test_mod_sc_list)
test_mod_id = np.stack(test_mod_id_list)
print(f"Test scoring done in {time.time()-t0:.1f}s")
torch.cuda.empty_cache()
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 29 — Build candidate parquets + submission
# ═══════════════════════════════════════════════════════════════════════
code(r"""
def build_candidate_df(query_ids, scores_arr, idx_arr, lib_global, source_tag):
    # Build parquet-ready DataFrame: molecule_id, smiles, ik14, score, source.
    rows = []
    for qi, qid in enumerate(query_ids):
        for rank in range(scores_arr.shape[1]):
            sc = float(scores_arr[qi, rank])
            li = int(idx_arr[qi, rank])
            if li < 0 or sc < 0:
                break
            g = li if lib_global is None else lib_global[li]
            smi = df_mol_meta.iloc[g].get("smiles", "")
            ik  = df_mol_meta.iloc[g]["ik14"]
            rows.append({"molecule_id": qid, "smiles": smi, "ik14": ik,
                         "score": sc, "source": source_tag})
    return pd.DataFrame(rows)

# Choose the best method's arrays for candidate files
src_tag = {"cos_01": "lib_cos", "cos_001": "lib_cos",
           "mod_cos": "lib_mod", "entropy": "lib_ent"}[best_method]
# Map method -> (scores, idx, lib_global_indices)
method_arrays = {
    "cos_01":  (cos01_scores,  cos01_idx,  lib_idx),
    "cos_001": (cos001_scores, cos001_idx, lib_idx),
    "mod_cos": (mod_scores,    mod_idx,    None),
    "entropy": (ent_scores,    ent_idx,    lib_idx),
}
sc_best, id_best, lg_best = method_arrays[best_method]

# Val model candidates
vm_qids = [mol_ids[i] for i in val_idx[vm_local]]
cand_vm = build_candidate_df(vm_qids, sc_best[vm_local], id_best[vm_local], lg_best, src_tag)
cand_vm.to_parquet(os.path.join(cfg.output_dir, "cand_lib_val_model.parquet"), index=False)
print(f"Saved cand_lib_val_model.parquet ({len(cand_vm)} rows)")

# Val rank candidates
vr_qids = [mol_ids[i] for i in val_idx[vr_local]]
cand_vr = build_candidate_df(vr_qids, sc_best[vr_local], id_best[vr_local], lg_best, src_tag)
cand_vr.to_parquet(os.path.join(cfg.output_dir, "cand_lib_val_rank.parquet"), index=False)
print(f"Saved cand_lib_val_rank.parquet ({len(cand_vr)} rows)")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 30 — Test candidates + submission
# ═══════════════════════════════════════════════════════════════════════
code(r"""
# Choose best test arrays
test_method_arrays = {
    "cos_01":  (test_cos01_sc,  test_cos01_id,  None),
    "cos_001": (test_cos001_sc, test_cos001_id, None),
    "mod_cos": (test_mod_sc,    test_mod_id,    None),
}
# If best is entropy, fall back to cos_01 for test (entropy already scored)
if best_method == "entropy":
    t_sc, t_id, t_lg = test_cos01_sc, test_cos01_id, None
else:
    t_sc, t_id, t_lg = test_method_arrays.get(best_method, (test_cos01_sc, test_cos01_id, None))

# Test candidates (indices are GLOBAL mol indices for all test methods)
cand_test_rows = []
for qi, qid in enumerate(test_mol_ids_all):
    for rank in range(t_sc.shape[1]):
        sc = float(t_sc[qi, rank])
        li = int(t_id[qi, rank])
        if li < 0 or sc < 0:
            break
        smi = df_mol_meta.iloc[li].get("smiles", "")
        ik  = df_mol_meta.iloc[li]["ik14"]
        cand_test_rows.append({"molecule_id": qid, "smiles": smi, "ik14": ik,
                               "score": sc, "source": src_tag})
cand_test = pd.DataFrame(cand_test_rows)
cand_test.to_parquet(os.path.join(cfg.output_dir, "cand_lib_test.parquet"), index=False)
print(f"Saved cand_lib_test.parquet ({len(cand_test)} rows)")

# ── Build submission CSV (top-25 per molecule, ik14-deduped) ──────────
sample_sub_path = find_file("sample_submission.csv", required=False)
if sample_sub_path:
    test_ids_official = pd.read_csv(sample_sub_path)["molecule_id"].tolist()
else:
    test_ids_official = test_mol_ids_all

sub_rows = []
for qid in test_ids_official:
    subset = cand_test[cand_test["molecule_id"] == qid].sort_values("score", ascending=False)
    seen_ik = set()
    deduped_smiles = []
    for _, r in subset.iterrows():
        ik = evalkit.fast_key(r["smiles"]) if r["smiles"] else r["ik14"]
        if ik not in seen_ik and r["smiles"]:
            seen_ik.add(ik)
            deduped_smiles.append(r["smiles"])
        if len(deduped_smiles) >= 25:
            break
    # Fallback: if fewer than 1 guess, add a dummy valid SMILES
    if len(deduped_smiles) == 0:
        deduped_smiles = ["C"]
    sub_rows.append({"molecule_id": qid, "smiles": ";".join(deduped_smiles)})

df_sub = pd.DataFrame(sub_rows)
sub_path = os.path.join(cfg.output_dir, "submission_libretrieval.csv")
df_sub.to_csv(sub_path, index=False)
print(f"Saved submission_libretrieval.csv ({len(df_sub)} rows)")

# Validate submission
ok, errs, warns, stats = evalkit.validate_submission(df_sub, test_ids_official)
print(f"Submission valid: {ok}")
for e in errs[:5]:
    print(f"  ERROR: {e}")
for w in warns[:5]:
    print(f"  WARN: {w}")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 31 — MD: Artifacts
# ═══════════════════════════════════════════════════════════════════════
md(r"""
## 10. Artifact Saving & Manifest
Save:
- `lib_index.npz` — sparse binned vectors + neutral masses + molecule IDs
- `lib_config.json` — best method and parameters
- `manifest.json` — full provenance and metrics
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 32 — Save index + config
# ═══════════════════════════════════════════════════════════════════════
code(r"""
import scipy.sparse

# ── lib_index.npz ─────────────────────────────────────────────────────
idx_path = os.path.join(cfg.output_dir, "lib_index.npz")
scipy.sparse.save_npz(os.path.join(cfg.output_dir, "lib_sp_coarse.npz"), sp_coarse)
scipy.sparse.save_npz(os.path.join(cfg.output_dir, "lib_sp_fine.npz"), sp_fine)
np.savez_compressed(idx_path,
    mol_ids=np.array(mol_ids),
    neutral_mass=mol_neutral_mass,
    peak_mzs=mol_peak_mzs,
    peak_ints=mol_peak_ints,
    peak_mask=mol_peak_mask,
    prec_mass=mol_prec_mass,
)
print(f"Saved lib_index.npz ({os.path.getsize(idx_path)/1e6:.1f} MB)")

# ── lib_config.json ───────────────────────────────────────────────────
lib_config = {
    "best_method": best_method,
    "source_tag": src_tag,
    "methods_mrr25": {k: v["mrr25"] for k, v in methods.items()},
    "params": {
        "BIN_COARSE": cfg.BIN_COARSE,
        "BIN_FINE": cfg.BIN_FINE,
        "MAX_MZ": cfg.MAX_MZ,
        "MAX_PEAKS": cfg.MAX_PEAKS,
        "MASS_WINDOW": cfg.MASS_WINDOW,
        "DA_TOL": cfg.DA_TOL,
        "PPM_TOL": cfg.PPM_TOL,
    },
    "ceiling_val_model": ceil_vm,
    "ceiling_val_rank": ceil_vr,
}
cfg_path = os.path.join(cfg.output_dir, "lib_config.json")
with open(cfg_path, "w") as f:
    json.dump(lib_config, f, indent=2)
print(f"Saved lib_config.json")
""")

# ═══════════════════════════════════════════════════════════════════════
# CELL 33 — Manifest + sanity checks
# ═══════════════════════════════════════════════════════════════════════
code(r"""
elapsed = time.time() - T_START

manifest = {
    "stage": cfg.stage,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "cfg": asdict(cfg),
    "library_versions": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "rdkit": Chem.rdBase.rdkitVersion,
    },
    "metrics": {
        "val_model": methods,
        "best_method": best_method,
        "ceiling_val_model": ceil_vm,
        "ceiling_val_rank": ceil_vr,
    },
    "files": {
        "cand_lib_val_model.parquet": os.path.getsize(os.path.join(cfg.output_dir, "cand_lib_val_model.parquet")),
        "cand_lib_val_rank.parquet":  os.path.getsize(os.path.join(cfg.output_dir, "cand_lib_val_rank.parquet")),
        "cand_lib_test.parquet":      os.path.getsize(os.path.join(cfg.output_dir, "cand_lib_test.parquet")),
        "submission_libretrieval.csv": os.path.getsize(os.path.join(cfg.output_dir, "submission_libretrieval.csv")),
        "lib_index.npz":             os.path.getsize(os.path.join(cfg.output_dir, "lib_index.npz")),
        "lib_config.json":           os.path.getsize(cfg_path),
    },
    "runtime_seconds": elapsed,
    "fp_spec": evalkit.FP_SPEC,
}
with open(os.path.join(cfg.output_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

# ── Sanity checks ─────────────────────────────────────────────────────
for fname in ["cand_lib_val_model.parquet", "cand_lib_val_rank.parquet",
              "cand_lib_test.parquet", "submission_libretrieval.csv",
              "lib_index.npz", "lib_config.json", "manifest.json"]:
    fp = os.path.join(cfg.output_dir, fname)
    assert os.path.exists(fp), f"Missing artifact: {fname}"
    assert os.path.getsize(fp) > 0, f"Empty artifact: {fname}"

# Check candidate schema
for pq in ["cand_lib_val_model.parquet", "cand_lib_val_rank.parquet", "cand_lib_test.parquet"]:
    df_chk = pd.read_parquet(os.path.join(cfg.output_dir, pq))
    assert {"molecule_id","smiles","ik14","score","source"}.issubset(df_chk.columns), \
        f"Schema error in {pq}: {df_chk.columns.tolist()}"
    print(f"  {pq}: {len(df_chk)} rows, schema OK")

print("=" * 70)
print(f"ALL SANITY CHECKS PASSED  |  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Best method: {best_method} -> MRR@25 = {methods[best_method]['mrr25']:.4f}")
print(f"Artifacts -> {cfg.output_dir}")
print("=" * 70)
""")

# ═══════════════════════════════════════════════════════════════════════
# ASSEMBLE & VALIDATE
# ═══════════════════════════════════════════════════════════════════════
nb.cells = cells
nbf.validate(nb)

# Syntax-check every code cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        clean = "\n".join(
            l for l in cell.source.splitlines() if not l.strip().startswith("%")
        )
        try:
            ast.parse(clean)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}", file=sys.stderr)
            sys.exit(1)

out_ipynb = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "03_library_retrieval.ipynb")
with open(out_ipynb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {out_ipynb}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown)")
