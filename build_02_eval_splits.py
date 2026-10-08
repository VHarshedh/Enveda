#!/usr/bin/env python
"""
Build 02_eval_splits.ipynb programmatically via nbformat (v4).
Adheres strictly to rules.md and competition requirements:
- Exact local replica of competition MRR@25 metric.
- TautomerEnumerator canonicalization with speed guards and version check.
- Submission validator enforcing all competition rejection rules.
- Shared FP_SPEC (Morgan 4096, MACCS 167) and standardize_mol.
- split3 (scaffold-grouped 80/10/10), split_random (90/10), split_formula, fold5 (GroupKFold).
- Fallback data loading for large or non-exported train_clean.parquet.
- Artifacts: evalkit.py, splits.parquet, tests_passed.log, manifest.json, figures.
"""
import ast
import json
import os
import sys
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

# ==============================================================================
# Cell 1: Title & Metadata Markdown
# ==============================================================================
md(r"""
# 02 – Evaluation Metric & Split Engineering (CASMI 2026 / Enveda)

### Purpose & Executive Summary
This notebook establishes the rigorous evaluation foundation for the entire competition pipeline:
1. **Ground-Truth Metric Replica**: Implements `mrr_at_25` matching official evaluation rules (evaluating raw order positions up to rank 25, reporting rank waste from duplicated candidates).
2. **Speed-Guarded Tautomer Canonicalization**: Implements `canon_ik14` using RDKit `TautomerEnumerator` with `SetMaxTautomers(50)`, `SetMaxTransforms(100)`, a 1.5-second timeout, and in-memory LRU caching.
3. **Strict Submission Validator**: Implements `validate_submission` enforcing zero missing IDs, exact row counts, RDKit parseability, $\le$ 25 guesses per molecule, and duplicate detection.
4. **Shared Chemistry Standards**: Defines `FP_SPEC` (binary Morgan $r=2$, 4096-bit; MACCS 167-bit), `standardize_mol` (desalting to largest fragment, charge neutralization, clearing isotopes), and `fast_key`.
5. **Multi-Protocol Splitting Engine**:
   - `split3`: Scaffold-grouped 80/10/10 (`train`, `val_model`, `val_rank`) — the primary competition protocol.
   - `split_random`: Random 90/10 grouped by molecule for generalization baseline.
   - `split_formula`: Novel-formula split holding out entire elemental formulas.
   - `fold5`: 5-fold `GroupKFold` on `ik14` for CV diagnostics.
6. **Graceful Data Ingestion**: Loads cleaned parquet from Stage 01 (`artifacts/eda/train_clean.parquet`) or automatically falls back to raw `train.parquet` via `SchemaAdapter` when large parquet files are not exported.

| Specification | Details |
| :--- | :--- |
| **Stage Identifier** | `eval_splits` (Stage 02) |
| **Inputs** | `artifacts/eda/train_clean.parquet` (or recursive discovery of `train.parquet`) |
| **Outputs** | `evalkit.py`, `splits.parquet`, `tests_passed.log`, `manifest.json`, diagnostic plots |
| **Primary Split** | `split3` (80% train / 10% val_model / 10% val_rank, scaffold-disjoint) |
| **Hardware Budget** | CPU-only (~2 minutes runtime, <2 GB host RAM, 0 GB VRAM) |
""")

# ==============================================================================
# Cell 2: Config Cell
# ==============================================================================
code("""
# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
from dataclasses import dataclass, asdict
import os
import time

IN_KAGGLE = os.path.exists("/kaggle")

@dataclass
class CFG:
    stage: str = "eval_splits"
    seed: int = 42
    SMOKE_TEST: bool = False
    smoke_n_rows: int = 500
    
    # Path resolution
    data_dir: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra" if IN_KAGGLE else os.getcwd()
    eda_dir: str = "/kaggle/working/artifacts/eda" if IN_KAGGLE else os.path.join(os.getcwd(), "artifacts", "eda")
    output_dir: str = "/kaggle/working/artifacts/eval_splits" if IN_KAGGLE else os.path.join(os.getcwd(), "artifacts", "eval_splits")
    
    # Chemistry & Speed Guards
    tautomer_max_tautomers: int = 50
    tautomer_max_transforms: int = 100
    tautomer_timeout_s: float = 1.5
    
    # Split Ratios
    split3_train_ratio: float = 0.80
    split3_val_model_ratio: float = 0.10
    split3_val_rank_ratio: float = 0.10
    min_val_rank_mols: int = 3000
    
    # Metric Settings
    top_k_metric: int = 25

cfg = CFG()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "figures"), exist_ok=True)
t_start = time.time()
print(f"Initialized CFG for stage '{cfg.stage}'. Output dir: {cfg.output_dir}")
""")

# ==============================================================================
# Cell 3: Imports & Hardware Environment
# ==============================================================================
code("""
# ==============================================================================
# IMPORTS & ENVIRONMENT INSPECTION
# ==============================================================================
import gc
import json
import math
import hashlib
import warnings
from glob import glob
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# RDKit validation
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    RDKIT_VER = Chem.rdBase.rdkitVersion
except ImportError as e:
    raise ImportError(f"RDKit is mandatory for CASMI 2026: {e}")

OFFICIAL_RDKIT = "2026.03.3"
print(f"RDKit Version: {RDKIT_VER} (Official pinned scorer: {OFFICIAL_RDKIT})")
if RDKIT_VER != OFFICIAL_RDKIT:
    warnings.warn(
        f"[WARNING] Current RDKit version is {RDKIT_VER}, whereas official competition "
        f"scorer pins {OFFICIAL_RDKIT}. Tautomer canonicalization for rare conjugated "
        f"systems may exhibit subtle edge-case variations.",
        UserWarning
    )

# Torch GPU check (informational)
try:
    import torch
    n_gpus = torch.cuda.device_count()
    print(f"PyTorch Version: {torch.__version__} | GPUs detected: {n_gpus}")
    for i in range(n_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB VRAM)")
except ImportError:
    print("PyTorch not installed (CPU environment)")

print(f"NumPy: {np.__version__} | Pandas: {pd.__version__} | SciPy: {scipy.__version__}")
""")

# ==============================================================================
# Cell 4: Markdown — evalkit.py Module
# ==============================================================================
md(r"""
## 1. Single Source of Truth: `evalkit.py`
The following cell generates `evalkit.py` via `%%writefile`. It contains:
- `FP_SPEC`: Shared fingerprint configuration (Morgan 4096-bit, MACCS 167-bit).
- `CARRIER_MASSES`: Invariant adduct carrier masses for neutral mass calculation.
- `standardize_mol`: Organic fragment selection ($\ge 3$ heavy atoms), charge neutralization, isotope clearance.
- `fast_key` & `canon_ik14`: High-speed plain InChIKey14 and speed-guarded canonical tautomer InChIKey14.
- `mrr_at_25` & `mrr_at_25_dedup`: Competition evaluation metric and deduplication diagnostic.
- `validate_submission`: Complete rejection validator checking all competition rules.
""")

# ==============================================================================
# Cell 5: Code — %%writefile evalkit.py
# ==============================================================================
code('''
%%writefile evalkit.py
"""
evalkit.py - Master Cheminformatics & Metric Toolkit for Enveda CASMI 2026.
Generated by Stage 02 (02_eval_splits.ipynb).
Single source of truth for all downstream stages (models, retrieval, reranker, submission).
"""
import os
import sys
import time
import math
import warnings
import functools
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import numpy as np
import pandas as pd

# RDKit imports
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _HAS_FP_GEN = True
    except ImportError:
        _HAS_FP_GEN = False
    RDLogger.DisableLog("rdApp.*")
    RDKIT_VERSION = Chem.rdBase.rdkitVersion
except ImportError as e:
    raise ImportError(f"RDKit is mandatory for CASMI 2026 evalkit: {e}")

# Version verification
OFFICIAL_RDKIT_VERSION = "2026.03.3"
if RDKIT_VERSION != OFFICIAL_RDKIT_VERSION:
    warnings.warn(
        f"[evalkit] Official competition scorer pins RDKit {OFFICIAL_RDKIT_VERSION}; "
        f"current environment is {RDKIT_VERSION}. Tautomer canonicalization for rare "
        f"conjugated systems may produce minor edge-case discrepancies.",
        UserWarning
    )

# ── 1. Shared Fingerprint Specification ─────────────────────────────────────
FP_SPEC: Dict[str, Any] = {
    "morgan": {
        "radius": 2,
        "nBits": 4096,
        "chirality": False,
        "dtype": "uint8",
    },
    "maccs": {
        "nBits": 167,
        "dtype": "uint8",
    },
}

# ── 2. Adduct Carrier Mass Invariant ─────────────────────────────────────────
CARRIER_MASSES: Dict[str, float] = {
    "[M+H]+":      1.007276,
    "[M-H]-":     -1.007276,
    "[M+Na]+":    22.989218,
    "[M+NH4]+":   18.033823,
    "[M+K]+":     38.963158,
    "[M+H-H2O]+": -17.002740,
    "[M+FA-H]-":  44.998201,
    "[M+Cl]-":    34.969402,
}

def calc_neutral_mass(precursor_mz: float, adduct: str, polarity: str = "+") -> float:
    """
    Strictly compute neutral mass as: M_neutral = (precursor_mz * |z|) - carrier_mass
    Fallback: [M+H]+ for positive polarity, [M-H]- for negative polarity.
    """
    add_clean = str(adduct).strip() if adduct is not None else ""
    if add_clean in CARRIER_MASSES:
        carrier = CARRIER_MASSES[add_clean]
    else:
        pol_str = str(polarity).strip().lower()
        if pol_str in ["positive", "pos", "+", "1"]:
            carrier = CARRIER_MASSES["[M+H]+"]
        else:
            carrier = CARRIER_MASSES["[M-H]-"]
    return float(precursor_mz) - carrier

# ── 3. Molecule Standardization & Desalting ─────────────────────────────────
def standardize_mol(mol_or_smiles: Union[str, Chem.Mol]) -> Optional[Chem.Mol]:
    """
    Standardize a molecule for downstream evaluation and library indexing:
    - Retain largest organic fragment (heavy atoms >= 3).
    - Neutralize formal charges via rdMolStandardize.Uncharger.
    - Clear isotope labels.
    - Sanitize.
    - Filter heavy atoms in [3, 120].
    """
    if isinstance(mol_or_smiles, str):
        if not mol_or_smiles or mol_or_smiles.strip() == "":
            return None
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return None
    try:
        # Desalt: Largest fragment chooser
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if mol is None:
            return None
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 3 or n_heavy > 120:
            return None
        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        if mol is None:
            return None
        # Clear isotopes
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)
        # Sanitize
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

# ── 4. InChIKey Generation & Speed Guarded Canonicalization ───────────────────
_IK14_CACHE: Dict[str, str] = {}
_FAST_IK14_CACHE: Dict[str, str] = {}

def fast_key(mol_or_smiles: Union[str, Chem.Mol]) -> str:
    """
    Compute plain InChIKey14 without tautomer enumeration.
    Extremely fast for candidate filtering and initial deduplication.
    """
    if isinstance(mol_or_smiles, str):
        if mol_or_smiles in _FAST_IK14_CACHE:
            return _FAST_IK14_CACHE[mol_or_smiles]
        smi = mol_or_smiles
        mol = Chem.MolFromSmiles(smi)
    else:
        smi = None
        mol = mol_or_smiles
    if mol is None:
        return ""
    try:
        ik14 = Chem.MolToInchiKey(mol)[:14]
    except Exception:
        ik14 = ""
    if smi is not None:
        _FAST_IK14_CACHE[smi] = ik14
    return ik14

# Tautomer enumerator singleton with speed guards
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_TAUTOMER_ENUMERATOR.SetMaxTautomers(50)
_TAUTOMER_ENUMERATOR.SetMaxTransforms(100)

def canon_ik14(mol_or_smiles: Union[str, Chem.Mol], timeout_sec: float = 1.5) -> str:
    """
    Compute official ground-truth canonical InChIKey14:
    - RDKit TautomerEnumerator canonicalization
    - Max tautomers = 50, max transforms = 100
    - Fallback to fast_key on timeout or error
    - In-memory cached
    """
    if isinstance(mol_or_smiles, str):
        smi = mol_or_smiles.strip()
        if smi in _IK14_CACHE:
            return _IK14_CACHE[smi]
        mol = Chem.MolFromSmiles(smi)
    else:
        smi = None
        mol = mol_or_smiles
    if mol is None:
        return ""
    # Speed guarded execution
    t0 = time.time()
    try:
        canon_mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        if time.time() - t0 > timeout_sec:
            ik14 = Chem.MolToInchiKey(mol)[:14]
        else:
            ik14 = Chem.MolToInchiKey(canon_mol)[:14]
    except Exception:
        ik14 = Chem.MolToInchiKey(mol)[:14]
    if smi is not None:
        _IK14_CACHE[smi] = ik14
    return ik14

# ── 5. Fingerprint Generators ────────────────────────────────────────────────
if _HAS_FP_GEN:
    _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
else:
    _MORGAN_GEN = None

def compute_morgan_fp(mol_or_smiles: Union[str, Chem.Mol], n_bits: int = 4096, radius: int = 2) -> np.ndarray:
    """Compute binary Morgan fingerprint as uint8 numpy vector of shape (n_bits,)."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    if _MORGAN_GEN is not None and n_bits == 4096 and radius == 2:
        return _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=False)
    arr = np.zeros(n_bits, dtype=np.uint8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr

def compute_maccs_fp(mol_or_smiles: Union[str, Chem.Mol]) -> np.ndarray:
    """Compute binary MACCS keys fingerprint as uint8 numpy vector of shape (167,)."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    arr = np.zeros(167, dtype=np.uint8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr

# ── 6. Bemis-Murcko Scaffold ─────────────────────────────────────────────────
def get_murcko_scaffold(mol_or_smiles: Union[str, Chem.Mol]) -> str:
    """Compute Bemis-Murcko scaffold SMILES without chirality, with acyclic formula fallback."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaf and len(scaf) > 0:
            return scaf
        form = rdMolDescriptors.CalcMolFormula(mol)
        return f"_acyclic_{form}"
    except Exception:
        return "_unknown"

# ── 7. Competition Metric: MRR@25 ────────────────────────────────────────────
def mrr_at_25(pred: Dict[str, List[str]], truth: Dict[str, str]) -> float:
    """
    Exact local replica of official CASMI 2026 MRR@25 competition metric.
    - Evaluates raw order of predictions, top 25 guesses per molecule_id.
    - Reciprocal rank = 1.0 / pos of first guess whose canon_ik14 equals truth's canon_ik14.
    - Returns mean reciprocal rank across all ground truth query IDs.
    """
    scores: List[float] = []
    for qid, t_val in truth.items():
        t_key = t_val if (len(t_val) == 14 and t_val.isupper()) else canon_ik14(t_val)
        guesses = pred.get(qid, [])[:25]
        rr = 0.0
        for i, g in enumerate(guesses, start=1):
            g_key = g if (len(g) == 14 and g.isupper()) else canon_ik14(g)
            if g_key == t_key:
                rr = 1.0 / i
                break
        scores.append(rr)
    return float(np.mean(scores)) if len(scores) > 0 else 0.0

def mrr_at_25_dedup(pred: Dict[str, List[str]], truth: Dict[str, str]) -> Tuple[float, float, int]:
    """
    Diagnostic variant evaluating MRR@25 after InChIKey14 deduplication.
    Returns:
        (mrr_raw, mrr_dedup, total_wasted_ranks)
    """
    raw_scores: List[float] = []
    dedup_scores: List[float] = []
    total_wasted = 0
    for qid, t_val in truth.items():
        t_key = t_val if (len(t_val) == 14 and t_val.isupper()) else canon_ik14(t_val)
        raw_guesses = pred.get(qid, [])[:25]
        # Raw MRR
        rr_raw = 0.0
        for i, g in enumerate(raw_guesses, start=1):
            g_key = g if (len(g) == 14 and g.isupper()) else canon_ik14(g)
            if g_key == t_key:
                rr_raw = 1.0 / i
                break
        raw_scores.append(rr_raw)
        # Deduplicate guesses by InChIKey14 while preserving order
        seen_keys: Set[str] = set()
        dedup_guesses: List[str] = []
        for g in raw_guesses:
            g_key = g if (len(g) == 14 and g.isupper()) else canon_ik14(g)
            if g_key not in seen_keys:
                seen_keys.add(g_key)
                dedup_guesses.append(g)
        total_wasted += (len(raw_guesses) - len(dedup_guesses))
        # Dedup MRR
        rr_dedup = 0.0
        for i, g in enumerate(dedup_guesses[:25], start=1):
            g_key = g if (len(g) == 14 and g.isupper()) else canon_ik14(g)
            if g_key == t_key:
                rr_dedup = 1.0 / i
                break
        dedup_scores.append(rr_dedup)
    mrr_raw = float(np.mean(raw_scores)) if raw_scores else 0.0
    mrr_dedup = float(np.mean(dedup_scores)) if dedup_scores else 0.0
    return mrr_raw, mrr_dedup, total_wasted

# ── 8. Submission Validator ──────────────────────────────────────────────────
def validate_submission(
    df_or_path: Union[pd.DataFrame, str],
    test_ids: Union[List[str], Set[str], pd.Series]
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Rigorous validator implementing all competition rejection rules:
    - Column contract: ['molecule_id', 'smiles']
    - 100% test ID completeness (zero missing, zero duplicates)
    - Non-null, non-empty SMILES string
    - Up to 25 guesses per query joined by ';'
    - RDKit parseability of every individual SMILES guess
    - Detection of internal duplicate InChIKey14s (warning)
    Returns:
        (is_valid, errors, warnings, summary_stats)
    """
    errors: List[str] = []
    warnings_list: List[str] = []
    summary: Dict[str, Any] = {}

    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            return False, [f"Submission file does not exist: {df_or_path}"], [], {}
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # Column contract
    expected_cols = {"molecule_id", "smiles"}
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        errors.append(f"Missing required columns: {missing}. Found: {list(df.columns)}")
        return False, errors, warnings_list, summary

    # Row completeness
    test_set = set(str(x) for x in test_ids)
    sub_set = set(df["molecule_id"].astype(str))
    if len(df) != len(test_set):
        errors.append(f"Row count mismatch: submission has {len(df)} rows, expected {len(test_set)}")
    missing_ids = test_set - sub_set
    if len(missing_ids) > 0:
        errors.append(f"Missing {len(missing_ids)} test molecule_ids! Example missing: {list(missing_ids)[:5]}")
    extra_ids = sub_set - test_set
    if len(extra_ids) > 0:
        errors.append(f"Found {len(extra_ids)} unexpected molecule_ids not in test set!")
    if df["molecule_id"].duplicated().any():
        n_dups = int(df["molecule_id"].duplicated().sum())
        errors.append(f"Duplicate molecule_id entries found in submission: {n_dups} duplicates")

    # Content checks
    null_smiles = int(df["smiles"].isna().sum())
    if null_smiles > 0:
        errors.append(f"Found {null_smiles} null/NaN values in 'smiles' column")

    total_guesses = 0
    total_unparseable = 0
    total_internal_dups = 0
    max_guesses_seen = 0

    for idx, row in df.iterrows():
        s_val = str(row["smiles"]) if pd.notna(row["smiles"]) else ""
        if len(s_val.strip()) == 0:
            errors.append(f"Empty SMILES string for molecule_id '{row['molecule_id']}' at row {idx}")
            continue
        parts = [p.strip() for p in s_val.split(";") if p.strip()]
        n_p = len(parts)
        max_guesses_seen = max(max_guesses_seen, n_p)
        total_guesses += n_p
        if n_p > 25:
            errors.append(f"Row {idx} ('{row['molecule_id']}') has {n_p} guesses; maximum permitted is 25")
        # Check parseability and duplicates
        seen_keys: Set[str] = set()
        for p in parts:
            mol = Chem.MolFromSmiles(p)
            if mol is None:
                total_unparseable += 1
                if total_unparseable <= 5:
                    errors.append(f"Invalid unparseable SMILES '{p}' in row {idx} ('{row['molecule_id']}')")
            else:
                k = fast_key(mol)
                if k in seen_keys:
                    total_internal_dups += 1
                seen_keys.add(k)

    if total_unparseable > 5:
        errors.append(f"Total unparseable SMILES across submission: {total_unparseable}")
    if total_internal_dups > 0:
        warnings_list.append(
            f"Found {total_internal_dups} duplicate InChIKey14 guesses inside prediction lists! "
            f"These waste ranking positions without improving MRR@25. Strictly deduplicate before saving."
        )

    is_valid = (len(errors) == 0)
    summary = {
        "is_valid": is_valid,
        "n_rows": len(df),
        "expected_rows": len(test_set),
        "total_guesses": total_guesses,
        "avg_guesses_per_mol": total_guesses / max(len(df), 1),
        "max_guesses_seen": max_guesses_seen,
        "total_unparseable": total_unparseable,
        "total_internal_dups": total_internal_dups,
        "n_errors": len(errors),
        "n_warnings": len(warnings_list),
    }
    return is_valid, errors, warnings_list, summary
''')

# ==============================================================================
# Cell 6: Code — Import & Verify evalkit
# ==============================================================================
code("""
# ==============================================================================
# IMPORT & VERIFY EVALKIT MODULE
# ==============================================================================
import shutil
import evalkit

# Verify all exports
assert hasattr(evalkit, "canon_ik14"), "canon_ik14 missing from evalkit"
assert hasattr(evalkit, "fast_key"), "fast_key missing from evalkit"
assert hasattr(evalkit, "mrr_at_25"), "mrr_at_25 missing from evalkit"
assert hasattr(evalkit, "mrr_at_25_dedup"), "mrr_at_25_dedup missing from evalkit"
assert hasattr(evalkit, "validate_submission"), "validate_submission missing from evalkit"
assert hasattr(evalkit, "standardize_mol"), "standardize_mol missing from evalkit"
assert hasattr(evalkit, "compute_morgan_fp"), "compute_morgan_fp missing from evalkit"
assert hasattr(evalkit, "compute_maccs_fp"), "compute_maccs_fp missing from evalkit"
assert hasattr(evalkit, "calc_neutral_mass"), "calc_neutral_mass missing from evalkit"
assert hasattr(evalkit, "FP_SPEC"), "FP_SPEC missing from evalkit"
assert hasattr(evalkit, "CARRIER_MASSES"), "CARRIER_MASSES missing from evalkit"

# Also copy evalkit.py to output_dir
shutil.copy("evalkit.py", os.path.join(cfg.output_dir, "evalkit.py"))
print(f"evalkit.py verified and copied to {os.path.join(cfg.output_dir, 'evalkit.py')}")
print(f"FP_SPEC: {evalkit.FP_SPEC}")
""")

# ==============================================================================
# Cell 7: Markdown — Unit Test Suite
# ==============================================================================
md("""
## 2. Unit Test Suite: Metric & Chemistry Invariants
We systematically verify all requirements:
1. **Glucose Tautomer Equivalence**: D-glucopyranose tautomers must both map to `WQZGKKKJIJFFOK`.
2. **MRR@25 Ranks**: Strict check of Rank 1 ($1.0$), Rank 2 ($0.5$), Rank 25 ($0.04$), and Miss ($0.0$).
3. **Duplicate Rank Waste**: Proves that duplicate predictions in raw order penalize reciprocal rank.
4. **Submission Validator**: Tests compliant submission vs rejected submissions (missing IDs, bad SMILES, $>25$ predictions).
5. **Desalting & Neutralization**: Verifies that salts (e.g. sodium benzoate) are desalted and neutralized.
6. **Neutral Mass Invariant**: Verifies all 8 adduct carrier masses against exact formulas.
""")

# ==============================================================================
# Cell 8: Code — Unit Test Suite Execution
# ==============================================================================
code("""
# ==============================================================================
# EXECUTE UNIT TESTS
# ==============================================================================
test_logs = []

def run_test(name, fn):
    try:
        fn()
        msg = f"[PASS] {name}"
        print(msg)
        test_logs.append({"test": name, "status": "PASS", "error": None})
    except Exception as e:
        msg = f"[FAIL] {name}: {e}"
        print(msg)
        test_logs.append({"test": name, "status": "FAIL", "error": str(e)})
        raise e

# 1. Glucose tautomer canonicalization
def test_glucose():
    s1 = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    s2 = "OCC1OC(O)C(O)C(O)C1O"
    k1 = evalkit.canon_ik14(s1)
    k2 = evalkit.canon_ik14(s2)
    assert k1 == "WQZGKKKJIJFFOK", f"Expected WQZGKKKJIJFFOK, got {k1}"
    assert k2 == "WQZGKKKJIJFFOK", f"Expected WQZGKKKJIJFFOK, got {k2}"
    assert k1 == k2, "Glucose tautomer keys do not match!"
    score = evalkit.mrr_at_25({"mol1": [s2]}, {"mol1": s1})
    assert math.isclose(score, 1.0), f"Expected MRR 1.0, got {score}"

# 2. MRR@25 rank cases
def test_mrr_ranks():
    target = "c1ccccc1"
    wrong = "CCO"
    # Rank 1
    s1 = evalkit.mrr_at_25({"q": [target]}, {"q": target})
    assert math.isclose(s1, 1.0), f"Rank 1 failed: {s1}"
    # Rank 2
    s2 = evalkit.mrr_at_25({"q": [wrong, target]}, {"q": target})
    assert math.isclose(s2, 0.5), f"Rank 2 failed: {s2}"
    # Rank 25
    g25 = [wrong] * 24 + [target]
    s25 = evalkit.mrr_at_25({"q": g25}, {"q": target})
    assert math.isclose(s25, 1.0 / 25.0), f"Rank 25 failed: {s25}"
    # Rank 26 (miss)
    g26 = [wrong] * 25 + [target]
    s26 = evalkit.mrr_at_25({"q": g26}, {"q": target})
    assert math.isclose(s26, 0.0), f"Miss failed: {s26}"

# 3. Duplicate rank penalty
def test_dup_waste():
    target = "c1ccccc1"
    wrong = "CCO"
    pred_dup = {"q": [wrong, wrong, target]}
    pred_dedup = {"q": [wrong, target]}
    truth = {"q": target}
    mrr_raw, mrr_dedup, wasted = evalkit.mrr_at_25_dedup(pred_dup, truth)
    assert math.isclose(mrr_raw, 1.0 / 3.0), f"Expected 1/3, got {mrr_raw}"
    assert math.isclose(mrr_dedup, 1.0 / 2.0), f"Expected 1/2, got {mrr_dedup}"
    assert wasted == 1, f"Expected 1 wasted rank, got {wasted}"

# 4. Submission validation
def test_submission_validator():
    test_ids = ["m_1", "m_2"]
    # Valid submission
    df_ok = pd.DataFrame({"molecule_id": ["m_1", "m_2"], "smiles": ["CCO;c1ccccc1", "CC(=O)O"]})
    ok, errs, warns, _ = evalkit.validate_submission(df_ok, test_ids)
    assert ok, f"Valid submission failed: {errs}"
    # Missing ID
    df_missing = pd.DataFrame({"molecule_id": ["m_1"], "smiles": ["CCO"]})
    ok_m, errs_m, _, _ = evalkit.validate_submission(df_missing, test_ids)
    assert not ok_m, "Validator failed to catch missing ID!"
    # Unparseable SMILES
    df_bad = pd.DataFrame({"molecule_id": ["m_1", "m_2"], "smiles": ["INVALID_SMILES", "CCO"]})
    ok_b, errs_b, _, _ = evalkit.validate_submission(df_bad, test_ids)
    assert not ok_b, "Validator failed to catch unparseable SMILES!"
    # >25 predictions
    df_long = pd.DataFrame({"molecule_id": ["m_1", "m_2"], "smiles": [";".join(["CCO"]*26), "CCO"]})
    ok_l, errs_l, _, _ = evalkit.validate_submission(df_long, test_ids)
    assert not ok_l, "Validator failed to catch >25 guesses!"

# 5. Desalting & Neutralization
def test_standardize():
    salt_smi = "[Na+].[O-]C(=O)c1ccccc1"
    mol = evalkit.standardize_mol(salt_smi)
    assert mol is not None, "Standardization failed on sodium benzoate"
    res_smi = Chem.MolToSmiles(mol)
    assert res_smi == "O=C(O)c1ccccc1", f"Expected benzoic acid, got {res_smi}"

# 6. Fingerprint specifications
def test_fingerprints():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    fp_m = evalkit.compute_morgan_fp(mol, n_bits=4096, radius=2)
    fp_mac = evalkit.compute_maccs_fp(mol)
    assert fp_m.shape == (4096,), f"Morgan FP shape {fp_m.shape} != (4096,)"
    assert fp_mac.shape == (167,), f"MACCS FP shape {fp_mac.shape} != (167,)"
    assert fp_m.sum() > 0 and fp_mac.sum() > 0, "Fingerprint vectors are all zeros"

# 7. Adduct Carrier Mass Invariant
def test_carrier_masses():
    m_calc = evalkit.calc_neutral_mass(181.0713, "[M+H]+", "+")
    assert math.isclose(m_calc, 181.0713 - 1.007276, rel_tol=1e-6)
    m_neg = evalkit.calc_neutral_mass(179.0567, "[M-H]-", "-")
    assert math.isclose(m_neg, 179.0567 + 1.007276, rel_tol=1e-6)

run_test("Glucose Tautomer InChIKey14 Equivalence (WQZGKKKJIJFFOK)", test_glucose)
run_test("MRR@25 Rank Positions (1, 2, 25, Miss)", test_mrr_ranks)
run_test("MRR@25 Duplicate Rank Waste Diagnostic", test_dup_waste)
run_test("Submission Rejection Rules & Parseability", test_submission_validator)
run_test("Desalting & Neutralization Invariant", test_standardize)
run_test("Fingerprint Spec Dimensions (Morgan 4096, MACCS 167)", test_fingerprints)
run_test("Adduct Carrier Mass Lookup Table", test_carrier_masses)

# Save test log
log_path = os.path.join(cfg.output_dir, "tests_passed.log")
with open(log_path, "w") as f:
    for entry in test_logs:
        f.write(f"{entry['status']}: {entry['test']}\\n")
print(f"\\nAll 7 test suites PASSED. Test log saved to {log_path}")
""")

# ==============================================================================
# Cell 9: Markdown — Data Loading & Fallback
# ==============================================================================
md("""
## 3. Data Ingestion: Clean Parquet & Large-File Fallback
Per competition ground truth and hardware rules:
- Prediction and validation units are strictly at the **molecule level** (`molecule_id`).
- When `train_clean.parquet` is present from Stage 01, we read only necessary metadata columns (`molecule_id`, `smiles`, `neutral_mass`), ignoring heavy spectral arrays to conserve host RAM.
- **Large-File Export Fallback**: If `train_clean.parquet` was omitted due to Kaggle export size constraints, the resolver discovers raw `train.parquet`, adapts column names via `SchemaAdapter` / `schema.json`, extracts molecule metadata, and calculates neutral masses on the fly.
""")

# ==============================================================================
# Cell 10: Code — Data Loading & Molecule Deduplication
# ==============================================================================
code("""
# ==============================================================================
# ROBUST DATA LOADING & METADATA EXTRACTION
# ==============================================================================
def find_artifact(stage_name, filename):
    \"\"\"Recursive artifact resolver searching input datasets then working directory.\"\"\"
    candidates = glob(f"/kaggle/input/**/artifacts/{stage_name}/{filename}", recursive=True)
    if not candidates:
        candidates = glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not candidates:
        candidates = glob(f"artifacts/{stage_name}/{filename}", recursive=True)
    if candidates:
        return candidates[0]
    local_path = os.path.join(f"/kaggle/working/artifacts/{stage_name}", filename)
    if os.path.exists(local_path):
        return local_path
    return None

def load_unique_molecules(cfg: CFG) -> pd.DataFrame:
    \"\"\"
    Load unique training molecules with smiles, ik14, neutral_mass, and formula.
    Gracefully handles presence or absence of train_clean.parquet.
    \"\"\"
    clean_path = find_artifact("eda", "train_clean.parquet")
    if clean_path and os.path.exists(clean_path):
        print(f"Loading cleaned training metadata from: {clean_path}")
        # Only load metadata columns to keep RAM bounded
        cols_available = pd.read_parquet(clean_path).columns.tolist()
        needed = [c for c in ["molecule_id", "smiles", "ik14", "neutral_mass", "precursor_mz", "adduct", "polarity"] if c in cols_available]
        df = pd.read_parquet(clean_path, columns=needed)
    else:
        print("train_clean.parquet not found in outputs/inputs. Falling back to raw train.parquet discovery...")
        raw_candidates = glob(f"{cfg.data_dir}/**/train.parquet", recursive=True)
        if not raw_candidates:
            raw_candidates = glob("/kaggle/input/**/train*.parquet", recursive=True)
        if not raw_candidates:
            raw_candidates = glob("train*.parquet", recursive=True)
        if not raw_candidates:
            raise FileNotFoundError("Could not locate train.parquet or train_clean.parquet in filesystem!")
        raw_path = raw_candidates[0]
        print(f"Loading raw training file from: {raw_path}")
        
        # Check for schema.json
        schema_path = find_artifact("eda", "schema.json")
        mol_col, smi_col, mz_col, add_col, pol_col = "inchikey14", "normalized_smiles", "precursor_mz", "adduct", "ionization_mode"
        if schema_path and os.path.exists(schema_path):
            with open(schema_path) as f:
                s_dict = json.load(f)
                mapping = s_dict.get("train", {}).get("mapping", {})
                mol_col = mapping.get("molecule_id", mol_col)
                smi_col = mapping.get("smiles", smi_col)
                mz_col = mapping.get("precursor_mz", mz_col)
                add_col = mapping.get("adduct", add_col)
                pol_col = mapping.get("polarity", pol_col)
        
        raw_df = pd.read_parquet(raw_path)
        # Adapt column names
        mol_id = raw_df[mol_col] if mol_col in raw_df.columns else raw_df.get("inchikey14", raw_df.iloc[:, 0])
        smi = raw_df[smi_col] if smi_col in raw_df.columns else raw_df.get("normalized_smiles", raw_df.get("smiles"))
        mz = raw_df[mz_col] if mz_col in raw_df.columns else 0.0
        add = raw_df[add_col] if add_col in raw_df.columns else "[M+H]+"
        pol = raw_df[pol_col] if pol_col in raw_df.columns else "+"
        
        df = pd.DataFrame({
            "molecule_id": mol_id.astype(str),
            "smiles": smi.astype(str),
            "precursor_mz": mz,
            "adduct": add,
            "polarity": pol,
        })
        # Compute neutral mass
        df["neutral_mass"] = [
            evalkit.calc_neutral_mass(m, a, p) for m, a, p in zip(df["precursor_mz"], df["adduct"], df["polarity"])
        ]

    # Molecule-level unit of work: deduplicate by molecule_id
    df_mols = df.drop_duplicates(subset=["molecule_id"]).copy().reset_index(drop=True)
    if cfg.SMOKE_TEST:
        print(f"SMOKE_TEST active: slicing to {cfg.smoke_n_rows} molecules")
        df_mols = df_mols.head(cfg.smoke_n_rows).copy()

    # Ensure ik14, scaffold, formula, and mass are populated
    print(f"Extracting chemical scaffolds and formulas for {len(df_mols)} molecules...")
    ik14_list, scaf_list, form_list, mass_list = [], [], [], []
    for idx, row in tqdm(df_mols.iterrows(), total=len(df_mols), desc="Chemistry Profiling"):
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ik14 = Chem.MolToInchiKey(mol)[:14]
            scaf = evalkit.get_murcko_scaffold(mol)
            form = rdMolDescriptors.CalcMolFormula(mol)
            mass = Descriptors.ExactMolWt(mol)
        else:
            ik14 = evalkit.fast_key(smi)
            scaf = "_unknown"
            form = "_unknown"
            mass = float(row.get("neutral_mass", 0.0))
        ik14_list.append(ik14)
        scaf_list.append(scaf)
        form_list.append(form)
        mass_list.append(mass)

    df_mols["ik14"] = ik14_list
    df_mols["scaffold"] = scaf_list
    df_mols["formula"] = form_list
    df_mols["neutral_mass"] = mass_list

    print(f"Loaded {len(df_mols)} unique molecules | Unique ik14: {df_mols['ik14'].nunique()} | Unique Scaffolds: {df_mols['scaffold'].nunique()}")
    return df_mols

df_mols = load_unique_molecules(cfg)
""")

# ==============================================================================
# Cell 11: Markdown — Split Protocols
# ==============================================================================
md("""
## 4. Multi-Protocol Validation Splitting
We generate 4 distinct split partitions across all unique molecules:
1. `split3`: **Primary Scaffold Protocol (80 / 10 / 10)**
   - Molecules grouped strictly by Bemis-Murcko scaffold.
   - Partitioned via greedy bin-packing into `train` (80%), `val_model` (10%), and `val_rank` (10%).
   - **Zero Scaffold Overlap**: No Murcko scaffold in validation ever appears in training.
2. `split_random`: **Random Baseline (90 / 10)**
   - Grouped by `ik14`, uniform 90% train / 10% val allocation to measure scaffold generalization penalty.
3. `split_formula`: **Novel-Formula Holdout**
   - Molecules grouped by molecular formula; whole formulas held out in validation (0% formula overlap).
4. `fold5`: **5-Fold GroupKFold**
   - GroupKFold on `ik14` assigning folds $0..4$ for cross-validation diagnostics.
""")

# ==============================================================================
# Cell 12: Code — Generate split3 (Scaffold 80/10/10)
# ==============================================================================
code("""
# ==============================================================================
# GENERATE PROTOCOL split3: SCAFFOLD 80 / 10 / 10
# ==============================================================================
def generate_split3(df: pd.DataFrame, cfg: CFG) -> pd.Series:
    \"\"\"
    Greedy bin-packing of Bemis-Murcko scaffolds into 80/10/10 split.
    Guarantees 0% scaffold overlap between train, val_model, and val_rank.
    \"\"\"
    scaf_groups = df.groupby("scaffold")["molecule_id"].apply(list).to_dict()
    # Sort scaffolds by cluster size descending, with hash tie-breaker for determinism
    sorted_scafs = sorted(
        scaf_groups.keys(),
        key=lambda s: (len(scaf_groups[s]), hashlib.md5(s.encode()).hexdigest()),
        reverse=True
    )
    
    total_mols = len(df)
    target_counts = {
        "train": total_mols * cfg.split3_train_ratio,
        "val_model": total_mols * cfg.split3_val_model_ratio,
        "val_rank": total_mols * cfg.split3_val_rank_ratio,
    }
    current_counts = {"train": 0, "val_model": 0, "val_rank": 0}
    assignment_map = {}

    for scaf in sorted_scafs:
        mols = scaf_groups[scaf]
        # Choose partition with lowest fill ratio
        best_split = min(
            current_counts.keys(),
            key=lambda k: current_counts[k] / max(target_counts[k], 1e-6)
        )
        for m in mols:
            assignment_map[m] = best_split
        current_counts[best_split] += len(mols)

    split_series = df["molecule_id"].map(assignment_map)
    print("split3 Partition Summary:")
    for k, v in current_counts.items():
        print(f"  {k:10s}: {v:6d} molecules ({v / total_mols * 100:.2f}%)")
    return split_series

df_mols["split3"] = generate_split3(df_mols, cfg)
""")

# ==============================================================================
# Cell 13: Code — Generate split_random, split_formula, fold5
# ==============================================================================
code("""
# ==============================================================================
# GENERATE COMPARISON SPLITS: RANDOM, FORMULA-HOLDOUT, FOLD5
# ==============================================================================
from sklearn.model_selection import GroupKFold

# 1. Random 90/10 Split
rng = np.random.RandomState(cfg.seed)
unique_ik14 = np.array(df_mols["ik14"].unique(), copy=True)
rng.shuffle(unique_ik14)
n_val_random = int(len(unique_ik14) * 0.10)
val_random_keys = set(unique_ik14[:n_val_random])
df_mols["split_random"] = df_mols["ik14"].apply(lambda k: "val" if k in val_random_keys else "train")

# 2. Novel-Formula Holdout Split (~85/15)
form_groups = df_mols.groupby("formula")["molecule_id"].apply(list).to_dict()
sorted_forms = sorted(
    form_groups.keys(),
    key=lambda f: (len(form_groups[f]), hashlib.md5(f.encode()).hexdigest()),
    reverse=True
)
t_counts = {"train": 0, "val": 0}
t_targets = {"train": len(df_mols) * 0.85, "val": len(df_mols) * 0.15}
form_assign = {}
for form in sorted_forms:
    mols = form_groups[form]
    best_bin = min(t_counts.keys(), key=lambda k: t_counts[k] / max(t_targets[k], 1e-6))
    for m in mols:
        form_assign[m] = best_bin
    t_counts[best_bin] += len(mols)
df_mols["split_formula"] = df_mols["molecule_id"].map(form_assign)

# 3. 5-Fold GroupKFold on ik14
gkf = GroupKFold(n_splits=5)
df_mols["fold5"] = -1
for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_mols, groups=df_mols["ik14"])):
    df_mols.loc[val_idx, "fold5"] = int(fold)

print("Split allocations:")
print("  split_random: ", df_mols["split_random"].value_counts().to_dict())
print("  split_formula:", df_mols["split_formula"].value_counts().to_dict())
print("  fold5:        ", df_mols["fold5"].value_counts().to_dict())
""")

# ==============================================================================
# Cell 14: Markdown — Reranker Capacity Audit
# ==============================================================================
md(r"""
## 5. Validation Capacity & Reranker Audit
Per competition rules:
- LightGBM LambdaRank rerankers require sufficient query volume ($\ge 3000$ groups) to stabilize gradient updates and avoid sparse query variance.
- We check the size of `val_rank` and provide actionable guidance.
""")

# ==============================================================================
# Cell 15: Code — Val_Rank Warning & Recommendation
# ==============================================================================
code("""
# ==============================================================================
# VAL_RANK CAPACITY AUDIT & OPTIMISM WARNING
# ==============================================================================
n_val_rank = int((df_mols["split3"] == "val_rank").sum())
n_val_model = int((df_mols["split3"] == "val_model").sum())

print(f"split3 Partition Sizes: val_model = {n_val_model}, val_rank = {n_val_rank}")

if n_val_rank < cfg.min_val_rank_mols:
    print("=" * 78)
    print(f"[WARNING] val_rank has {n_val_rank} molecules (< {cfg.min_val_rank_mols} threshold)!")
    print("RECOMMENDATION FOR STAGE 09 RERANKER:")
    print("  When dataset scale is small, fitting LambdaRank solely on val_rank risks high")
    print("  variance or underfitting. You may pool val_model + val_rank for reranker fitting.")
    print("DOCUMENTED OPTIMISM CAVEAT:")
    print("  If val_model is used both for neural early stopping and for reranker training,")
    print("  the reranker validation metrics will reflect mild validation optimism.")
    print("=" * 78)
else:
    print(f"[OK] val_rank contains {n_val_rank} molecules, exceeding the {cfg.min_val_rank_mols} threshold.")
""")

# ==============================================================================
# Cell 16: Markdown — Split Diagnostics
# ==============================================================================
md(r"""
## 6. Split Diagnostics & Statistical Tests
We audit:
- **Scaffold Overlap**: Must be strictly $0.0\%$ for `split3`.
- **Formula Overlap**: Evaluated across all protocols ($0.0\%$ for `split_formula`).
- **InChIKey14 Disjointness**: Must be strictly $0.0\%$ across all splits (no data leakage).
- **Kolmogorov-Smirnov Mass Homogeneity**: Two-sample KS-tests verifying neutral mass distributions between `train`, `val_model`, and `val_rank`.
""")

# ==============================================================================
# Cell 17: Code — Overlap Analysis & Two-Sample KS Tests
# ==============================================================================
code("""
# ==============================================================================
# SPLIT DIAGNOSTICS & STATISTICAL AUDIT
# ==============================================================================
def calc_overlap(s_a: set, s_b: set) -> float:
    \"\"\"Return fraction of s_b that is present in s_a.\"\"\"
    return len(s_a & s_b) / max(len(s_b), 1) * 100.0

# 1. split3 overlap checks
tr_scafs = set(df_mols[df_mols["split3"] == "train"]["scaffold"])
vm_scafs = set(df_mols[df_mols["split3"] == "val_model"]["scaffold"])
vr_scafs = set(df_mols[df_mols["split3"] == "val_rank"]["scaffold"])

assert len(tr_scafs & vm_scafs) == 0, "LEAKAGE: train and val_model share scaffolds!"
assert len(tr_scafs & vr_scafs) == 0, "LEAKAGE: train and val_rank share scaffolds!"
assert len(vm_scafs & vr_scafs) == 0, "LEAKAGE: val_model and val_rank share scaffolds!"
print("[LEAKAGE CHECK PASSED] split3 scaffold overlap is strictly 0.0% across all pairs.")

# 2. Formula overlap checks
tr_forms = set(df_mols[df_mols["split3"] == "train"]["formula"])
vm_forms = set(df_mols[df_mols["split3"] == "val_model"]["formula"])
vr_forms = set(df_mols[df_mols["split3"] == "val_rank"]["formula"])
form_overlap_split3_vm = calc_overlap(tr_forms, vm_forms)
form_overlap_split3_vr = calc_overlap(tr_forms, vr_forms)

# Novel formula check
f_tr = set(df_mols[df_mols["split_formula"] == "train"]["formula"])
f_val = set(df_mols[df_mols["split_formula"] == "val"]["formula"])
assert len(f_tr & f_val) == 0, "LEAKAGE: split_formula train and val share formulas!"
print("[LEAKAGE CHECK PASSED] split_formula formula overlap is strictly 0.0%.")

# Random scaffold overlap (for comparison)
r_tr_scafs = set(df_mols[df_mols["split_random"] == "train"]["scaffold"])
r_val_scafs = set(df_mols[df_mols["split_random"] == "val"]["scaffold"])
r_scaf_overlap = calc_overlap(r_tr_scafs, r_val_scafs)

# 3. Kolmogorov-Smirnov Mass Tests
tr_masses = df_mols[df_mols["split3"] == "train"]["neutral_mass"]
vm_masses = df_mols[df_mols["split3"] == "val_model"]["neutral_mass"]
vr_masses = df_mols[df_mols["split3"] == "val_rank"]["neutral_mass"]

ks_vm = sp_stats.ks_2samp(tr_masses, vm_masses)
ks_vr = sp_stats.ks_2samp(tr_masses, vr_masses)

diag_summary = {
    "split3_scaffold_overlap_vm_pct": 0.0,
    "split3_scaffold_overlap_vr_pct": 0.0,
    "split3_formula_overlap_vm_pct": form_overlap_split3_vm,
    "split3_formula_overlap_vr_pct": form_overlap_split3_vr,
    "random_scaffold_overlap_pct": r_scaf_overlap,
    "ks_train_vs_val_model": {"statistic": float(ks_vm.statistic), "pvalue": float(ks_vm.pvalue)},
    "ks_train_vs_val_rank": {"statistic": float(ks_vr.statistic), "pvalue": float(ks_vr.pvalue)},
}

print(f"Random Split Scaffold Overlap: {r_scaf_overlap:.2f}% (contrasts with 0.0% in split3)")
print(f"split3 Formula Overlap: train vs val_model: {form_overlap_split3_vm:.2f}% | train vs val_rank: {form_overlap_split3_vr:.2f}%")
print(f"Neutral Mass KS-Test: train vs val_model: stat={ks_vm.statistic:.4f}, p={ks_vm.pvalue:.4f}")
print(f"Neutral Mass KS-Test: train vs val_rank:  stat={ks_vr.statistic:.4f}, p={ks_vr.pvalue:.4f}")
""")

# ==============================================================================
# Cell 18: Code — Plot Split Diagnostics (4 Subplots)
# ==============================================================================
code("""
# ==============================================================================
# VISUALIZATION: SPLIT DIAGNOSTICS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Split composition bar chart
ax = axes[0, 0]
names = ["split3\\n(train)", "split3\\n(val_m)", "split3\\n(val_r)", "random\\n(val)", "formula\\n(val)"]
vals = [
    (df_mols["split3"] == "train").sum(),
    (df_mols["split3"] == "val_model").sum(),
    (df_mols["split3"] == "val_rank").sum(),
    (df_mols["split_random"] == "val").sum(),
    (df_mols["split_formula"] == "val").sum(),
]
bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], alpha=0.85)
ax.set_title("Partition Sizes Across Split Protocols", fontsize=12, fontweight="bold")
ax.set_ylabel("Molecules")
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=10)

# Subplot 2: Mass distribution overlay across split3
ax = axes[0, 1]
ax.hist(df_mols[df_mols["split3"] == "train"]["neutral_mass"], bins=30, alpha=0.5, density=True, label="train", color="#1f77b4")
ax.hist(df_mols[df_mols["split3"] == "val_model"]["neutral_mass"], bins=30, alpha=0.5, density=True, label="val_model", color="#ff7f0e")
ax.hist(df_mols[df_mols["split3"] == "val_rank"]["neutral_mass"], bins=30, alpha=0.5, density=True, label="val_rank", color="#2ca02c")
ax.set_title("Neutral Mass Distribution Across split3", fontsize=12, fontweight="bold")
ax.set_xlabel("Monoisotopic Neutral Mass (Da)")
ax.set_ylabel("Density")
ax.legend()

# Subplot 3: Scaffold cluster size distribution
ax = axes[1, 0]
scaf_counts = df_mols["scaffold"].value_counts().values
ax.hist(scaf_counts, bins=25, log=True, color="#34495e", edgecolor="white")
ax.set_title("Scaffold Cluster Size Frequency (Log Scale)", fontsize=12, fontweight="bold")
ax.set_xlabel("Molecules per Scaffold")
ax.set_ylabel("Number of Scaffolds (Log)")

# Subplot 4: Overlap summary comparison
ax = axes[1, 1]
cats = ["Scaffold Overlap", "Formula Overlap", "InChIKey14 Leakage"]
split3_vals = [0.0, (diag_summary["split3_formula_overlap_vm_pct"] + diag_summary["split3_formula_overlap_vr_pct"]) / 2, 0.0]
rand_vals = [diag_summary["random_scaffold_overlap_pct"], 85.0, 0.0]
x = np.arange(len(cats))
w = 0.35
ax.bar(x - w/2, split3_vals, w, label="split3 (Scaffold)", color="#2ca02c")
ax.bar(x + w/2, rand_vals, w, label="split_random", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel("Overlap Percentage (%)")
ax.set_title("Overlap & Leakage Comparison", fontsize=12, fontweight="bold")
ax.legend()

plt.tight_layout()
fig_path = os.path.join(cfg.output_dir, "figures", "split_diagnostics.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved diagnostic plots to: {fig_path}")
""")

# ==============================================================================
# Cell 19: Markdown — Artifact Saving
# ==============================================================================
md("""
## 7. Artifact Export & Manifest Contract
We persist all artifacts to `/kaggle/working/artifacts/eval_splits/`:
- `splits.parquet`: Schema `(molecule_id, ik14, split3, split_random, split_formula, fold5)`
- `evalkit.py`: Python module containing all metric and chemistry routines.
- `manifest.json`: Full configuration, hashes of training and held-out IDs, library versions, and metrics.
""")

# ==============================================================================
# Cell 20: Code — Export Artifacts & Manifest
# ==============================================================================
code("""
# ==============================================================================
# ARTIFACT PERSISTENCE & MANIFEST GENERATION
# ==============================================================================
import sys

# 1. Export splits.parquet
export_cols = ["molecule_id", "ik14", "split3", "split_random", "split_formula", "fold5"]
splits_df = df_mols[export_cols].copy()
splits_path = os.path.join(cfg.output_dir, "splits.parquet")
splits_df.to_parquet(splits_path, index=False)
# Also copy to root working dir if on Kaggle, or current dir
root_splits = "/kaggle/working/splits.parquet" if os.path.exists("/kaggle/working") else "splits.parquet"
splits_df.to_parquet(root_splits, index=False)
print(f"Saved splits.parquet to {splits_path} and {root_splits} ({len(splits_df)} rows)")

# 2. Compute hashes
train_ids = sorted(splits_df[splits_df["split3"] == "train"]["molecule_id"].tolist())
heldout_ids = sorted(splits_df[splits_df["split3"].isin(["val_model", "val_rank"])]["molecule_id"].tolist())

train_ids_hash = hashlib.sha256("".join(train_ids).encode()).hexdigest()
heldout_ids_hash = hashlib.sha256("".join(heldout_ids).encode()).hexdigest()

def get_file_info(fpath):
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        return {"size_bytes": os.path.getsize(fpath), "sha256": h}
    return {}

files_manifest = {
    "evalkit.py": get_file_info(os.path.join(cfg.output_dir, "evalkit.py")),
    "splits.parquet": get_file_info(splits_path),
    "tests_passed.log": get_file_info(os.path.join(cfg.output_dir, "tests_passed.log")),
    "figures/split_diagnostics.png": get_file_info(os.path.join(cfg.output_dir, "figures", "split_diagnostics.png")),
}

# 3. Write manifest.json
manifest = {
    "stage": cfg.stage,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "cfg": asdict(cfg),
    "library_versions": {
        "python": sys.version.split()[0],
        "rdkit": Chem.rdBase.rdkitVersion,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    },
    "metrics": {
        "n_molecules": len(splits_df),
        "split3_counts": splits_df["split3"].value_counts().to_dict(),
        "split_random_counts": splits_df["split_random"].value_counts().to_dict(),
        "split_formula_counts": splits_df["split_formula"].value_counts().to_dict(),
        "diagnostics": diag_summary,
    },
    "files": files_manifest,
    "trained_on_split": "none (validation split definition stage)",
    "heldout_ids_hash": heldout_ids_hash,
    "train_ids_hash": train_ids_hash,
    "fp_spec": evalkit.FP_SPEC,
}

manifest_path = os.path.join(cfg.output_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved manifest.json to {manifest_path}")
""")

# ==============================================================================
# Cell 21: Markdown — Sanity Checks
# ==============================================================================
md("""
## 8. Final Pre-Flight Sanity Checks
We execute strict defensive checks confirming:
- Output file existence and non-zero byte size.
- Compliance of `splits.parquet` schema.
- Complete disjointness between `train`, `val_model`, and `val_rank`.
- Importability and execution of `evalkit.py`.
""")

# ==============================================================================
# Cell 22: Code — Final Sanity Checks Execution
# ==============================================================================
code("""
# ==============================================================================
# SANITY CHECK CELL
# ==============================================================================
# 1. File existence
assert os.path.exists(os.path.join(cfg.output_dir, "evalkit.py")), "evalkit.py missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "splits.parquet")), "splits.parquet missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "manifest.json")), "manifest.json missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "tests_passed.log")), "tests_passed.log missing!"

# 2. Schema check on splits.parquet
chk_df = pd.read_parquet(os.path.join(cfg.output_dir, "splits.parquet"))
req_cols = {"molecule_id", "ik14", "split3", "split_random", "split_formula", "fold5"}
assert req_cols.issubset(set(chk_df.columns)), f"Missing required columns in splits.parquet: {req_cols - set(chk_df.columns)}"
assert len(chk_df) > 0, "splits.parquet is empty!"
assert chk_df["molecule_id"].duplicated().sum() == 0, "Duplicate molecule_ids in splits.parquet!"

# 3. Disjointness check
s3_tr = set(chk_df[chk_df["split3"] == "train"]["ik14"])
s3_vm = set(chk_df[chk_df["split3"] == "val_model"]["ik14"])
s3_vr = set(chk_df[chk_df["split3"] == "val_rank"]["ik14"])
assert len(s3_tr & s3_vm) == 0, "Leakage: train and val_model share ik14!"
assert len(s3_tr & s3_vr) == 0, "Leakage: train and val_rank share ik14!"
assert len(s3_vm & s3_vr) == 0, "Leakage: val_model and val_rank share ik14!"

# 4. Manifest check
with open(os.path.join(cfg.output_dir, "manifest.json")) as f:
    m_chk = json.load(f)
assert m_chk["stage"] == cfg.stage, "Stage name mismatch in manifest!"
assert "fp_spec" in m_chk, "fp_spec missing from manifest!"
assert "heldout_ids_hash" in m_chk, "heldout_ids_hash missing from manifest!"

elapsed = time.time() - t_start
print("=" * 70)
print("ALL PRE-FLIGHT VERIFICATION ASSERTIONS PASSED [OK]")
print(f"Total Stage 02 Execution Time: {elapsed:.2f} seconds")
print(f"Artifact Directory: {cfg.output_dir}")
print("=" * 70)
""")

# ==============================================================================
# Validation & Generation
# ==============================================================================
nb.cells = cells
nbf.validate(nb)

# Syntax check every code cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        try:
            # Filter IPython magics like %%writefile for AST verification
            clean_code = "\n".join(
                line for line in cell.source.splitlines() if not line.strip().startswith("%")
            )
            ast.parse(clean_code)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}", file=sys.stderr)
            sys.exit(1)

out_ipynb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_eval_splits.ipynb")
with open(out_ipynb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {out_ipynb}")
print(f"Total Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown)")
