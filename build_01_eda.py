#!/usr/bin/env python
"""Build 01_eda_schema.ipynb programmatically with nbformat."""
import nbformat as nbf
import ast, sys, os

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

# ── Cell 1: Title ──────────────────────────────────────────────────────────
md("""
# 01 – EDA & Schema Discovery (CASMI26 / Enveda)

**Purpose**: Understand train/test parquet files, auto-detect schema, compute
descriptive statistics, perform overlap analysis, generate plots, and save
reusable artifacts.

| Item | Value |
|------|-------|
| Inputs | `train.parquet`, `test.parquet` from DATA_DIR |
| Outputs | `schema.json`, `train_clean.parquet`, `test_clean.parquet`, `stats.json`, 6 figures |
| Expected Runtime | ~5–10 min (CPU only) |
| Expected GPU Memory | None (CPU notebook) |
""")

# ── Cell 2: Config ─────────────────────────────────────────────────────────
code("""
# ── CONFIG ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field, asdict
import time

@dataclass
class CFG:
    seed: int = 42
    SMOKE_TEST: bool = False
    # Paths
    data_dir: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"
    output_dir: str = "/kaggle/working/artifacts/eda"
    # Limits
    smoke_n_rows: int = 500
    time_budget_s: int = 3600   # 1 h for EDA
    top_k_peaks: int = 200      # max peaks to keep per spectrum
    ppm_tolerance: float = 10.0
    mass_match_ppm: float = 5.0
    # Display
    truncate_array_len: int = 8

cfg = CFG()
T0 = time.time()
print(f"Config: SMOKE_TEST={cfg.SMOKE_TEST}, seed={cfg.seed}")
""")

# ── Cell 3: Imports & environment ──────────────────────────────────────────
code("""
# ── IMPORTS & ENVIRONMENT ──────────────────────────────────────────────────
import os, sys, json, gc, warnings, random, hashlib, re
from pathlib import Path
from glob import glob
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as sp_stats
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# RDKit – mandatory
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, AllChem, MolToSmiles, MolFromSmiles
    from rdkit.Chem import rdMolDescriptors, Scaffolds
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    RDLogger.DisableLog("rdApp.*")
    RDKIT_VERSION = Chem.rdBase.rdkitVersion
except ImportError as e:
    raise ImportError(f"RDKit is mandatory for this competition: {e}")

# Torch – optional for this notebook
try:
    import torch
    TORCH_VERSION = torch.__version__
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}, "
                  f"{torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")
    else:
        print("No GPU detected (CPU-only run)")
except ImportError:
    TORCH_VERSION = "N/A"
    print("torch not available – CPU notebook")

print(f"Python {sys.version}")
print(f"numpy {np.__version__}, pandas {pd.__version__}, RDKit {RDKIT_VERSION}")
if TORCH_VERSION != "N/A":
    print(f"torch {TORCH_VERSION}")
""")

# ── Cell 4: Seeding & Utilities ────────────────────────────────────────────
code("""
# ── UTILITIES ──────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

seed_everything(cfg.seed)

def elapsed():
    return time.time() - T0

def check_budget():
    if elapsed() > cfg.time_budget_s:
        raise RuntimeError(f"Time budget exceeded: {elapsed():.0f}s > {cfg.time_budget_s}s")

# InChIKey14 with tautomer canonicalization (competition match rule)
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_ik14_cache = {}

def ik14(smiles):
    \"\"\"Compute InChIKey14 from SMILES using tautomer canonicalization.\"\"\"
    if smiles in _ik14_cache:
        return _ik14_cache[smiles]
    try:
        mol = MolFromSmiles(smiles)
        if mol is None:
            _ik14_cache[smiles] = None
            return None
        mol = _tautomer_enumerator.Canonicalize(mol)
        inchi = MolToInchi(mol)
        if inchi is None:
            _ik14_cache[smiles] = None
            return None
        ik = InchiToInchiKey(inchi)
        result = ik[:14] if ik else None
    except Exception:
        result = None
    _ik14_cache[smiles] = result
    return result

def truncate_list(lst, max_len=8):
    \"\"\"Truncate list for display.\"\"\"
    if lst is None:
        return None
    if hasattr(lst, '__len__') and len(lst) > max_len:
        return list(lst[:max_len]) + [f"... ({len(lst)} total)"]
    return list(lst) if hasattr(lst, '__len__') else lst

print(f"Utilities loaded. [{elapsed():.1f}s]")
""")

# ── Cell 5: Data Loading ──────────────────────────────────────────────────
code("""
# ── DATA LOADING ───────────────────────────────────────────────────────────
def find_data_dir():
    \"\"\"Find the data directory with fallback glob.\"\"\"
    if os.path.isdir(cfg.data_dir):
        return cfg.data_dir
    # Fallback: glob for train.parquet
    candidates = glob("/kaggle/input/**/train.parquet", recursive=True)
    if candidates:
        return str(Path(candidates[0]).parent)
    # Local dev fallback
    local = glob("**/train.parquet", recursive=True)
    if local:
        return str(Path(local[0]).parent)
    raise FileNotFoundError(
        f"Cannot find data. Expected at {cfg.data_dir} or via glob."
    )

DATA_DIR = find_data_dir()
print(f"DATA_DIR = {DATA_DIR}")

train_raw = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
test_raw  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

if cfg.SMOKE_TEST:
    train_raw = train_raw.head(cfg.smoke_n_rows).copy()
    test_raw  = test_raw.head(min(cfg.smoke_n_rows, len(test_raw))).copy()
    print(f"SMOKE_TEST: using {len(train_raw)} train, {len(test_raw)} test rows")

print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")
print(f"Train memory: {train_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Test memory:  {test_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 6: Raw data inspection ───────────────────────────────────────────
code("""
# ── RAW DATA INSPECTION ───────────────────────────────────────────────────
print("=== TRAIN DTYPES ===")
print(train_raw.dtypes)
print()

print("=== TRAIN NULL COUNTS ===")
null_counts = train_raw.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No nulls")
print()

print("=== TEST DTYPES ===")
print(test_raw.dtypes)
print()

print("=== TEST NULL COUNTS ===")
null_counts_test = test_raw.isnull().sum()
print(null_counts_test[null_counts_test > 0] if null_counts_test.sum() > 0 else "No nulls")
print()

# Show 3 sample rows (truncate long arrays)
print("=== TRAIN SAMPLE (3 rows, arrays truncated) ===")
sample = train_raw.head(3).copy()
for col in sample.columns:
    first_val = sample[col].iloc[0]
    if hasattr(first_val, '__len__') and not isinstance(first_val, str) and len(first_val) > cfg.truncate_array_len:
        sample[col] = sample[col].apply(lambda x: truncate_list(x, cfg.truncate_array_len))
pd.set_option("display.max_colwidth", 80)
print(sample.to_string())
print()

print("=== TEST SAMPLE (3 rows, arrays truncated) ===")
sample_t = test_raw.head(3).copy()
for col in sample_t.columns:
    first_val = sample_t[col].iloc[0]
    if hasattr(first_val, '__len__') and not isinstance(first_val, str) and len(first_val) > cfg.truncate_array_len:
        sample_t[col] = sample_t[col].apply(lambda x: truncate_list(x, cfg.truncate_array_len))
print(sample_t.to_string())
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 7: SchemaAdapter ─────────────────────────────────────────────────
code("""
# ── SCHEMA ADAPTER ─────────────────────────────────────────────────────────
class SchemaAdapter:
    \"\"\"Auto-detect column mapping from raw parquet files.

    Handles: standard column names, nested arrays, JSON strings, list-of-structs.
    \"\"\"

    # Candidate column names for each semantic field
    CANDIDATES = {
        "molecule_id":  ["molecule_id", "mol_id", "compound_id", "id"],
        "spectrum_id":  ["spectrum_id", "spec_id", "scan_id"],
        "smiles":       ["normalized_smiles", "smiles", "SMILES", "canonical_smiles",
                         "structure", "smi"],
        "precursor_mz": ["precursor_mz", "precursor_mass", "prec_mz", "mz",
                         "parent_mz", "precursormz"],
        "adduct":       ["adduct", "adduct_type", "precursor_type", "ion_type"],
        "adduct_orig":  ["adduct_orig", "original_adduct"],
        "polarity":     ["ionization_mode", "polarity", "ion_mode", "mode",
                         "ionmode"],
        "collision_energy": ["collision_energy_ev", "collision_energy",
                             "ce", "CE", "energy"],
        "collision_energy_orig": ["collision_energy_orig"],
        "collision_energy_orig_units": ["collision_energy_orig_units"],
        "instrument":   ["instrument_type", "instrument", "inst_type",
                         "instrument_name"],
        "formula":      ["molecular_formula", "formula", "mol_formula",
                         "molecularformula"],
        "inchikey":     ["inchikey", "InChIKey", "inchi_key"],
        "inchikey14":   ["inchikey14", "InChIKey14", "ik14"],
        "mz_array":     ["ms2_mzs", "mz_array", "peaks_mz", "mzs",
                         "fragment_mz", "mz_values"],
        "intensity_array": ["ms2_normalized_intensities", "intensity_array",
                            "peaks_intensity", "intensities",
                            "normalized_intensities", "intensity_values"],
        "ingest_lib":   ["ingest_lib", "library", "source_lib", "source"],
        "base_peak_intensity": ["base_peak_intensity", "base_peak_int"],
    }

    def __init__(self, df, name="unknown"):
        self.name = name
        self.mapping = {}
        self.col_types = {}
        cols_lower = {c.lower(): c for c in df.columns}

        for field, candidates in self.CANDIDATES.items():
            found = None
            for cand in candidates:
                # Exact match
                if cand in df.columns:
                    found = cand
                    break
                # Case-insensitive
                if cand.lower() in cols_lower:
                    found = cols_lower[cand.lower()]
                    break
            if found is not None:
                self.mapping[field] = found
                # Detect type (nested, JSON, etc.)
                self.col_types[field] = self._detect_type(df, found)

        # Report
        print(f"\\nSchemaAdapter({name}): detected {len(self.mapping)}/{len(self.CANDIDATES)} fields")
        for field, col in self.mapping.items():
            print(f"  {field:30s} -> {col:35s} [{self.col_types.get(field, '?')}]")

        missing_critical = []
        for req in ["molecule_id", "precursor_mz", "mz_array", "intensity_array"]:
            if req not in self.mapping:
                missing_critical.append(req)
        if missing_critical:
            raise ValueError(
                f"SchemaAdapter({name}): CRITICAL columns not found: {missing_critical}\\n"
                f"Available columns: {list(df.columns)}"
            )

    def _detect_type(self, df, col):
        \"\"\"Detect if column is scalar, list, nested struct, or JSON string.\"\"\"
        sample = df[col].dropna().iloc[:5] if len(df) > 0 else pd.Series()
        if len(sample) == 0:
            return "empty"
        val = sample.iloc[0]
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return "json_list"
                return "json_other"
            except (json.JSONDecodeError, TypeError):
                return "string"
        if isinstance(val, (list, np.ndarray)):
            if len(val) > 0 and isinstance(val[0], dict):
                return "list_of_structs"
            return "array"
        if isinstance(val, (int, float, np.integer, np.floating)):
            return "numeric"
        return f"other({type(val).__name__})"

    def get(self, field, df):
        \"\"\"Get a column's values, handling JSON/nested formats.\"\"\"
        if field not in self.mapping:
            return None
        col = self.mapping[field]
        ctype = self.col_types.get(field, "unknown")
        series = df[col]

        if ctype == "json_list":
            return series.apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        if ctype == "list_of_structs":
            # Extract the first key's values as separate arrays
            return series  # caller handles
        return series

    def col(self, field):
        \"\"\"Get column name for a field, or None.\"\"\"
        return self.mapping.get(field, None)

    def has(self, field):
        return field in self.mapping

    def to_dict(self):
        return {
            "mapping": self.mapping,
            "col_types": self.col_types,
        }

# Build adapters
adapter_train = SchemaAdapter(train_raw, "train")
adapter_test  = SchemaAdapter(test_raw, "test")
print(f"\\n[{elapsed():.1f}s]")
""")

# ── Cell 8: Save schema.json ──────────────────────────────────────────────
code("""
# ── SAVE SCHEMA ────────────────────────────────────────────────────────────
os.makedirs(cfg.output_dir, exist_ok=True)

schema = {
    "train": adapter_train.to_dict(),
    "test":  adapter_test.to_dict(),
    "train_columns": list(train_raw.columns),
    "test_columns":  list(test_raw.columns),
    "train_shape": list(train_raw.shape),
    "test_shape":  list(test_raw.shape),
    "train_only_columns": sorted(set(train_raw.columns) - set(test_raw.columns)),
    "test_only_columns":  sorted(set(test_raw.columns) - set(train_raw.columns)),
    "timestamp": datetime.now().isoformat(),
}

schema_path = os.path.join(cfg.output_dir, "schema.json")
with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)
print(f"Schema saved to {schema_path}")
print(f"Train-only columns: {schema['train_only_columns']}")
print(f"Test-only columns:  {schema['test_only_columns']}")
""")

# ── Cell 9: Adduct table ──────────────────────────────────────────────────
md("## Adduct Parsing & Neutral Mass")
code("""
# ── ADDUCT TABLE ───────────────────────────────────────────────────────────
PROTON_MASS = 1.00727646677

ADDUCT_TABLE = {
    "[M+H]+":       ("positive", 1,  PROTON_MASS),
    "[M+Na]+":      ("positive", 1,  22.989218),
    "[M+K]+":       ("positive", 1,  38.963158),
    "[M+NH4]+":     ("positive", 1,  18.033823),
    "[M+H-H2O]+":   ("positive", 1, PROTON_MASS - 18.010565),
    "[M+2H]2+":     ("positive", 2,  PROTON_MASS),       # doubly charged
    "[M-H]-":       ("negative", 1, -PROTON_MASS),
    "[M+FA-H]-":    ("negative", 1, 44.998201 - PROTON_MASS),
    "[M+Cl]-":      ("negative", 1, 34.969402),
    "[M-2H]2-":     ("negative", 2, -PROTON_MASS),       # doubly charged
    "[M+HAc-H]-":   ("negative", 1, 60.021129 - PROTON_MASS),  # acetate
    "[M-H2O+H]+":   ("positive", 1, PROTON_MASS - 18.010565),
    "[M-H2O-H]-":   ("negative", 1, -PROTON_MASS - 18.010565),
    "[M+2Na-H]+":   ("positive", 1, 2 * 22.989218 - PROTON_MASS),
}

_unknown_adducts = Counter()

def neutral_mass_from_precursor(precursor_mz, adduct, polarity=None):
    \"\"\"Compute neutral monoisotopic mass from precursor m/z and adduct.\"\"\"
    adduct_str = str(adduct).strip()
    if adduct_str in ADDUCT_TABLE:
        _, charge, shift = ADDUCT_TABLE[adduct_str]
        return precursor_mz * charge - shift
    # Unknown adduct – fall back
    _unknown_adducts[adduct_str] += 1
    pol = str(polarity).lower() if polarity else "positive"
    if "neg" in pol:
        return precursor_mz + PROTON_MASS   # assume [M-H]-
    return precursor_mz - PROTON_MASS        # assume [M+H]+

# Show known adducts
print("Known adducts:")
for a, (pol, charge, shift) in sorted(ADDUCT_TABLE.items()):
    print(f"  {a:20s}  pol={pol:8s}  charge={charge}  shift={shift:+.6f}")
""")

# ── Cell 10: Spectra statistics (train) ────────────────────────────────────
md("## Spectral Statistics")
code("""
# ── SPECTRAL STATISTICS ───────────────────────────────────────────────────
check_budget()

def compute_spectra_stats(df, adapter, name="train"):
    \"\"\"Compute spectrum-level statistics.\"\"\"
    stats = {"name": name, "n_rows": len(df)}

    # Unique molecules
    mol_col = adapter.col("molecule_id")
    stats["n_unique_molecules"] = df[mol_col].nunique()

    # Spectra per molecule
    spm = df.groupby(mol_col).size()
    stats["spectra_per_molecule"] = {
        "mean": float(spm.mean()),
        "median": float(spm.median()),
        "max": int(spm.max()),
        "min": int(spm.min()),
    }

    # Peaks per spectrum
    mz_col = adapter.col("mz_array")
    peaks_per_spec = df[mz_col].apply(lambda x: len(x) if x is not None and hasattr(x, '__len__') else 0)
    stats["peaks_per_spectrum"] = {
        "mean": float(peaks_per_spec.mean()),
        "median": float(peaks_per_spec.median()),
        "max": int(peaks_per_spec.max()),
        "min": int(peaks_per_spec.min()),
        "q95": float(peaks_per_spec.quantile(0.95)),
        "q99": float(peaks_per_spec.quantile(0.99)),
    }
    stats["_peaks_per_spec_array"] = peaks_per_spec.values  # for hist

    # Precursor m/z range
    pmz_col = adapter.col("precursor_mz")
    pmz = df[pmz_col].dropna()
    stats["precursor_mz"] = {
        "min": float(pmz.min()),
        "max": float(pmz.max()),
        "mean": float(pmz.mean()),
        "median": float(pmz.median()),
    }
    stats["_precursor_mz_array"] = pmz.values

    # Adduct counts
    adduct_col = adapter.col("adduct")
    if adduct_col:
        stats["adduct_counts"] = df[adduct_col].value_counts().head(20).to_dict()

    # Polarity split
    pol_col = adapter.col("polarity")
    if pol_col:
        stats["polarity_counts"] = df[pol_col].value_counts().to_dict()

    # Collision energy
    ce_col = adapter.col("collision_energy")
    if ce_col:
        ce_vals = df[ce_col]
        # CE may be a list column (e.g., [20] or [20,40,60])
        sample_ce = ce_vals.dropna().iloc[0] if len(ce_vals.dropna()) > 0 else None
        if isinstance(sample_ce, (list, np.ndarray)):
            # Flatten and take first value for stats
            ce_flat = ce_vals.dropna().apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
            ).dropna()
            stats["collision_energy"] = {
                "type": "list",
                "first_value_stats": {
                    "mean": float(ce_flat.mean()),
                    "median": float(ce_flat.median()),
                    "unique_count": int(ce_flat.nunique()),
                },
                "list_len_mean": float(ce_vals.dropna().apply(len).mean()),
            }
            stats["_ce_flat"] = ce_flat.values
        else:
            stats["collision_energy"] = {
                "type": "scalar",
                "unique_count": int(ce_vals.nunique()),
            }

    # Instrument counts
    inst_col = adapter.col("instrument")
    if inst_col:
        stats["instrument_counts"] = df[inst_col].value_counts().head(20).to_dict()

    # Fraction of spectra with precursor peak present
    if mz_col and pmz_col:
        def has_precursor_peak(row):
            mzs = row[mz_col]
            pmz_val = row[pmz_col]
            if mzs is None or not hasattr(mzs, '__len__') or len(mzs) == 0:
                return False
            if pd.isna(pmz_val):
                return False
            mzs_arr = np.array(mzs, dtype=np.float64)
            return bool(np.any(np.abs(mzs_arr - pmz_val) < 0.5))

        # Sample for speed
        sample_n = min(10000, len(df))
        sample_df = df.sample(n=sample_n, random_state=cfg.seed)
        frac = sample_df.apply(has_precursor_peak, axis=1).mean()
        stats["frac_precursor_peak_present"] = float(frac)

    # Intensity distribution (sample)
    int_col = adapter.col("intensity_array")
    if int_col:
        sample_n = min(5000, len(df))
        sample_ints = df[int_col].sample(n=sample_n, random_state=cfg.seed)
        all_ints = []
        for arr in sample_ints:
            if arr is not None and hasattr(arr, '__len__'):
                all_ints.extend(list(arr)[:200])  # cap per spectrum
        all_ints = np.array(all_ints, dtype=np.float64)
        stats["intensity_distribution"] = {
            "mean": float(np.mean(all_ints)),
            "median": float(np.median(all_ints)),
            "max": float(np.max(all_ints)),
            "min": float(np.min(all_ints)),
            "frac_above_0.1": float(np.mean(all_ints > 0.1)),
            "frac_above_0.5": float(np.mean(all_ints > 0.5)),
        }
        stats["_intensity_sample"] = all_ints

    print(f"Stats for {name}: {stats['n_rows']} rows, "
          f"{stats['n_unique_molecules']} molecules, "
          f"peaks/spec mean={stats['peaks_per_spectrum']['mean']:.1f}")
    return stats

train_stats = compute_spectra_stats(train_raw, adapter_train, "train")
test_stats  = compute_spectra_stats(test_raw, adapter_test, "test")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 11: KS Tests ─────────────────────────────────────────────────────
md("## Train vs Test Distribution Comparison (KS Test)")
code("""
# ── KS TESTS ──────────────────────────────────────────────────────────────
ks_results = {}

# Precursor m/z
ks_pmz = sp_stats.ks_2samp(
    train_stats["_precursor_mz_array"],
    test_stats["_precursor_mz_array"]
)
ks_results["precursor_mz"] = {"statistic": float(ks_pmz.statistic), "pvalue": float(ks_pmz.pvalue)}
print(f"KS test (precursor m/z): stat={ks_pmz.statistic:.4f}, p={ks_pmz.pvalue:.2e}")

# Number of peaks
ks_peaks = sp_stats.ks_2samp(
    train_stats["_peaks_per_spec_array"],
    test_stats["_peaks_per_spec_array"]
)
ks_results["n_peaks"] = {"statistic": float(ks_peaks.statistic), "pvalue": float(ks_peaks.pvalue)}
print(f"KS test (#peaks/spectrum): stat={ks_peaks.statistic:.4f}, p={ks_peaks.pvalue:.2e}")

train_stats["ks_tests"] = ks_results
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 12: Molecule statistics (train) ──────────────────────────────────
md("## Molecule Statistics (Train Only)")
code("""
# ── MOLECULE STATISTICS ────────────────────────────────────────────────────
check_budget()

mol_col = adapter_train.col("molecule_id")
smi_col = adapter_train.col("smiles")

# Deduplicate to one row per molecule
mol_df = train_raw.drop_duplicates(subset=[mol_col]).copy()
print(f"Unique molecules: {len(mol_df)}")

# RDKit parse rate
parse_ok = 0
parse_fail = 0
heavy_atoms = []
mol_weights = []
element_sets = []
formulas = []
salts = 0
charged = 0
multi_frag = 0
scaffolds = []
valid_smiles_list = []
valid_ik14_list = []

for smi in tqdm(mol_df[smi_col], desc="Parsing SMILES"):
    mol = MolFromSmiles(str(smi))
    if mol is None:
        parse_fail += 1
        continue
    parse_ok += 1

    # Heavy atom count
    heavy_atoms.append(mol.GetNumHeavyAtoms())

    # Molecular weight
    mw = Descriptors.ExactMolWt(mol)
    mol_weights.append(mw)

    # Elements
    elems = set()
    for atom in mol.GetAtoms():
        elems.add(atom.GetSymbol())
    element_sets.append(frozenset(elems))

    # Formula
    formulas.append(rdMolDescriptors.CalcMolFormula(mol))

    # Salt / multi-fragment detection (contains ".")
    smi_str = str(smi)
    if "." in smi_str:
        multi_frag += 1
        salts += 1

    # Charged
    net_charge = Chem.GetFormalCharge(mol)
    if net_charge != 0:
        charged += 1

    # Scaffold
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(MolToSmiles(core))
    except Exception:
        scaffolds.append("FAIL")

    # InChIKey14
    valid_smiles_list.append(smi_str)
    valid_ik14_list.append(ik14(smi_str))

n_total = parse_ok + parse_fail
mol_stats = {
    "n_molecules": n_total,
    "rdkit_parse_rate": parse_ok / n_total if n_total > 0 else 0,
    "rdkit_failures": parse_fail,
    "heavy_atom_count": {
        "mean": float(np.mean(heavy_atoms)),
        "median": float(np.median(heavy_atoms)),
        "max": int(np.max(heavy_atoms)),
        "min": int(np.min(heavy_atoms)),
    },
    "molecular_weight": {
        "mean": float(np.mean(mol_weights)),
        "median": float(np.median(mol_weights)),
        "max": float(np.max(mol_weights)),
        "min": float(np.min(mol_weights)),
    },
    "frac_salts": salts / n_total if n_total > 0 else 0,
    "frac_charged": charged / n_total if n_total > 0 else 0,
    "frac_multi_fragment": multi_frag / n_total if n_total > 0 else 0,
    "n_unique_ik14": len(set(x for x in valid_ik14_list if x is not None)),
    "n_unique_scaffolds": len(set(scaffolds)),
}

# Element frequency
all_elements = Counter()
for es in element_sets:
    all_elements.update(es)
mol_stats["element_frequency"] = dict(all_elements.most_common(20))

# Formula frequency (top 20)
formula_counts = Counter(formulas)
mol_stats["top_formulas"] = dict(formula_counts.most_common(20))

# Scaffold frequency (top 20)
scaffold_counts = Counter(scaffolds)
mol_stats["n_unique_scaffolds"] = len(scaffold_counts)
mol_stats["top_scaffolds"] = dict(scaffold_counts.most_common(20))

print(f"\\nRDKit parse rate: {mol_stats['rdkit_parse_rate']:.4f} "
      f"({parse_ok}/{n_total}, {parse_fail} failures)")
print(f"Heavy atoms: mean={mol_stats['heavy_atom_count']['mean']:.1f}, "
      f"median={mol_stats['heavy_atom_count']['median']:.0f}")
print(f"MW: mean={mol_stats['molecular_weight']['mean']:.1f}, "
      f"median={mol_stats['molecular_weight']['median']:.1f}")
print(f"Salts: {mol_stats['frac_salts']:.4f}, "
      f"Charged: {mol_stats['frac_charged']:.4f}, "
      f"Multi-fragment: {mol_stats['frac_multi_fragment']:.4f}")
print(f"Unique InChIKey14: {mol_stats['n_unique_ik14']}")
print(f"Unique scaffolds: {mol_stats['n_unique_scaffolds']}")
print(f"Elements found: {dict(all_elements.most_common(15))}")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 13: Mass sanity check ─────────────────────────────────────────────
md("## Adduct Neutral Mass vs SMILES Mass Sanity Check")
code("""
# ── MASS SANITY CHECK ──────────────────────────────────────────────────────
check_budget()

# For each molecule, compute neutral mass from precursor+adduct, compare to SMILES MW
pmz_col = adapter_train.col("precursor_mz")
adduct_col = adapter_train.col("adduct")
pol_col = adapter_train.col("polarity")

# Take one spectrum per molecule (first occurrence)
mol_first = train_raw.drop_duplicates(subset=[mol_col]).copy()

mass_matches = 0
mass_total = 0
ppm_errors = []

for _, row in tqdm(mol_first.iterrows(), total=len(mol_first), desc="Mass sanity check"):
    smi = str(row[smi_col])
    mol = MolFromSmiles(smi)
    if mol is None:
        continue

    smiles_mw = Descriptors.ExactMolWt(mol)
    pmz_val = row[pmz_col]
    adduct_val = row[adduct_col] if adduct_col else "[M+H]+"
    pol_val = row[pol_col] if pol_col else None

    neutral = neutral_mass_from_precursor(pmz_val, adduct_val, pol_val)
    mass_total += 1

    ppm_err = abs(neutral - smiles_mw) / smiles_mw * 1e6 if smiles_mw > 0 else 1e6
    ppm_errors.append(ppm_err)

    if ppm_err <= cfg.ppm_tolerance:
        mass_matches += 1

ppm_errors = np.array(ppm_errors)
mol_stats["mass_sanity"] = {
    "n_checked": mass_total,
    "n_match_within_10ppm": mass_matches,
    "frac_match": mass_matches / mass_total if mass_total > 0 else 0,
    "ppm_error_median": float(np.median(ppm_errors)),
    "ppm_error_mean": float(np.mean(ppm_errors)),
    "ppm_error_q95": float(np.percentile(ppm_errors, 95)),
}

if _unknown_adducts:
    mol_stats["unknown_adducts"] = dict(_unknown_adducts.most_common(20))
    print(f"Unknown adducts encountered: {dict(_unknown_adducts.most_common(10))}")

print(f"Mass sanity: {mass_matches}/{mass_total} match within {cfg.ppm_tolerance} ppm "
      f"({mol_stats['mass_sanity']['frac_match']:.4f})")
print(f"PPM error: median={np.median(ppm_errors):.2f}, "
      f"mean={np.mean(ppm_errors):.2f}, q95={np.percentile(ppm_errors, 95):.2f}")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 14: Overlap analysis ─────────────────────────────────────────────
md("## Overlap Analysis (Mass-Based)")
code("""
# ── OVERLAP ANALYSIS ───────────────────────────────────────────────────────
check_budget()

# Compute neutral masses for all train molecules
print("Computing neutral masses for train molecules...")
train_neutral_masses = []
train_mol_first = train_raw.drop_duplicates(subset=[mol_col])
for _, row in tqdm(train_mol_first.iterrows(), total=len(train_mol_first),
                   desc="Train neutral masses"):
    pmz_val = row[pmz_col]
    adduct_val = row[adduct_col] if adduct_col else "[M+H]+"
    pol_val = row[pol_col] if pol_col else None
    nm = neutral_mass_from_precursor(pmz_val, adduct_val, pol_val)
    train_neutral_masses.append(nm)

train_neutral_masses = np.sort(np.array(train_neutral_masses, dtype=np.float64))
print(f"Train neutral masses: {len(train_neutral_masses)} molecules, "
      f"range [{train_neutral_masses.min():.2f}, {train_neutral_masses.max():.2f}]")

# Compute neutral masses for all test molecules
print("Computing neutral masses for test molecules...")
test_mol_col = adapter_test.col("molecule_id")
test_pmz_col = adapter_test.col("precursor_mz")
test_adduct_col = adapter_test.col("adduct")
test_pol_col = adapter_test.col("polarity")

test_mol_first = test_raw.drop_duplicates(subset=[test_mol_col])
test_neutral_masses = []
for _, row in tqdm(test_mol_first.iterrows(), total=len(test_mol_first),
                   desc="Test neutral masses"):
    pmz_val = row[test_pmz_col]
    adduct_val = row[test_adduct_col] if test_adduct_col else "[M+H]+"
    pol_val = row[test_pol_col] if test_pol_col else None
    nm = neutral_mass_from_precursor(pmz_val, adduct_val, pol_val)
    test_neutral_masses.append(nm)

test_neutral_masses = np.array(test_neutral_masses, dtype=np.float64)

# For each test molecule, check if ANY train mass is within 5 ppm
def count_mass_overlaps(test_masses, train_masses_sorted, ppm=5.0):
    \"\"\"Binary search approach for ppm matching.\"\"\"
    n_match = 0
    for tm in tqdm(test_masses, desc="Mass overlap check"):
        tol = tm * ppm / 1e6
        lo = np.searchsorted(train_masses_sorted, tm - tol)
        hi = np.searchsorted(train_masses_sorted, tm + tol)
        if hi > lo:
            n_match += 1
    return n_match

n_overlap = count_mass_overlaps(test_neutral_masses, train_neutral_masses,
                                 cfg.mass_match_ppm)
overlap_frac = n_overlap / len(test_neutral_masses) if len(test_neutral_masses) > 0 else 0

overlap_stats = {
    "n_test_molecules": len(test_neutral_masses),
    "n_test_with_train_mass_match": n_overlap,
    "frac_test_mass_covered": overlap_frac,
    "ppm_tolerance": cfg.mass_match_ppm,
    "note": "This is an UPPER BOUND on library lookup coverage. "
            "Same mass != same structure."
}

print(f"\\nMass overlap: {n_overlap}/{len(test_neutral_masses)} test molecules "
      f"({overlap_frac:.4f}) have a train molecule within {cfg.mass_match_ppm} ppm")
print("NOTE: This is an upper bound – same neutral mass does NOT mean same structure.")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 15: Plots ─────────────────────────────────────────────────────────
md("## Visualizations")
code("""
# ── PLOTS ──────────────────────────────────────────────────────────────────
check_budget()
fig_dir = os.path.join(cfg.output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"train": "#2196F3", "test": "#FF5722"}

# ─── Figure 1: Precursor m/z distribution ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(train_stats["_precursor_mz_array"], bins=200, alpha=0.6,
        color=COLORS["train"], label="Train", density=True)
ax.hist(test_stats["_precursor_mz_array"], bins=200, alpha=0.6,
        color=COLORS["test"], label="Test", density=True)
ax.set_xlabel("Precursor m/z (Da)")
ax.set_ylabel("Density")
ax.set_title("Precursor m/z Distribution")
ax.legend()
ax.set_xlim(0, min(2000, train_stats["precursor_mz"]["max"] * 1.1))
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "01_precursor_mz.png"), dpi=150)
plt.show()
plt.close(fig)

# ─── Figure 2: Peaks per spectrum ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(train_stats["_peaks_per_spec_array"], bins=100, alpha=0.6,
        color=COLORS["train"], label="Train", density=True)
ax.hist(test_stats["_peaks_per_spec_array"], bins=100, alpha=0.6,
        color=COLORS["test"], label="Test", density=True)
ax.set_xlabel("Number of Peaks per Spectrum")
ax.set_ylabel("Density")
ax.set_title("Peaks per Spectrum Distribution")
ax.legend()
ax.set_xlim(0, min(500, np.percentile(train_stats["_peaks_per_spec_array"], 99) * 1.2))
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "02_peaks_per_spectrum.png"), dpi=150)
plt.show()
plt.close(fig)

# ─── Figure 3: Adduct and polarity counts ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Adduct counts (train)
if "adduct_counts" in train_stats:
    adducts = train_stats["adduct_counts"]
    top_adducts = dict(sorted(adducts.items(), key=lambda x: -x[1])[:10])
    axes[0].barh(list(top_adducts.keys()), list(top_adducts.values()),
                 color=COLORS["train"], alpha=0.8)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Top 10 Adducts (Train)")
    axes[0].invert_yaxis()

# Polarity (train)
if "polarity_counts" in train_stats:
    pol = train_stats["polarity_counts"]
    axes[1].bar(list(pol.keys()), list(pol.values()),
                color=[COLORS["train"], COLORS["test"]][:len(pol)], alpha=0.8)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Polarity Split (Train)")

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "03_adduct_polarity.png"), dpi=150)
plt.show()
plt.close(fig)

# ─── Figure 4: Molecular weight & heavy atom distributions ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(mol_weights, bins=100, color=COLORS["train"], alpha=0.7)
axes[0].set_xlabel("Exact Molecular Weight (Da)")
axes[0].set_ylabel("Count")
axes[0].set_title("Molecular Weight Distribution (Train)")

axes[1].hist(heavy_atoms, bins=80, color=COLORS["train"], alpha=0.7)
axes[1].set_xlabel("Heavy Atom Count")
axes[1].set_ylabel("Count")
axes[1].set_title("Heavy Atom Count Distribution (Train)")

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "04_mol_properties.png"), dpi=150)
plt.show()
plt.close(fig)

# ─── Figure 5: PPM error histogram ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ppm_clipped = ppm_errors[ppm_errors < 100]
ax.hist(ppm_clipped, bins=200, color="#4CAF50", alpha=0.7)
ax.axvline(x=10, color="red", linestyle="--", label="10 ppm threshold")
ax.set_xlabel("PPM Error (adduct-derived mass vs SMILES mass)")
ax.set_ylabel("Count")
ax.set_title("Mass Sanity Check: PPM Error Distribution")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "05_ppm_error.png"), dpi=150)
plt.show()
plt.close(fig)

print(f"Figures saved to {fig_dir}")
print(f"[{elapsed():.1f}s]")
""")

# ── Cell 16: Example spectra ──────────────────────────────────────────────
md("## Example Spectra")
code("""
# ── EXAMPLE SPECTRA ────────────────────────────────────────────────────────
check_budget()

mz_col = adapter_train.col("mz_array")
int_col = adapter_train.col("intensity_array")
pmz_col_name = adapter_train.col("precursor_mz")
adduct_col_name = adapter_train.col("adduct")
smi_col_name = adapter_train.col("smiles")

# Pick 3 spectra from different molecules with decent peak counts
sample_specs = []
seen_mols = set()
for idx, row in train_raw.iterrows():
    mid = row[mol_col]
    if mid in seen_mols:
        continue
    mzs = row[mz_col]
    if mzs is not None and hasattr(mzs, '__len__') and 10 < len(mzs) < 200:
        sample_specs.append(row)
        seen_mols.add(mid)
    if len(sample_specs) >= 3:
        break

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

for i, row in enumerate(sample_specs):
    ax = axes[i]
    mzs = np.array(row[mz_col], dtype=np.float64)
    ints = np.array(row[int_col], dtype=np.float64)
    pmz_val = row[pmz_col_name]
    adduct_val = row[adduct_col_name] if adduct_col_name else "?"

    # Stem plot
    markerline, stemlines, baseline = ax.stem(mzs, ints, linefmt="-",
                                               markerfmt=" ", basefmt="-")
    stemlines.set_linewidth(0.8)
    stemlines.set_color(COLORS["train"])

    # Precursor line
    ax.axvline(x=pmz_val, color="red", linestyle="--", alpha=0.7,
               label=f"Precursor: {pmz_val:.2f}")

    smi_str = str(row[smi_col_name])[:60] if smi_col_name else "?"
    ax.set_title(f"Spectrum {i+1}: {smi_str}  |  Adduct: {adduct_val}  |  "
                 f"{len(mzs)} peaks")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(fontsize=8)

axes[-1].set_xlabel("m/z (Da)")
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "06_example_spectra.png"), dpi=150)
plt.show()
plt.close(fig)

print(f"[{elapsed():.1f}s]")
""")

# ── Cell 17: Build clean parquets ──────────────────────────────────────────
md("## Build Normalized Clean Parquet Files")
code("""
# ── BUILD CLEAN PARQUETS ──────────────────────────────────────────────────
check_budget()
print("Building train_clean.parquet ...")

def build_clean_df(df, adapter, is_train=True):
    \"\"\"Pure vectorized and zero-copy columnar parquet construction.\"\"\"
    n_rows = len(df)
    out = pd.DataFrame(index=df.index)

    # 1. molecule_id
    mol_col = adapter.col("molecule_id")
    if mol_col and mol_col in df.columns:
        out["molecule_id"] = df[mol_col].astype(str)
    elif "inchikey14" in df.columns:
        out["molecule_id"] = df["inchikey14"].astype(str)
    elif is_train and adapter.col("smiles") and adapter.col("smiles") in df.columns:
        out["molecule_id"] = df[adapter.col("smiles")].astype(str)
    else:
        out["molecule_id"] = [f"{'train' if is_train else 'test'}_mol_{i}" for i in range(n_rows)]

    # 2. precursor_mz
    pmz_c = adapter.col("precursor_mz")
    out["precursor_mz"] = pd.to_numeric(df[pmz_c], errors="coerce").fillna(0.0).astype(np.float64)

    # 3. adduct
    add_c = adapter.col("adduct")
    if add_c and add_c in df.columns:
        out["adduct"] = df[add_c].fillna("unknown").astype(str)
    else:
        out["adduct"] = "unknown"

    # 4. polarity
    pol_c = adapter.col("polarity")
    if pol_c and pol_c in df.columns:
        out["polarity"] = df[pol_c].fillna("unknown").astype(str)
    else:
        out["polarity"] = "unknown"

    # 5. collision energy
    ce_c = adapter.col("collision_energy")
    if ce_c and ce_c in df.columns:
        sample_ce = df[ce_c].dropna().iloc[0] if len(df[ce_c].dropna()) > 0 else None
        if isinstance(sample_ce, (list, np.ndarray)):
            out["ce"] = pd.to_numeric(df[ce_c].str[0], errors="coerce").astype(np.float32)
        else:
            out["ce"] = pd.to_numeric(df[ce_c], errors="coerce").astype(np.float32)
    else:
        out["ce"] = np.nan

    # 6. instrument
    inst_c = adapter.col("instrument")
    if inst_c and inst_c in df.columns:
        out["instrument"] = df[inst_c].fillna("unknown").astype(str)
    else:
        out["instrument"] = "unknown"

    # 7. mz & inten - Zero-copy direct series assignment
    mz_c = adapter.col("mz_array")
    int_c = adapter.col("intensity_array")
    out["mz"] = df[mz_c]
    out["inten"] = df[int_c]

    # 8. neutral_mass - Vectorized lookup arithmetic
    charge_map = {k: v[0] for k, v in ADDUCT_TABLE.items()}
    shift_map  = {k: v[1] for k, v in ADDUCT_TABLE.items()}
    is_neg = out["polarity"].str.lower().str.contains("neg")
    default_shift = np.where(is_neg, -PROTON_MASS, PROTON_MASS)
    charges = out["adduct"].map(charge_map).fillna(1.0).to_numpy(dtype=np.float64)
    shifts  = out["adduct"].map(shift_map).fillna(pd.Series(default_shift, index=out.index)).to_numpy(dtype=np.float64)
    out["neutral_mass"] = out["precursor_mz"].to_numpy(dtype=np.float64) * charges - shifts

    # 9. Train-only: smiles and ik14 - Direct assignment from existing columns
    if is_train:
        smi_c = adapter.col("smiles")
        out["smiles"] = df[smi_c].astype(str)
        ik14_c = adapter.col("inchikey14")
        if ik14_c and ik14_c in df.columns:
            out["ik14"] = df[ik14_c].astype(str)
        elif "inchikey14" in df.columns:
            out["ik14"] = df["inchikey14"].astype(str)
        elif "molecule_id" in df.columns:
            out["ik14"] = df["molecule_id"].astype(str)
        else:
            out["ik14"] = out["molecule_id"]

    # Reorder columns to exact contract
    if is_train:
        cols = ["molecule_id", "smiles", "ik14", "precursor_mz", "adduct", "polarity", "ce", "instrument", "mz", "inten", "neutral_mass"]
    else:
        cols = ["molecule_id", "precursor_mz", "adduct", "polarity", "ce", "instrument", "mz", "inten", "neutral_mass"]
    return out[cols]

train_clean = build_clean_df(train_raw, adapter_train, is_train=True)
print(f"train_clean: {train_clean.shape}")
print(train_clean.dtypes)
print()

# Save
train_clean_path = os.path.join(cfg.output_dir, "train_clean.parquet")
train_clean.to_parquet(train_clean_path, index=False)
print(f"Saved {train_clean_path} ({os.path.getsize(train_clean_path) / 1e6:.1f} MB)")
print(f"[{elapsed():.1f}s]")

del train_clean
gc.collect()
""")

code("""
# ── TEST CLEAN ─────────────────────────────────────────────────────────────
check_budget()
print("Building test_clean.parquet ...")

test_clean = build_clean_df(test_raw, adapter_test, is_train=False)
print(f"test_clean: {test_clean.shape}")
print(test_clean.dtypes)

test_clean_path = os.path.join(cfg.output_dir, "test_clean.parquet")
test_clean.to_parquet(test_clean_path, index=False)
print(f"Saved {test_clean_path} ({os.path.getsize(test_clean_path) / 1e6:.1f} MB)")
print(f"[{elapsed():.1f}s]")

del test_clean
gc.collect()
""")

# ── Cell 18: Save stats.json ──────────────────────────────────────────────
md("## Save Comprehensive Stats")
code("""
# ── SAVE STATS ─────────────────────────────────────────────────────────────

def make_serializable(obj):
    \"\"\"Convert numpy types to Python natives for JSON serialization.\"\"\"
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

all_stats = {
    "train": make_serializable(train_stats),
    "test":  make_serializable(test_stats),
    "molecule_stats": make_serializable(mol_stats),
    "overlap": make_serializable(overlap_stats),
    "config": asdict(cfg),
    "timestamp": datetime.now().isoformat(),
}

stats_path = os.path.join(cfg.output_dir, "stats.json")
with open(stats_path, "w") as f:
    json.dump(all_stats, f, indent=2, default=str)
print(f"Stats saved to {stats_path}")
""")

# ── Cell 19: Manifest ─────────────────────────────────────────────────────
code("""
# ── MANIFEST ───────────────────────────────────────────────────────────────
import platform

manifest = {
    "stage": "eda",
    "config": asdict(cfg),
    "library_versions": {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": RDKIT_VERSION,
        "scipy": sp_stats.scipy.__version__,
    },
    "files": [
        "schema.json",
        "train_clean.parquet",
        "test_clean.parquet",
        "stats.json",
        "manifest.json",
        "figures/01_precursor_mz.png",
        "figures/02_peaks_per_spectrum.png",
        "figures/03_adduct_polarity.png",
        "figures/04_mol_properties.png",
        "figures/05_ppm_error.png",
        "figures/06_example_spectra.png",
    ],
    "metrics": {
        "train_rows": train_stats["n_rows"],
        "test_rows": test_stats["n_rows"],
        "train_molecules": train_stats["n_unique_molecules"],
        "test_molecules": test_stats["n_unique_molecules"],
        "rdkit_parse_rate": mol_stats["rdkit_parse_rate"],
        "mass_match_frac": mol_stats["mass_sanity"]["frac_match"],
        "mass_overlap_frac": overlap_stats["frac_test_mass_covered"],
    },
    "timestamp": datetime.now().isoformat(),
    "platform": platform.platform(),
}

manifest_path = os.path.join(cfg.output_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"Manifest saved to {manifest_path}")
""")

# ── Cell 20: Recommendations ──────────────────────────────────────────────
md("## Recommendations")
code("""
# ── RECOMMENDATIONS (programmatic) ─────────────────────────────────────────
check_budget()

# Best K peaks cap
peaks_q95 = train_stats["peaks_per_spectrum"]["q95"]
peaks_q99 = train_stats["peaks_per_spectrum"]["q99"]
suggested_k = int(min(300, max(50, peaks_q95 * 1.2)))

# m/z bin range & resolution
mz_max = train_stats["precursor_mz"]["max"]
suggested_mz_max = int(min(2000, mz_max + 50))
# Resolution: ~0.1 Da bins is common for unit-resolution; 0.01 for high-res
suggested_bin_res = 0.1

# Max formula size (based on MW)
mw_q99 = np.percentile(mol_weights, 99) if mol_weights else 1500
suggested_max_formula_mw = int(mw_q99 + 100)

# Split strategy
n_scaffolds = mol_stats["n_unique_scaffolds"]
n_molecules = mol_stats["n_molecules"]

recommendations = f\"\"\"
## EDA Recommendations

### Spectrum Processing
- **Suggested peak cap (K)**: `{suggested_k}` (covers 95th percentile = {peaks_q95:.0f}, 99th = {peaks_q99:.0f})
- **m/z bin range**: `[0, {suggested_mz_max}]` Da
- **m/z bin resolution**: `{suggested_bin_res}` Da (unit-resolution binning; use 0.01 for high-res models)
- **Intensity normalization**: already base-peak normalized in data; apply sqrt-transform for spectral matching

### Formula / Mass Constraints
- **Max formula MW**: `{suggested_max_formula_mw}` Da (99th percentile MW = {mw_q99:.0f})
- **Adduct mass match rate**: `{mol_stats['mass_sanity']['frac_match']:.2%}` within 10 ppm — adduct table is reliable
- **Unknown adducts**: {len(_unknown_adducts)} types encountered

### Train/Val Split Strategy
- **Molecule-disjoint split** is mandatory (multiple spectra per molecule)
- **Scaffold-disjoint split** is recommended ({n_scaffolds} unique scaffolds across {n_molecules} molecules)
  - Use `MurckoScaffold.GetScaffoldForMol` → group by canonical scaffold SMILES
  - Alternative: InChIKey14-disjoint (no scaffold computation needed)

### Library Lookup Potential
- **{overlap_stats['frac_test_mass_covered']:.1%}** of test molecules have a matching mass (±{cfg.mass_match_ppm} ppm) in training set
- This is an UPPER BOUND: many isobars will have different structures
- A pure mass-lookup baseline would cover at most this fraction

### Data Quality Notes
- RDKit parse rate: {mol_stats['rdkit_parse_rate']:.2%}
- Multi-fragment (salt) SMILES: {mol_stats['frac_multi_fragment']:.2%} — strip to largest fragment
- Charged SMILES: {mol_stats['frac_charged']:.2%}
\"\"\"

# Display as markdown (in Jupyter this renders nicely)
from IPython.display import Markdown, display
display(Markdown(recommendations))

# Also save as text
rec_path = os.path.join(cfg.output_dir, "recommendations.md")
with open(rec_path, "w") as f:
    f.write(recommendations)
print(f"Recommendations saved to {rec_path}")
print(f"\\nTotal runtime: {elapsed():.1f}s")
""")

# ── Cell 21: Sanity assertions ─────────────────────────────────────────────
md("## Final Sanity Checks")
code("""
# ── SANITY CHECKS ──────────────────────────────────────────────────────────
import os

assert os.path.exists(os.path.join(cfg.output_dir, "schema.json")), "schema.json missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "train_clean.parquet")), "train_clean.parquet missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "test_clean.parquet")), "test_clean.parquet missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "stats.json")), "stats.json missing!"
assert os.path.exists(os.path.join(cfg.output_dir, "manifest.json")), "manifest.json missing!"

# Validate clean parquets
tc = pd.read_parquet(os.path.join(cfg.output_dir, "train_clean.parquet"))
expected_train_cols = {"molecule_id", "smiles", "ik14", "precursor_mz", "adduct",
                       "polarity", "ce", "instrument", "mz", "inten", "neutral_mass"}
assert expected_train_cols.issubset(set(tc.columns)), \\
    f"train_clean missing columns: {expected_train_cols - set(tc.columns)}"
assert len(tc) > 0, "train_clean is empty!"
print(f"train_clean: {tc.shape}, columns OK")

tt = pd.read_parquet(os.path.join(cfg.output_dir, "test_clean.parquet"))
expected_test_cols = {"molecule_id", "precursor_mz", "adduct", "polarity",
                      "ce", "instrument", "mz", "inten", "neutral_mass"}
assert expected_test_cols.issubset(set(tt.columns)), \\
    f"test_clean missing columns: {expected_test_cols - set(tt.columns)}"
assert "smiles" not in tt.columns, "test_clean should NOT have smiles!"
assert "ik14" not in tt.columns, "test_clean should NOT have ik14!"
assert len(tt) > 0, "test_clean is empty!"
print(f"test_clean:  {tt.shape}, columns OK, no leakage")

# Validate schema.json
with open(os.path.join(cfg.output_dir, "schema.json")) as f:
    schema_check = json.load(f)
assert "train" in schema_check and "test" in schema_check
print("schema.json: valid")

del tc, tt
gc.collect()

print()
print("=" * 60)
print("ALL SANITY CHECKS PASSED ✓")
print(f"Artifacts directory: {cfg.output_dir}")
print(f"Total runtime: {elapsed():.1f}s")
print("=" * 60)
""")

# ── Assemble notebook ─────────────────────────────────────────────────────
nb.cells = cells

# Validate
nbf.validate(nb)

# Syntax check every code cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        try:
            ast.parse(cell.source)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in code cell {i}: {e}", file=sys.stderr)
            sys.exit(1)

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_eda_schema.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook saved to: {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown)")
print("All code cells pass syntax check.")
print("nbformat validation passed.")
