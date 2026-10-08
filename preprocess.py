"""preprocess.py - Preprocessing and schema adaptation pipeline for LC-MS/MS data.

Consolidates the preprocessing and cleaning pipeline from Stage 01 (EDA & Schema Discovery)
to clean raw mass spectrometry train and test sets offline.
"""

from collections import Counter
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── ADDUCT MAPPING CONSTANTS ─────────────────────────────────────────────────
PROTON_MASS: float = 1.00727646677

# Mapping: adduct_str -> (polarity, charge, mass_shift)
# neutral_mass = precursor_mz * charge - shift
ADDUCT_TABLE: Dict[str, Tuple[str, int, float]] = {
    "[M+H]+":       ("positive", 1,  PROTON_MASS),
    "[M+Na]+":      ("positive", 1,  22.989218),
    "[M+K]+":       ("positive", 1,  38.963158),
    "[M+NH4]+":     ("positive", 1,  18.033823),
    "[M+H-H2O]+":   ("positive", 1,  PROTON_MASS - 18.010565),
    "[M+2H]2+":     ("positive", 2,  PROTON_MASS),       # doubly charged
    "[M-H]-":       ("negative", 1, -PROTON_MASS),
    "[M+FA-H]-":    ("negative", 1,  44.998201 - PROTON_MASS),
    "[M+Cl]-":      ("negative", 1,  34.969402),
    "[M-2H]2-":     ("negative", 2, -PROTON_MASS),       # doubly charged
    "[M+HAc-H]-":   ("negative", 1,  60.021129 - PROTON_MASS),  # acetate
    "[M-H2O+H]+":   ("positive", 1,  PROTON_MASS - 18.010565),
    "[M-H2O-H]-":   ("negative", 1, -PROTON_MASS - 18.010565),
    "[M+2Na-H]+":   ("positive", 1,  2 * 22.989218 - PROTON_MASS),
}

_UNKNOWN_ADDUCTS: Counter = Counter()


# ── NEUTRAL MASS CALCULATOR ──────────────────────────────────────────────────
def neutral_mass_from_precursor(
    precursor_mz: Union[float, np.ndarray, pd.Series],
    adduct: Union[str, np.ndarray, pd.Series],
    polarity: Optional[Union[str, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Compute neutral monoisotopic mass from precursor m/z and adduct.

    Supports scalar inputs as well as vectorized numpy arrays or pandas Series.
    Fallback logic: adds PROTON_MASS for negative polarity, subtracts PROTON_MASS
    for positive polarity when adduct is unknown.
    """
    if np.isscalar(precursor_mz):
        adduct_str = str(adduct).strip() if adduct is not None else ""
        if adduct_str in ADDUCT_TABLE:
            _, charge, shift = ADDUCT_TABLE[adduct_str]
            return float(precursor_mz) * charge - shift

        _UNKNOWN_ADDUCTS[adduct_str] += 1
        pol = str(polarity).strip().lower() if polarity is not None else "positive"
        if "neg" in pol or pol.startswith("-") or pol in ["negative", "neg", "-", "-1"]:
            return float(precursor_mz) + PROTON_MASS
        return float(precursor_mz) - PROTON_MASS

    # Vectorized execution
    pmz = np.asarray(precursor_mz, dtype=np.float64)
    adduct_s = pd.Series(adduct) if not isinstance(adduct, pd.Series) else adduct.copy()
    if polarity is not None:
        pol_s = pd.Series(polarity, index=adduct_s.index) if not isinstance(polarity, pd.Series) else polarity
    else:
        pol_s = pd.Series("", index=adduct_s.index)

    charge_map = {k: v[1] for k, v in ADDUCT_TABLE.items()}
    shift_map = {k: v[2] for k, v in ADDUCT_TABLE.items()}

    clean_adduct = adduct_s.astype(str).str.strip()
    charges = clean_adduct.map(charge_map).fillna(1.0).to_numpy(dtype=np.float64)

    is_neg = (
        pol_s.astype(str).str.lower().str.contains("neg")
        | pol_s.astype(str).str.startswith("-")
        | pol_s.astype(str).isin(["negative", "neg", "-", "-1"])
    )
    default_shift = np.where(is_neg, -PROTON_MASS, PROTON_MASS)

    shifts = (
        clean_adduct.map(shift_map)
        .fillna(pd.Series(default_shift, index=adduct_s.index))
        .to_numpy(dtype=np.float64)
    )

    res = pmz * charges - shifts
    if isinstance(precursor_mz, pd.Series):
        return pd.Series(res, index=precursor_mz.index, name="neutral_mass")
    return res


# ── SCHEMA ADAPTER ───────────────────────────────────────────────────────────
class SchemaAdapter:
    """Auto-detect column mapping from raw tabular mass spec inputs.

    Handles standard column names, case-insensitive variations, nested arrays,
    and JSON-encoded strings.
    """

    CANDIDATES: Dict[str, List[str]] = {
        "molecule_id": [
            "molecule_id", "mol_id", "compound_id", "id",
        ],
        "spectrum_id": [
            "spectrum_id", "spec_id", "scan_id",
        ],
        "smiles": [
            "normalized_smiles", "smiles", "SMILES", "canonical_smiles",
            "structure", "smi",
        ],
        "precursor_mz": [
            "precursor_mz", "precursor_mass", "prec_mz",
            "parent_mz", "precursormz", "mz",
        ],
        "adduct": [
            "adduct", "adduct_type", "precursor_type", "ion_type",
        ],
        "adduct_orig": [
            "adduct_orig", "original_adduct",
        ],
        "polarity": [
            "ionization_mode", "polarity", "ion_mode", "mode", "ionmode",
        ],
        "collision_energy": [
            "collision_energy_ev", "collision_energy", "ce", "CE", "energy",
        ],
        "collision_energy_orig": [
            "collision_energy_orig",
        ],
        "collision_energy_orig_units": [
            "collision_energy_orig_units",
        ],
        "instrument": [
            "instrument_type", "instrument", "inst_type", "instrument_name",
        ],
        "formula": [
            "molecular_formula", "formula", "mol_formula", "molecularformula",
        ],
        "inchikey": [
            "inchikey", "InChIKey", "inchi_key",
        ],
        "inchikey14": [
            "inchikey14", "InChIKey14", "ik14",
        ],
        "mz_array": [
            "ms2_mzs", "mz_array", "peaks_mz", "mzs",
            "fragment_mz", "mz_values", "mz",
        ],
        "intensity_array": [
            "ms2_normalized_intensities", "intensity_array",
            "peaks_intensity", "intensities", "normalized_intensities",
            "intensity_values", "inten",
        ],
        "ingest_lib": [
            "ingest_lib", "library", "source_lib", "source",
        ],
        "base_peak_intensity": [
            "base_peak_intensity", "base_peak_int",
        ],
    }

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = "unknown",
        check_critical: bool = True,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.mapping: Dict[str, str] = {}
        self.col_types: Dict[str, str] = {}
        cols_lower = {c.lower(): c for c in df.columns}

        for field, candidates in self.CANDIDATES.items():
            found = None
            for cand in candidates:
                # Exact match first
                if cand in df.columns:
                    found = cand
                    break
                # Case-insensitive match fallback
                if cand.lower() in cols_lower:
                    found = cols_lower[cand.lower()]
                    break
            if found is not None:
                self.mapping[field] = found
                self.col_types[field] = self._detect_type(df, found)

        if verbose:
            print(f"SchemaAdapter({name}): detected {len(self.mapping)}/{len(self.CANDIDATES)} fields")
            for field, col in self.mapping.items():
                print(f"  {field:30s} -> {col:35s} [{self.col_types.get(field, '?')}]")

        if check_critical:
            missing_critical = []
            for req in ["molecule_id", "precursor_mz", "mz_array", "intensity_array"]:
                if req not in self.mapping:
                    missing_critical.append(req)
            if missing_critical:
                raise ValueError(
                    f"SchemaAdapter({name}): CRITICAL columns not found: {missing_critical}\n"
                    f"Available columns: {list(df.columns)}"
                )

    def _detect_type(self, df: pd.DataFrame, col: str) -> str:
        """Detect whether a column stores scalars, lists, nested structs, or JSON strings."""
        sample = df[col].dropna().iloc[:5] if len(df) > 0 else pd.Series(dtype=object)
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

    def col(self, field: str) -> Optional[str]:
        """Return the physical column name mapped to the semantic field."""
        return self.mapping.get(field, None)

    def has(self, field: str) -> bool:
        """Return True if the semantic field was successfully mapped."""
        return field in self.mapping

    def get(self, field: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """Retrieve values for a semantic field, parsing JSON lists if needed."""
        if field not in self.mapping:
            return None
        cname = self.mapping[field]
        ctype = self.col_types.get(field, "unknown")
        series = df[cname]

        if ctype == "json_list":
            return series.apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return series

    def to_dict(self) -> Dict[str, Any]:
        """Export mapping and detected data types as a dictionary."""
        return {
            "mapping": self.mapping,
            "col_types": self.col_types,
        }


# ── DATA NORMALIZATION ENTRY POINT ───────────────────────────────────────────
def clean_raw_df(
    df: pd.DataFrame,
    adapter: Optional[SchemaAdapter] = None,
    is_train: bool = False,
) -> pd.DataFrame:
    """Standardize raw spectral dataframes into the canonical schema.

    Canonical Schema Contract:
      If is_train=True:
        ["molecule_id", "smiles", "ik14", "precursor_mz", "adduct",
         "polarity", "ce", "instrument", "mz", "inten", "neutral_mass"]
      If is_train=False:
        ["molecule_id", "precursor_mz", "adduct", "polarity", "ce",
         "instrument", "mz", "inten", "neutral_mass"]

    Key guarantees:
      - Vectorized arithmetic for charges and shifts.
      - Zero-copy peak array series assignments.
      - Proper numeric casting (precursor_mz -> float64, ce -> float32).
    """
    n_rows = len(df)
    if adapter is None:
        adapter = SchemaAdapter(
            df,
            name="train" if is_train else "test",
            check_critical=True,
        )

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
        prefix = "train" if is_train else "test"
        out["molecule_id"] = [f"{prefix}_mol_{i}" for i in range(n_rows)]

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

    # 7. mz & inten - Zero-copy peak series assignment
    mz_series = adapter.get("mz_array", df)
    int_series = adapter.get("intensity_array", df)
    out["mz"] = mz_series
    out["inten"] = int_series

    # 8. neutral_mass - Vectorized lookup arithmetic
    charge_map = {k: v[1] for k, v in ADDUCT_TABLE.items()}
    shift_map = {k: v[2] for k, v in ADDUCT_TABLE.items()}

    clean_adduct = out["adduct"].astype(str).str.strip()
    charges = clean_adduct.map(charge_map).fillna(1.0).to_numpy(dtype=np.float64)

    is_neg = (
        out["polarity"].astype(str).str.lower().str.contains("neg")
        | out["polarity"].astype(str).str.startswith("-")
        | out["polarity"].astype(str).isin(["negative", "neg", "-", "-1"])
    )
    default_shift = np.where(is_neg, -PROTON_MASS, PROTON_MASS)

    shifts = (
        clean_adduct.map(shift_map)
        .fillna(pd.Series(default_shift, index=out.index))
        .to_numpy(dtype=np.float64)
    )
    out["neutral_mass"] = out["precursor_mz"].to_numpy(dtype=np.float64) * charges - shifts

    # 9. Train-only: smiles and ik14
    if is_train:
        smi_c = adapter.col("smiles")
        out["smiles"] = df[smi_c].astype(str) if (smi_c and smi_c in df.columns) else ""

        ik14_c = adapter.col("inchikey14")
        if ik14_c and ik14_c in df.columns:
            out["ik14"] = df[ik14_c].astype(str)
        elif "inchikey14" in df.columns:
            out["ik14"] = df["inchikey14"].astype(str)
        elif "molecule_id" in df.columns:
            out["ik14"] = df["molecule_id"].astype(str)
        else:
            out["ik14"] = out["molecule_id"]

        cols = [
            "molecule_id", "smiles", "ik14", "precursor_mz", "adduct",
            "polarity", "ce", "instrument", "mz", "inten", "neutral_mass",
        ]
    else:
        cols = [
            "molecule_id", "precursor_mz", "adduct", "polarity", "ce",
            "instrument", "mz", "inten", "neutral_mass",
        ]

    return out[cols]


# Stage 01 backward compatibility alias
build_clean_df = clean_raw_df
