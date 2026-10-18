"""Generate the Kaggle candidate database notebook; no dataset or module imports."""
from pathlib import Path
import ast
import argparse
import nbformat as nbf


# This source is embedded verbatim in %%writefile cells. It is never imported
# from this repository: the generated notebook imports /kaggle/working only.
MODULE_SOURCE = r'''
"""Disk-backed candidate retrieval. Import only from /kaggle/working.

Candidate pools uniformly use fast_key, standardized with standardize_mol().
Use finalize_candidates AFTER ranking to compute canon_ik14 only on the filtered
top-256 candidates per query for tautomer-canonical deduplication.
Row IDs are stable positions in db.parquet and all row-aligned .npy files.
"""
import csv
import gc
import gzip
import hashlib
import heapq
import importlib.util
import io
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import Counter, OrderedDict
from dataclasses import asdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")
WORK = Path("/kaggle/working")
_max_csv = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_csv)
        break
    except OverflowError:
        _max_csv = int(_max_csv / 10)
BASE_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
EXTRA_ELEMENTS = {"Na", "K", "Ca", "Li", "Mg", "Fe", "Zn", "Co", "Cu", "Mn", "Mo", "Ni", "V", "W", "B", "Si", "Se", "As"}
TRAIN_BIT = 1
DB_COLUMNS = ["row_id", "ik14", "smiles", "formula", "mass",
              "source_mask", "fast_key", "identity_kind"]
CTX = None
EK = None


def find_file(filename, hint="", required=True, roots=None):
    if filename.endswith(".py"):
        paths = [WORK / filename] if (WORK / filename).is_file() else []
    else:
        roots = roots or [WORK, Path("/kaggle/input")]
        paths = list({p.resolve() for root in roots for p in Path(root).rglob(filename)})
    paths.sort(key=lambda p: (bool(hint and hint in str(p)), p.stat().st_mtime,
                              str(p)), reverse=True)
    if paths:
        print(f"[find_file] {filename}: {paths[0]}")
        return paths[0]
    if required:
        raise FileNotFoundError(f"Missing {filename}; modules must be in {WORK}")
    print(f"[find_file] optional {filename}: absent")
    return None


def find_manifest(stage):
    matches = []
    for root in (WORK, Path("/kaggle/input")):
        for path in root.rglob("manifest*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                print(f"Unreadable manifest {path}: {exc}")
                continue
            if record.get("stage") == stage:
                matches.append((path.stat().st_mtime, path, record))
    return max(matches, default=(None, None, None), key=lambda x: x[0])[1:]


def load_evalkit():
    global EK
    if EK is None:
        path = find_file("evalkit.py")
        if path.resolve() != (WORK / "evalkit.py").resolve():
            raise RuntimeError("evalkit.py must be read only from /kaggle/working")
        spec = importlib.util.spec_from_file_location("evalkit", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["evalkit"] = module
        spec.loader.exec_module(module)
        for name in ("FP_SPEC", "fast_key", "canon_ik14", "mrr_at_25"):
            if not hasattr(module, name):
                raise AttributeError(f"evalkit.py is missing {name}")
        EK = module
    return EK


def open_sql(path, readonly=False):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri if readonly else str(path), uri=readonly, timeout=90)
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA mmap_size=0")
    if not readonly:
        db.execute("PRAGMA journal_mode=DELETE")
    return db


def resource_guard(cfg, reserve_bytes=0):
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if rss + reserve_bytes > cfg.RAM_GB * 1e9:
        raise MemoryError(f"RAM guard: {rss/1e9:.2f} GB exceeds {cfg.RAM_GB} GB")
    free = shutil.disk_usage(cfg.output_dir).free
    if free < cfg.MIN_FREE_DISK_GB * 1e9:
        raise OSError(f"Only {free/1e9:.2f} GB disk free; insufficient space")
    return rss


def batched(iterable, size):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    load_evalkit()


def pool_map(pool, function, iterable, cfg, total=None, desc="Processing"):
    with tqdm(total=total, desc=desc, unit="rows") as progress:
        for wave in batched(iterable, cfg.WORKERS * cfg.CHUNK_ROWS):
            jobs = [pool.apply_async(function, (chunk,))
                    for chunk in batched(wave, cfg.CHUNK_ROWS)]
            for job in jobs:
                result = job.get()
                yield result
                progress.update(len(result))
            resource_guard(cfg)


def standardize_mol(mol_or_smiles, allowed=None):
    if isinstance(mol_or_smiles, str):
        if not mol_or_smiles or not mol_or_smiles.strip():
            return None
        mol = Chem.MolFromSmiles(mol_or_smiles.strip())
    else:
        mol = mol_or_smiles
    if mol is None:
        return None
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if mol is None:
            return None
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        if mol is None or Chem.GetFormalCharge(mol) != 0:
            return None
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 1 or n_heavy > 150:
            return None
        if allowed is not None:
            atoms = {a.GetSymbol() for a in mol.GetAtoms()}
            if not atoms.issubset(allowed):
                return None
        return mol
    except Exception:
        return None


def parent_mol(smiles, allowed):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None, "invalid"
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, "invalid"
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if mol is None:
            return None, "fragment_failure"
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        if mol is None or Chem.GetFormalCharge(mol) != 0:
            return None, "non_neutralizable"
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 1 or n_heavy > 150:
            return None, "heavy_atoms_out_of_bounds"
        atoms = {a.GetSymbol() for a in mol.GetAtoms()}
        if not atoms.issubset(allowed):
            return None, "disallowed_element"
        return mol, "ok"
    except Exception:
        return None, "rdkit_failure"


def standardize_chunk(tasks):
    ek = load_evalkit()
    results = []
    for raw, source, allowed, known_ik14 in tasks:
        mol, status = parent_mol(raw, set(allowed))
        row = None
        if mol is not None:
            key = ek.fast_key(mol)
            ik14 = known_ik14 if (known_ik14 and isinstance(known_ik14, str)) else None
            if not key:
                status = "key_failure"
            else:
                row = (key, Chem.MolToSmiles(mol, isomericSmiles=False),
                       rdMolDescriptors.CalcMolFormula(mol),
                       float(rdMolDescriptors.CalcExactMolWt(mol)), source, ik14,
                       ",".join(sorted({a.GetSymbol() for a in mol.GetAtoms()})))
        results.append((raw, status, row))
    return results


def fp_layout():
    spec = load_evalkit().FP_SPEC
    morgan = spec["morgan"]
    if morgan.get("chirality", False):
        raise ValueError("FP_SPEC must use achiral Morgan fingerprints")
    widths = [int(morgan["nBits"])]
    if "maccs" in spec:
        if int(spec["maccs"]["nBits"]) != 167:
            raise ValueError("RDKit MACCS requires FP_SPEC.maccs.nBits=167")
        widths.append(167)
    return widths, sum((n + 7) // 8 for n in widths)


def fingerprint_chunk(rows):
    spec = load_evalkit().FP_SPEC
    widths, _ = fp_layout()
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(spec["morgan"]["radius"]),
        fpSize=int(spec["morgan"]["nBits"]), includeChirality=False)
    result = []
    for row_id, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid standardized SMILES at row {row_id}")
        arrays = [gen.GetFingerprintAsNumPy(mol).astype(np.uint8)]
        if len(widths) == 2:
            bits = np.zeros(167, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), bits)
            arrays.append(bits)
        packed = np.concatenate([np.packbits(a, bitorder="little") for a in arrays])
        result.append((row_id, packed))
    return result


class SchemaAdapter:
    aliases = {
        "molecule_id": ["molecule_id", "inchikey14", "id"],
        "smiles": ["smiles", "canonical_smiles", "canonicalsmiles", "normalized_smiles",
                   "isomeric_smiles", "isomericsmiles", "smiles_iso", "structure_smiles"],
        "neutral_mass": ["neutral_mass", "neutral_precursor_mass"],
        "precursor_mz": ["precursor_mz", "precursormz", "parent_mz"],
        "adduct": ["adduct", "precursor_type"],
        "polarity": ["polarity", "ionization_mode", "ionmode"],
        "split3": ["split3"],
    }

    def __init__(self, columns, name, required=(), schema=None):
        self.name = name
        lower = {str(c).lower(): c for c in columns}
        self.mapping = {}
        hints = (schema or {}).get(name, {}).get("mapping", {})
        for field, aliases in self.aliases.items():
            choices = aliases + [hints.get(field, "")]
            for alias in choices:
                if alias.lower() in lower:
                    self.mapping[field] = lower[alias.lower()]
                    break
        missing = set(required) - set(self.mapping)
        if missing:
            raise ValueError(f"SchemaAdapter({name}): missing {sorted(missing)}; columns={columns}")
        print(f"SchemaAdapter({name}): {self.mapping}")

    def adapt(self, table):
        frame = table.to_pandas()
        return frame.rename(columns={v: k for k, v in self.mapping.items()})


def observed_mass(frame):
    mass = pd.to_numeric(frame.get("neutral_mass", pd.Series(np.nan, index=frame.index)),
                         errors="coerce").to_numpy(dtype=np.float64)
    missing = ~np.isfinite(mass) | (mass <= 0)
    if not missing.any():
        return mass
    if "precursor_mz" not in frame:
        raise ValueError("Missing valid observed neutral_mass and precursor_mz")
    add = frame.get("adduct", pd.Series("", index=frame.index)).fillna("").astype(str)
    pol = frame.get("polarity", pd.Series("+", index=frame.index)).fillna("+").astype(str)
    table = {"[M+H]+": (1, 1.007276), "[M-H]-": (1, -1.007276),
             "[M+Na]+": (1, 22.989218), "[M+K]+": (1, 38.963158),
             "[M+NH4]+": (1, 18.033823), "[M+Cl]-": (1, 34.969402),
             "[M+H-H2O]+": (1, -17.003289), "[M+FA-H]-": (1, 44.998201),
             "[M+2H]2+": (2, 2.014552), "[M-2H]2-": (2, -2.014552)}
    negative = pol.str.lower().isin(["negative", "neg", "-", "-1"])
    charge = add.map({k: v[0] for k, v in table.items()}).fillna(1).to_numpy()
    shift = add.map({k: v[1] for k, v in table.items()}).to_numpy()
    shift = np.where(np.isfinite(shift), shift, np.where(negative, -1.007276, 1.007276))
    mz = pd.to_numeric(frame.precursor_mz, errors="coerce").to_numpy(dtype=np.float64)
    mass[missing] = (mz * charge - shift)[missing]
    if (~np.isfinite(mass) | (mass <= 0)).any():
        raise ValueError("Non-positive/invalid observed neutral masses after adduct fallback")
    return mass


def initialize_stage(cfg):
    path = Path(cfg.output_dir) / "build.sqlite"
    db = open_sql(path)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS molecules(id TEXT PRIMARY KEY, smiles TEXT,
        mass_sum REAL, n INTEGER, lo REAL, hi REAL, split3 TEXT, ik14 TEXT);
    CREATE TABLE IF NOT EXISTS tests(id TEXT PRIMARY KEY, smiles TEXT,
        mass_sum REAL, n INTEGER, lo REAL, hi REAL, split3 TEXT);
    CREATE TABLE IF NOT EXISTS splits(id TEXT PRIMARY KEY, split3 TEXT, ik14 TEXT);
    CREATE TABLE IF NOT EXISTS train_structures(raw TEXT PRIMARY KEY,
        fast_key TEXT, ik14 TEXT, smiles TEXT, formula TEXT, mass REAL, status TEXT);
    CREATE TABLE IF NOT EXISTS structures(fast_key TEXT PRIMARY KEY, smiles TEXT,
        formula TEXT, mass REAL, source_mask INTEGER, ik14 TEXT, elements TEXT);
    CREATE TABLE IF NOT EXISTS sources(path TEXT PRIMARY KEY, done INTEGER);
    """)
    return db


def state_get(db, key, default=None):
    row = db.execute("SELECT v FROM state WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def state_set(db, key, value):
    db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))
    db.commit()


def ingest_metadata(db, path, kind, cfg, schema=None):
    if state_get(db, f"metadata_{kind}"):
        return
    table = "molecules" if kind == "train" else "tests"
    db.execute(f"DELETE FROM {table}")
    file = pq.ParquetFile(path)
    required = ("molecule_id", "smiles") if kind == "train" else ("molecule_id",)
    adapter = SchemaAdapter(file.schema_arrow.names, kind, required, schema)
    columns = [v for k, v in adapter.mapping.items() if k not in ("smiles", "split3")]
    if kind == "train":
        columns.append(adapter.mapping["smiles"])
    for batch in tqdm(file.iter_batches(cfg.BATCH_ROWS, columns=columns), desc=f"{kind} metadata"):
        frame = adapter.adapt(batch)
        if frame.molecule_id.isna().any():
            raise ValueError(f"Null molecule_id in {path}")
        frame["id"] = frame.molecule_id.astype(str)
        frame["mass"] = observed_mass(frame)
        frame["smiles"] = frame.smiles.fillna("").astype(str) if kind == "train" else ""
        groups = frame.groupby("id", sort=False)
        conflict_in_batch = (groups.smiles.nunique() > 1).sum()
        if conflict_in_batch > 0:
            print(f"Warning: {conflict_in_batch} molecule IDs have conflicting SMILES within batch; retaining first")
        grouped = groups.agg(smiles=("smiles", "first"), mass_sum=("mass", "sum"),
                             n=("mass", "size"), lo=("mass", "min"), hi=("mass", "max"))
        rows = list(grouped.reset_index().itertuples(index=False, name=None))
        db.execute("CREATE TEMP TABLE incoming(id TEXT, smiles TEXT, mass_sum REAL, n INT, lo REAL, hi REAL)")
        db.executemany("INSERT INTO incoming VALUES (?,?,?,?,?,?)", rows)
        conflict = db.execute(f"SELECT COUNT(*) FROM incoming i JOIN {table} t USING(id) "
                              "WHERE i.smiles != t.smiles").fetchone()[0]
        if conflict:
            print(f"Warning: {conflict} molecule IDs have conflicting SMILES across batches; retaining initial in {table}")
        db.execute(f"INSERT INTO {table} (id, smiles, mass_sum, n, lo, hi) "
                   f"SELECT id, smiles, mass_sum, n, lo, hi FROM incoming WHERE 1 "
                   f"ON CONFLICT(id) DO UPDATE SET mass_sum={table}.mass_sum+excluded.mass_sum, "
                   f"n={table}.n+excluded.n, lo=min({table}.lo,excluded.lo), hi=max({table}.hi,excluded.hi)")
        db.execute("DROP TABLE incoming")
        db.commit()
        resource_guard(cfg)
    state_set(db, f"metadata_{kind}", True)


def ingest_splits(db, path, cfg):
    if state_get(db, "splits_done"):
        return
    db.execute("DELETE FROM splits")
    file = pq.ParquetFile(path)
    has_ik14 = "ik14" in file.schema_arrow.names
    adapter = SchemaAdapter(file.schema_arrow.names, "splits", ("molecule_id", "split3"))
    columns = [adapter.mapping[k] for k in ("molecule_id", "split3")]
    if has_ik14 and "ik14" not in columns:
        columns.append("ik14")
    for batch in file.iter_batches(cfg.BATCH_ROWS, columns=columns):
        frame = adapter.adapt(batch)
        if frame.isna().any().any() or not frame.split3.isin(["train", "val_model", "val_rank"]).all():
            raise ValueError("splits must contain non-null, unique molecule_id and valid split3")
        ik14_vals = frame["ik14"].astype(str) if "ik14" in frame else [None] * len(frame)
        try:
            db.executemany("INSERT INTO splits VALUES (?,?,?)",
                           zip(frame.molecule_id.astype(str), frame.split3.astype(str), ik14_vals))
        except sqlite3.IntegrityError as exc:
            raise ValueError("splits has duplicate molecule_id assignments") from exc
    mismatch = db.execute("SELECT COUNT(*) FROM (SELECT id FROM molecules EXCEPT SELECT id FROM splits)").fetchone()[0]
    extra = db.execute("SELECT COUNT(*) FROM (SELECT id FROM splits EXCEPT SELECT id FROM molecules)").fetchone()[0]
    if mismatch or extra:
        raise ValueError(f"splits/train ID mismatch: missing={mismatch}, extra={extra}")
    db.execute("UPDATE molecules SET split3=(SELECT split3 FROM splits WHERE splits.id=molecules.id), "
               "ik14=(SELECT ik14 FROM splits WHERE splits.id=molecules.id)")
    db.execute("CREATE INDEX IF NOT EXISTS mol_smiles ON molecules(smiles)")
    db.execute("CREATE INDEX IF NOT EXISTS mol_split ON molecules(split3)")
    state_set(db, "splits_done", True)


UPSERT = """INSERT INTO structures VALUES (?,?,?,?,?,?,?)
ON CONFLICT(fast_key) DO UPDATE SET
 source_mask=structures.source_mask|excluded.source_mask,
 ik14=coalesce(structures.ik14,excluded.ik14)"""


def prepare_train(db, pool, cfg):
    if state_get(db, "train_done"):
        return state_get(db, "allowed_elements")
    total = db.execute("SELECT COUNT(DISTINCT smiles) FROM molecules").fetchone()[0]
    cursor = db.execute("SELECT smiles, max(ik14) FROM molecules GROUP BY smiles ORDER BY smiles")
    allowed = BASE_ELEMENTS | EXTRA_ELEMENTS
    tasks = ((r[0], TRAIN_BIT, tuple(sorted(allowed)), r[1]) for r in cursor)
    statuses = Counter()
    found_elements = set()
    for result in pool_map(pool, standardize_chunk, tasks, cfg, total, "Train chemistry"):
        good = []
        mapping = []
        for raw, status, row in result:
            statuses[status] += 1
            if row:
                key, smiles, formula, mass, source, ik14, elements = row
                good.append(row)
                found_elements.update(elements.split(","))
                mapping.append((raw, key, ik14, smiles, formula, mass, status))
            else:
                mapping.append((raw, None, None, None, None, None, status))
        if not cfg.EXCLUDE_TRAIN_FROM_DB:
            db.executemany(UPSERT, good)
        db.executemany("INSERT OR REPLACE INTO train_structures VALUES (?,?,?,?,?,?,?)", mapping)
        db.commit()
    allowed = sorted(BASE_ELEMENTS | EXTRA_ELEMENTS)
    state_set(db, "allowed_elements", allowed)
    state_set(db, "train_rejections", dict(statuses))
    state_set(db, "train_done", True)
    db.execute("CREATE INDEX IF NOT EXISTS train_fast ON train_structures(fast_key)")
    print("Train chemistry:", dict(statuses), "allowed elements:", allowed)
    leakage = db.execute("SELECT COUNT(*) FROM (SELECT t.fast_key FROM molecules m JOIN train_structures t "
                         "ON m.smiles=t.raw WHERE t.fast_key IS NOT NULL GROUP BY t.fast_key "
                         "HAVING COUNT(DISTINCT m.split3)>1)").fetchone()[0]
    if leakage:
        raise ValueError(f"{leakage} standardized fast_key identities cross split3; regenerate splits")
    return allowed


def smiles_column(columns, sample):
    names = {str(c).lower(): c for c in columns}
    preferred = SchemaAdapter.aliases["smiles"] + ["compound_smiles"]
    candidates = [names[s] for s in preferred if s in names]
    candidates += [c for c in columns if "smile" in str(c).lower() and c not in candidates]
    candidates += [c for c in columns if c not in candidates]
    for col in candidates:
        values = [v for v in sample[col].tolist() if isinstance(v, str) and v.strip()][:32]
        if not values:
            continue
        fraction = sum(Chem.MolFromSmiles(v) is not None for v in values) / len(values)
        if fraction >= (0.1 if "smile" in str(col).lower() else 0.8):
            return col
    return None


def detect_delimiter(text, fmt):
    if fmt == "tsv":
        return "\t"
    if fmt == "csv":
        return ","
    if fmt == "smi":
        return "\t" if "\t" in text[:1000] else " "
    try:
        return csv.Sniffer().sniff(text, delimiters="\t, ;|").delimiter
    except csv.Error:
        return "\t" if "\t" in text else "," if "," in text else " "


def probe_zip_member(zf, member, m_fmt, m_size, path_str):
    if m_fmt == "sdf":
        return {"path": path_str, "zip_member": member, "format": "sdf",
                "is_gz": False, "is_zip": True, "bytes": m_size,
                "column": "<SDF mol block>", "rows_estimate": None}
    with zf.open(member, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="") as handle:
            lines = list(itertools.islice(handle, 129))
    if not lines:
        return None
    text = "".join(lines)
    delimiter = detect_delimiter(text, m_fmt)
    records = list(csv.reader(lines, delimiter=delimiter, skipinitialspace=True))
    records = [r for r in records if r]
    if not records:
        return None
    header = any("smile" in str(x).lower() for x in records[0])
    names = records[0] if header else list(range(len(records[0])))
    width = len(names)
    sample = pd.DataFrame([r[:width] + [None] * max(0, width - len(r))
                           for r in records[int(header):]], columns=names)
    col = smiles_column(names, sample)
    if col is None:
        return None
    mean_bytes = len(text.encode("utf-8")) / max(1, len(lines))
    return {"path": path_str, "zip_member": member, "format": m_fmt,
            "is_gz": False, "is_zip": True, "bytes": m_size,
            "column": col, "column_index": names.index(col), "header": header,
            "delimiter": delimiter,
            "rows_estimate": max(1, round(m_size / max(1.0, mean_bytes)))}


def probe_zip_source(path):
    p = Path(path)
    try:
        with zipfile.ZipFile(p, "r") as zf:
            candidates = [
                (info.filename, Path(info.filename).suffix.lower()[1:], info.file_size)
                for info in zf.infolist()
                if not info.is_dir() and "__MACOSX" not in info.filename
                and Path(info.filename).suffix.lower() in {".sdf", ".tsv", ".csv", ".smi", ".txt"}
            ]
            if not candidates:
                return None
            specs = [probe_zip_member(zf, m, fmt, sz, str(p)) for m, fmt, sz in candidates]
            valid = [s for s in specs if s is not None]
            return valid if valid else None
    except Exception as exc:
        print(f"[probe_zip_source] cannot inspect zip {path}: {exc}")
        return None


def probe_delimited_source(p, format_name, is_gz, spec):
    opener = (lambda: gzip.open(p, mode="rt", encoding="utf-8-sig", errors="replace", newline="")) if is_gz \
             else (lambda: p.open(encoding="utf-8-sig", errors="replace", newline=""))
    try:
        with opener() as handle:
            lines = list(itertools.islice(handle, 129))
    except Exception as exc:
        print(f"[probe_delimited_source] cannot open {p}: {exc}")
        return None
    if not lines:
        return None
    text = "".join(lines)
    delimiter = detect_delimiter(text, format_name)
    records = list(csv.reader(lines, delimiter=delimiter, skipinitialspace=True))
    records = [r for r in records if r]
    if not records:
        return None
    header = any("smile" in str(x).lower() for x in records[0])
    names = records[0] if header else list(range(len(records[0])))
    width = len(names)
    sample = pd.DataFrame([r[:width] + [None] * max(0, width - len(r))
                           for r in records[int(header):]], columns=names)
    col = smiles_column(names, sample)
    if col is None:
        return None
    mean_bytes = len(text.encode("utf-8")) / max(1, len(lines))
    approx_size = spec["bytes"] * 4 if is_gz else spec["bytes"]
    spec.update(column=col, column_index=names.index(col), header=header,
                delimiter=delimiter, rows_estimate=max(1, round(approx_size / max(1.0, mean_bytes))))
    return spec


def probe_source(path):
    p = Path(path)
    name_lower = p.name.lower()
    is_gz = name_lower.endswith(".gz")
    if name_lower.endswith(".zip"):
        return probe_zip_source(p)
    base_ext = Path(p.stem).suffix.lower() if is_gz else p.suffix.lower()
    format_name = base_ext[1:] if base_ext else "txt"
    spec = {"path": str(p), "format": format_name, "is_gz": is_gz, "is_zip": False,
            "bytes": p.stat().st_size}
    if format_name == "parquet":
        file = pq.ParquetFile(p)
        columns = [f.name for f in file.schema_arrow if pa.types.is_string(f.type)
                   or pa.types.is_large_string(f.type)]
        if not columns or not file.metadata.num_rows:
            return None
        sample = next(file.iter_batches(batch_size=128, columns=columns)).to_pandas()
        col = smiles_column(columns, sample)
        if col is not None:
            spec.update(column=col, rows_estimate=file.metadata.num_rows)
            return spec
        return None
    if format_name == "sdf":
        spec.update(column="<SDF mol block>", rows_estimate=None)
        return spec
    return probe_delimited_source(p, format_name, is_gz, spec)


def discover_sources(root, core_paths):
    valid_extensions = {".smi", ".csv", ".tsv", ".sdf", ".parquet", ".txt",
                        ".smi.gz", ".csv.gz", ".tsv.gz", ".sdf.gz", ".txt.gz", ".gz", ".zip"}
    excluded = {Path(p).resolve() for p in core_paths}
    core_dirs = {p.parent for p in excluded}
    reserved = {"train.parquet", "test.parquet", "train_clean.parquet", "test_clean.parquet",
                "splits.parquet", "sample_submission.csv", "db.parquet"}
    sources = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        p_name = path.name.lower()
        if not any(p_name.endswith(ext) for ext in valid_extensions):
            continue
        if path.resolve() in excluded or path.name in reserved or path.parent.resolve() in core_dirs:
            print(f"[discovery] core input/output excluded: {path}")
            continue
        try:
            probed = probe_source(path)
        except (OSError, ValueError, pa.ArrowException, UnicodeError, csv.Error) as exc:
            print(f"[discovery] cannot inspect {path}: {type(exc).__name__}: {exc}")
            continue
        if probed is None:
            print(f"[discovery] no SMILES-like column: {path}")
            continue
        specs = probed if isinstance(probed, list) else [probed]
        for spec in specs:
            try:
                spec["dataset"] = str(path.relative_to(root).parts[0])
            except ValueError:
                spec["dataset"] = path.stem
            if spec.get("is_zip") and spec.get("zip_member"):
                spec["dataset"] += f"_{Path(spec['zip_member']).stem}"
            sources.append(spec)
            desc = f"{spec.get('zip_member', '')} {spec['column']!r}"
            print(f"[discovery] FOUND {path}: {desc.strip()}, estimated rows={spec['rows_estimate']}")
    datasets = sorted({s["dataset"] for s in sources})
    if len(datasets) > 62:
        raise ValueError("uint64 provenance supports train + 62 external datasets; combine attachments")
    bits = {name: 1 << (i + 1) for i, name in enumerate(datasets)}
    for spec in sources:
        spec["source_mask"] = bits[spec["dataset"]]
    if not sources:
        print("No external structure databases found. Building DB from train SMILES only.")
    print("Source flags:", {"train": TRAIN_BIT, **bits})
    return sources


def iter_sdf_source(path, is_gz, is_zip, zip_member):
    if is_zip:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(zip_member, "r") as handle:
                for mol in Chem.ForwardSDMolSupplier(handle, removeHs=True):
                    yield Chem.MolToSmiles(mol) if mol is not None else None
    elif is_gz:
        with gzip.open(path, "rb") as handle:
            for mol in Chem.ForwardSDMolSupplier(handle, removeHs=True):
                yield Chem.MolToSmiles(mol) if mol is not None else None
    else:
        with path.open("rb") as handle:
            for mol in Chem.ForwardSDMolSupplier(handle, removeHs=True):
                yield Chem.MolToSmiles(mol) if mol is not None else None


def iter_delimited_source(path, spec):
    is_gz, is_zip = spec.get("is_gz", False), spec.get("is_zip", False)
    col = spec["column_index"]

    def _read(reader):
        if spec["header"]:
            next(reader, None)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                print(f"[delimited_source] warning: {exc}")
                continue
            yield row[col] if len(row) > col else None

    if is_zip:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(spec["zip_member"], "r") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="") as handle:
                    yield from _read(csv.reader(handle, delimiter=spec["delimiter"], skipinitialspace=True))
    elif is_gz:
        with gzip.open(path, mode="rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield from _read(csv.reader(handle, delimiter=spec["delimiter"], skipinitialspace=True))
    else:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield from _read(csv.reader(handle, delimiter=spec["delimiter"], skipinitialspace=True))


def iter_source(spec, batch_rows=8192):
    path = Path(spec["path"])
    fmt = spec["format"]
    if fmt == "parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_rows, columns=[spec["column"]]):
            yield from batch.column(0).to_pylist()
    elif fmt == "sdf":
        yield from iter_sdf_source(path, spec.get("is_gz", False), spec.get("is_zip", False), spec.get("zip_member"))
    else:
        yield from iter_delimited_source(path, spec)


def benchmark_standardization(sources, db, pool, allowed, cfg):
    target = min(500, cfg.BENCHMARK_ROWS) if cfg.SMOKE_TEST else cfg.BENCHMARK_ROWS
    source_iter = itertools.chain.from_iterable(iter_source(s) for s in sources)
    if not sources:
        source_iter = (r[0] for r in db.execute("SELECT DISTINCT smiles FROM molecules"))
    sample = list(itertools.islice(source_iter, target))
    tasks = ((s, 0, tuple(allowed), None) for s in sample)
    start = time.perf_counter()
    accepted = 0
    for result in pool_map(pool, standardize_chunk, tasks, cfg, len(sample), "20k chemistry benchmark"):
        accepted += sum(row is not None for _, _, row in result)
    seconds = time.perf_counter() - start
    rate = len(sample) / max(seconds, 1e-9)
    estimates = [s["rows_estimate"] for s in sources]
    estimated_rows = sum(n for n in estimates if n is not None) if sources else len(sample)
    report = {"sample_rows": len(sample), "accepted": accepted, "rows_per_second": rate,
              "estimated_rows": estimated_rows, "unknown_size_sources": estimates.count(None),
              "projected_seconds_known_sources": estimated_rows / max(rate, 1e-9)}
    print("Standardization projection:", report)
    return report


def ingest_external(db, sources, pool, allowed, cfg):
    all_stats = state_get(db, "external_rejections", {})
    for source in sources:
        src_key = source["path"] + (f"::{source['zip_member']}" if source.get("zip_member") else "")
        if db.execute("SELECT done FROM sources WHERE path=?", (src_key,)).fetchone() == (1,):
            print("Resume: completed external source", src_key)
            continue
        counts = Counter()
        tasks = ((s, source["source_mask"], tuple(allowed), None)
                 for s in iter_source(source, cfg.BATCH_ROWS))
        desc_name = f"{Path(source['path']).name}:{source['zip_member']}" if source.get("zip_member") else Path(source["path"]).name
        for result in pool_map(pool, standardize_chunk, tasks, cfg,
                               source["rows_estimate"], desc_name):
            rows = []
            for _, status, row in result:
                counts[status] += 1
                if row:
                    rows.append(row)
            db.executemany(UPSERT, rows)
            db.commit()
        db.execute("INSERT OR REPLACE INTO sources VALUES (?,1)", (src_key,))
        all_stats[src_key] = dict(counts)
        state_set(db, "external_rejections", all_stats)
        print("Ingested:", src_key, dict(counts))
    db.commit()
    return all_stats


def apply_mass_cap(db, cfg):
    before = db.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    bounds = db.execute("SELECT min(mass_sum/n), max(mass_sum/n) FROM molecules WHERE mass_sum IS NOT NULL").fetchone()
    report = {"before": before, "applied": before > cfg.MAX_DB_ROWS}
    if before > cfg.MAX_DB_ROWS:
        if bounds[0] is None:
            raise ValueError("Cannot apply mass cap without valid train neutral masses")
        low, high = max(0.0, float(bounds[0]) - 100.0), float(bounds[1]) + 100.0
        db.execute("DELETE FROM structures WHERE mass < ? OR mass > ?", (low, high))
        db.commit()
        report.update(low=low, high=high, bound_source="train")
        print(f"DB > {cfg.MAX_DB_ROWS:,}: retaining train mass window [{low:.6f}, {high:.6f}] Da")
    report["after"] = db.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    print("Mass-cap report:", report)
    state_set(db, "mass_cap", report)
    return report


class BitOr:
    def __init__(self):
        self.value = 0

    def step(self, value):
        self.value |= int(value)

    def finalize(self):
        return self.value


def build_identity_groups(db):
    db.create_aggregate("bit_or", 1, BitOr)
    db.executescript("""
    DROP TABLE IF EXISTS grouped;
    CREATE TABLE grouped AS
    SELECT fast_key AS identity,
           fast_key AS representative,
           bit_or(source_mask) AS source_mask,
           max(ik14) AS ik14,
           NULL AS row_id
    FROM structures GROUP BY fast_key;
    CREATE UNIQUE INDEX grouped_identity ON grouped(identity);
    CREATE UNIQUE INDEX grouped_rep ON grouped(representative);
    """)
    db.commit()
    n = db.execute("SELECT COUNT(*) FROM grouped").fetchone()[0]
    print(f"Candidate pool bulk de-duplication: {n:,} rows uniformly indexed by fast_key")
    return n


def memmap_new(path, dtype, shape):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def map_read(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def build_mass_files(db, cfg, n):
    root = Path(cfg.output_dir)
    widths, fp_bytes = fp_layout()
    needed = n * (fp_bytes + 8 + 8 + 1 + 8)
    if shutil.disk_usage(root).free < needed + cfg.MIN_FREE_DISK_GB * 1e9:
        raise OSError(f"Need at least {needed/1e9:.2f} GB for arrays plus parquet/index; insufficient disk")
    mass = memmap_new(root / "mass.npy", np.float64, (n,))
    order = memmap_new(root / "mass_sorted_idx.npy", np.int64, (n,))
    schema = pa.schema([("row_id", pa.int64()), ("ik14", pa.string()), ("smiles", pa.string()),
                        ("formula", pa.string()), ("mass", pa.float64()), ("source_mask", pa.uint64()),
                        ("fast_key", pa.string()), ("identity_kind", pa.string())])
    sql = ("SELECT g.ik14,s.smiles,s.formula,s.mass,g.source_mask,s.fast_key,g.identity "
           "FROM grouped g JOIN structures s ON s.fast_key=g.representative "
           "ORDER BY s.mass,s.fast_key")
    cursor = db.execute(sql)
    start = 0
    offsets = [0]
    with pq.ParquetWriter(root / "db.parquet", schema, compression="zstd") as writer:
        for rows in tqdm(batched(cursor, cfg.BATCH_ROWS), total=math.ceil(n/cfg.BATCH_ROWS), desc="Mass-sorted export"):
            stop = start + len(rows)
            ids = np.arange(start, stop, dtype=np.int64)
            mass[start:stop] = [r[3] for r in rows]
            order[start:stop] = ids
            values = [ids, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                      mass[start:stop], [r[4] for r in rows], [r[5] for r in rows],
                      ["canonical" if r[0] else "fast_only" for r in rows]]
            writer.write_table(pa.Table.from_arrays([pa.array(v, type=f.type)
                                                     for v, f in zip(values, schema)], schema=schema))
            db.executemany("UPDATE grouped SET row_id=? WHERE identity=?",
                           [(int(i), r[6]) for i, r in zip(ids, rows)])
            db.commit()
            start = stop
            offsets.append(stop)
            mass.flush()
            order.flush()
            resource_guard(cfg)
    np.save(root / "row_group_offsets.npy", np.array(offsets, dtype=np.int64))
    del mass, order
    gc.collect()
    return widths, fp_bytes


def build_disk_indexes(db, cfg, n):
    root = Path(cfg.output_dir)
    index_path = root / "index.sqlite"
    index = open_sql(index_path)
    index.executescript("""
    DROP TABLE IF EXISTS aliases; DROP TABLE IF EXISTS formula_ranges;
    CREATE TABLE aliases(fast_key TEXT PRIMARY KEY, row_id INTEGER, source_mask INTEGER);
    CREATE INDEX aliases_row ON aliases(row_id);
    CREATE TABLE formula_ranges(formula TEXT PRIMARY KEY, start INTEGER, stop INTEGER);
    """)
    cursor = db.execute("SELECT s.fast_key,g.row_id,s.source_mask FROM structures s "
                        "JOIN grouped g ON g.identity=s.fast_key")
    for rows in batched(cursor, cfg.BATCH_ROWS):
        index.executemany("INSERT INTO aliases VALUES (?,?,?)", rows)
    formula_ids = memmap_new(root / "formula_row_ids.npy", np.int64, (n,))
    cursor = db.execute("SELECT s.formula,g.row_id FROM grouped g JOIN structures s "
                        "ON s.fast_key=g.representative ORDER BY s.formula,g.row_id")
    pos, start, previous = 0, 0, None
    offsets = [0]
    for rows in tqdm(batched(cursor, cfg.BATCH_ROWS), desc="Formula CSR index"):
        formula_ids[pos:pos+len(rows)] = [r[1] for r in rows]
        for formula, group in itertools.groupby(rows, key=lambda r: r[0]):
            count = sum(1 for _ in group)
            if previous is not None and formula != previous:
                index.execute("INSERT INTO formula_ranges VALUES (?,?,?)", (previous, start, pos))
                offsets.append(pos)
                start = pos
            previous = formula
            pos += count
        formula_ids.flush()
        resource_guard(cfg)
    if previous is not None:
        index.execute("INSERT INTO formula_ranges VALUES (?,?,?)", (previous, start, pos))
        offsets.append(pos)
    np.save(root / "formula_offsets.npy", np.array(offsets, dtype=np.int64))
    index.commit()
    index.close()
    del formula_ids
    gc.collect()


def query_masses(db):
    return (r[0] for r in db.execute("SELECT mass_sum/n FROM molecules WHERE mass_sum IS NOT NULL "
                                    "UNION ALL SELECT mass_sum/n FROM tests WHERE mass_sum IS NOT NULL"))


def mark_fp_rows(db, cfg, n):
    root = Path(cfg.output_dir)
    needed = memmap_new(root / "fp_needed.npy", np.uint8, (n,))
    mass = map_read(root / "mass.npy")
    for values in batched(query_masses(db), cfg.BATCH_ROWS):
        q = np.array(values, dtype=np.float64)
        delta = q * cfg.FP_WINDOW_PPM * 1e-6
        left = np.searchsorted(mass, q - delta)
        right = np.searchsorted(mass, q + delta, side="right")
        for lo, hi in zip(left, right):
            needed[lo:hi] = 1
        needed.flush()
    cursor = db.execute("SELECT DISTINCT g.row_id FROM grouped g JOIN structures s "
                        "ON g.identity=s.fast_key "
                        "WHERE EXISTS(SELECT 1 FROM train_structures t WHERE t.fast_key=s.fast_key)")
    for rows in batched(cursor, cfg.BATCH_ROWS):
        needed[np.array([r[0] for r in rows], dtype=np.int64)] = 1
    needed.flush()
    count = sum(int(np.count_nonzero(needed[i:i+cfg.BATCH_ROWS])) for i in range(0, n, cfg.BATCH_ROWS))
    del needed, mass
    gc.collect()
    print(f"FP selection: {count:,}/{n:,} query-window rows plus all retained train parents")
    return count


def iter_fp_rows(root, selection=None, batch_rows=8192):
    for batch in pq.ParquetFile(Path(root)/"db.parquet").iter_batches(
            batch_rows, columns=["row_id", "smiles"]):
        ids = batch.column(0).to_numpy()
        if selection is not None:
            keep = np.asarray(selection[ids], dtype=bool)
            batch = batch.filter(pa.array(keep))
            ids = ids[keep]
        yield from zip(ids.tolist(), batch.column(1).to_pylist())


def benchmark_fingerprints(cfg, pool, n, selected):
    target = min(500, cfg.BENCHMARK_ROWS) if cfg.SMOKE_TEST else cfg.BENCHMARK_ROWS
    rng = np.random.default_rng(cfg.seed)
    sample_ids = np.sort(rng.choice(n, min(n, target), replace=False)) if n else np.array([], dtype=int)
    sample = []
    file = pq.ParquetFile(Path(cfg.output_dir)/"db.parquet")
    for batch in file.iter_batches(cfg.BATCH_ROWS, columns=["row_id", "smiles"]):
        ids = batch.column(0).to_numpy()
        left = np.searchsorted(sample_ids, ids[0], side="left") if len(ids) else 0
        right = np.searchsorted(sample_ids, ids[-1], side="right") if len(ids) else 0
        if right > left:
            take = sample_ids[left:right] - ids[0]
            small = batch.take(pa.array(take))
            sample.extend(zip(small.column(0).to_pylist(), small.column(1).to_pylist()))
    start = time.perf_counter()
    for _ in pool_map(pool, fingerprint_chunk, sample, cfg, len(sample), "20k FP benchmark"):
        pass
    seconds = time.perf_counter() - start
    rate = len(sample) / max(seconds, 1e-9)
    report = {"sample_rows": len(sample), "rows_per_second": rate,
              "full_db_seconds": n/max(rate, 1e-9), "selected_seconds": selected/max(rate, 1e-9),
              "FULL_DB_FP": cfg.FULL_DB_FP, "benchmark_fps_persisted": False}
    print("FP projection:", report)
    return report


def write_fingerprints(cfg, pool, n, width):
    root = Path(cfg.output_dir)
    valid_path, fp_path = root/"fp_valid.npy", root/"fp_packed.npy"
    if valid_path.exists() and fp_path.exists():
        fp = np.load(fp_path, mmap_mode="r+", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r+", allow_pickle=False)
        if fp.shape != (n, width) or valid.shape != (n,):
            raise ValueError("Incompatible fingerprint checkpoint; use a fresh output_dir")
    else:
        fp = memmap_new(fp_path, np.uint8, (n, width))
        valid = memmap_new(valid_path, np.uint8, (n,))
    needed = map_read(root/"fp_needed.npy")
    def remaining():
        for row_id, smiles in iter_fp_rows(root, None if cfg.FULL_DB_FP else needed, cfg.BATCH_ROWS):
            if not valid[row_id]:
                yield row_id, smiles
    for result in pool_map(pool, fingerprint_chunk, remaining(), cfg, desc="Packed fingerprints"):
        ids = np.array([r[0] for r in result], dtype=np.int64)
        fp[ids] = np.stack([r[1] for r in result])
        fp.flush()
        valid[ids] = 1
        valid.flush()
    written = sum(int(np.count_nonzero(valid[i:i+cfg.BATCH_ROWS])) for i in range(0, n, cfg.BATCH_ROWS))
    del fp, valid, needed
    gc.collect()
    print(f"Stored {written:,}/{n:,} fingerprints; missing entries computed lazily at query time")
    return written


def configure(db_dir=None, exclusion_scope="train", cache_dir=None, row_group_cache=2):
    global CTX
    if exclusion_scope not in ("train", "all"):
        raise ValueError("exclusion_scope must be 'train' or 'all'")
    if CTX:
        CTX["index"].close()
        CTX["cache"].close()
    root = Path(db_dir or WORK/"artifacts"/"candidates").resolve()
    metadata = json.loads((root/"db_metadata.json").read_text())
    if metadata["fp_spec"] != load_evalkit().FP_SPEC:
        raise ValueError("FP_SPEC differs from database fingerprint schema")
    cache_root = Path(cache_dir or WORK/"candidate_cache"/metadata["build_id"])
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = open_sql(cache_root/"lazy.sqlite")
    cache.execute("CREATE TABLE IF NOT EXISTS chemistry(row_id INTEGER PRIMARY KEY, ik14 TEXT, fp BLOB)")
    CTX = {"root": root, "meta": metadata, "mass": map_read(root/"mass.npy"),
           "order": map_read(root/"mass_sorted_idx.npy"), "formula_ids": map_read(root/"formula_row_ids.npy"),
           "offsets": map_read(root/"row_group_offsets.npy"), "fp": map_read(root/"fp_packed.npy"),
           "valid": map_read(root/"fp_valid.npy"), "index": open_sql(root/"index.sqlite", True),
           "parquet": pq.ParquetFile(root/"db.parquet"), "cache": cache,
           "scope": exclusion_scope, "groups": OrderedDict(), "max_groups": max(1, row_group_cache)}
    return metadata


def context():
    if CTX is None:
        configure()
    return CTX


def fetch_rows(ids):
    state = context()
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0:
        return pd.DataFrame(columns=DB_COLUMNS)
    if (ids < 0).any() or (ids >= len(state["mass"])).any():
        raise IndexError("candidate row_id outside database")
    if len(ids) > state["meta"]["max_materialized_rows"]:
        raise MemoryError("Too many rows to materialize; use iter_by_mass / iter_by_formula")
    groups = np.searchsorted(state["offsets"], ids, side="right") - 1
    chunks = []
    for group in np.unique(groups):
        cache = state["groups"]
        if int(group) not in cache:
            cache[int(group)] = state["parquet"].read_row_group(int(group), columns=DB_COLUMNS)
        cache.move_to_end(int(group))
        while len(cache) > state["max_groups"]:
            cache.popitem(last=False)
        positions = np.flatnonzero(groups == group)
        offsets = ids[positions] - state["offsets"][group]
        frame = cache[int(group)].take(pa.array(offsets)).to_pandas()
        frame["_position"] = positions
        chunks.append(frame)
    return pd.concat(chunks).sort_values("_position").drop(columns="_position").reset_index(drop=True)


def excluded_rows(exclude_keys):
    if not exclude_keys:
        return set()
    result = set()
    for keys in batched(sorted(exclude_keys), 500):
        placeholders = ",".join("?" for _ in keys)
        rows = context()["index"].execute(
            f"SELECT row_id FROM aliases WHERE fast_key IN ({placeholders})", keys)
        result.update(r[0] for r in rows)
    return result


def filter_sources(frame, exclude_keys=None, external_only=False, excluded_ids=None):
    if frame.empty:
        return frame
    state = context()
    rejected = excluded_rows(exclude_keys) if excluded_ids is None else excluded_ids
    mask = frame.source_mask.to_numpy(dtype=np.uint64).copy()
    selected = frame.row_id.isin(rejected).to_numpy()
    if state["scope"] == "all":
        mask[selected] = 0
    else:
        mask[selected] &= np.uint64(0xFFFFFFFFFFFFFFFE)
    if external_only:
        mask &= np.uint64(0xFFFFFFFFFFFFFFFE)
    frame = frame.copy()
    frame["source_mask"] = mask
    return frame.loc[mask != 0].reset_index(drop=True)


def mass_bounds(neutral_mass, ppm):
    if not np.isfinite(neutral_mass) or neutral_mass <= 0 or not np.isfinite(ppm) or ppm < 0:
        raise ValueError("neutral_mass must be positive/finite; ppm nonnegative/finite")
    delta = float(neutral_mass) * float(ppm) * 1e-6
    mass = context()["mass"]
    return int(np.searchsorted(mass, neutral_mass - delta)), int(np.searchsorted(mass, neutral_mass + delta, side="right"))


def iter_by_mass(neutral_mass, ppm, exclude_keys=None, batch_rows=8192):
    lo, hi = mass_bounds(neutral_mass, ppm)
    blocked = excluded_rows(exclude_keys)
    for start in range(lo, hi, batch_rows):
        ids = context()["order"][start:min(hi, start + batch_rows)]
        yield filter_sources(fetch_rows(ids), excluded_ids=blocked)


def query_by_mass(neutral_mass, ppm, exclude_keys=None):
    lo, hi = mass_bounds(neutral_mass, ppm)
    return filter_sources(fetch_rows(context()["order"][lo:hi]), exclude_keys)


def hill_formula(formula):
    if not isinstance(formula, str) or not formula:
        raise ValueError("Expected a nonempty neutral molecular formula")
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if "".join(a+b for a, b in parts) != formula:
        raise ValueError("Formula must be an uncharged, non-isotopic elemental formula")
    counts = Counter()
    for symbol, number in parts:
        count = int(number or 1)
        if count <= 0 or symbol not in context()["meta"]["allowed_elements"]:
            raise ValueError(f"Invalid formula element/count: {symbol}{number}")
        counts[symbol] += count
    keys = ([s for s in ("C", "H") if s in counts] + sorted(set(counts) - {"C", "H"})
            if "C" in counts else sorted(counts))
    return "".join(k + (str(counts[k]) if counts[k] != 1 else "") for k in keys)


def formula_span(formula):
    row = context()["index"].execute("SELECT start,stop FROM formula_ranges WHERE formula=?",
                                     (hill_formula(formula),)).fetchone()
    return row if row else (0, 0)


def iter_by_formula(formula, exclude_keys=None, batch_rows=8192):
    lo, hi = formula_span(formula)
    blocked = excluded_rows(exclude_keys)
    for start in range(lo, hi, batch_rows):
        ids = context()["formula_ids"][start:min(hi, start + batch_rows)]
        yield filter_sources(fetch_rows(ids), excluded_ids=blocked)


def query_by_formula(formula, exclude_keys=None):
    lo, hi = formula_span(formula)
    return filter_sources(fetch_rows(context()["formula_ids"][lo:hi]), exclude_keys)


def get_fingerprints(row_ids):
    state = context()
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(ids) == 0:
        return np.empty((0, state["meta"]["fp_bytes"]), dtype=np.uint8)
    if len(ids) > state["meta"]["max_materialized_rows"]:
        raise MemoryError("Request fingerprints in bounded batches")
    if (ids < 0).any() or (ids >= len(state["mass"])).any():
        raise IndexError("candidate row_id outside database")
    out = np.empty((len(ids), state["meta"]["fp_bytes"]), dtype=np.uint8)
    ready = np.asarray(state["valid"][ids], dtype=bool)
    out[ready] = state["fp"][ids[ready]]
    missing = np.flatnonzero(~ready)
    if len(missing) > 0:
        missing_ids = ids[missing]
        for position, row_id in zip(missing, missing_ids):
            row_id = int(row_id)
            cached = state["cache"].execute("SELECT fp FROM chemistry WHERE row_id=?", (row_id,)).fetchone()
            if cached and cached[0] is not None:
                out[position] = np.frombuffer(cached[0], dtype=np.uint8)
            else:
                smiles = fetch_rows([row_id]).smiles.iloc[0]
                packed = fingerprint_chunk([(row_id, smiles)])[0][1]
                out[position] = packed
                state["cache"].execute(
                    "INSERT INTO chemistry(row_id,fp) VALUES (?,?) "
                    "ON CONFLICT(row_id) DO UPDATE SET fp=excluded.fp",
                    (row_id, packed.tobytes())
                )
        state["cache"].commit()
    return out


def canonical_key(row_id, smiles, known=None):
    if isinstance(known, str) and known:
        return known
    state = context()
    row = state["cache"].execute("SELECT ik14 FROM chemistry WHERE row_id=?", (int(row_id),)).fetchone()
    if row and row[0]:
        return row[0]
    key = load_evalkit().canon_ik14(Chem.MolFromSmiles(smiles))
    if not key:
        raise ValueError(f"evalkit.canon_ik14 failed at row {row_id}")
    state["cache"].execute("INSERT INTO chemistry(row_id,ik14) VALUES (?,?) "
                           "ON CONFLICT(row_id) DO UPDATE SET ik14=excluded.ik14", (int(row_id), key))
    return key


def finalize_candidates(ranked, top_n=256, limit=25):
    if top_n < 1 or limit < 1:
        raise ValueError("top_n and limit must be positive")
    shortlist = ranked.head(top_n).copy()
    merged = OrderedDict()
    for row in shortlist.to_dict("records"):
        key = canonical_key(row["row_id"], row["smiles"], row.get("ik14"))
        if key not in merged:
            row.update(ik14=key, identity_kind="canonical", member_row_ids=[int(row["row_id"])])
            merged[key] = row
        else:
            merged[key]["source_mask"] |= int(row["source_mask"])
            merged[key]["member_row_ids"].append(int(row["row_id"]))
    context()["cache"].commit()
    return pd.DataFrame(list(merged.values())[:limit], columns=list(shortlist.columns) + ["member_row_ids"])


def external_coverage(db):
    denominator = db.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
    valid = db.execute("SELECT COUNT(*) FROM molecules m JOIN train_structures t ON m.smiles=t.raw "
                       "WHERE t.fast_key IS NOT NULL").fetchone()[0]
    fast = db.execute("SELECT COUNT(*) FROM molecules m JOIN train_structures t ON m.smiles=t.raw "
                      "WHERE EXISTS(SELECT 1 FROM structures s WHERE s.fast_key=t.fast_key "
                      "AND (s.source_mask & ~1)!=0)").fetchone()[0]
    result = {"train_molecules": denominator, "valid_train_molecules": valid,
              "external_fast_key_hits": fast,
              "external_coverage_percent": 100 * fast / max(denominator, 1),
              "canonical_coverage_is_lower_bound": True,
              "denominator_includes_rejected_train": True}
    print("External DB coverage:", result)
    return result


def validation_exclusions(db):
    return {r[0] for r in db.execute("SELECT DISTINCT t.fast_key FROM molecules m "
                                     "JOIN train_structures t ON m.smiles=t.raw "
                                     "WHERE m.split3!='train' AND t.fast_key IS NOT NULL")}


def scan_oracle_query(mass, ppm, truth, true_row_ids, blocked, external_only, top_n=256):
    count, present, unknown, shortlist = 0, False, 0, []
    for frame in iter_by_mass(mass, ppm):
        frame = filter_sources(frame, external_only=external_only, excluded_ids=blocked)
        count += len(frame)
        if truth:
            present |= bool(frame.row_id.isin(true_row_ids).any() or frame.ik14.eq(truth).any())
        pending = frame.loc[frame.ik14.isna()]
        unknown += len(pending)
        if truth and not present:
            errors = np.abs(pending.mass.to_numpy(dtype=float) - mass)
            for offset in np.argsort(errors, kind="stable")[:top_n]:
                row = pending.iloc[offset]
                item = (-float(errors[offset]), -int(row.row_id), str(row.smiles))
                if len(shortlist) < top_n:
                    heapq.heappush(shortlist, item)
                elif item > shortlist[0]:
                    heapq.heapreplace(shortlist, item)
    checked = 0
    if truth and not present:
        for _, negative_id, smiles in sorted(shortlist, reverse=True):
            checked += 1
            if canonical_key(-negative_id, smiles) == truth:
                present = True
                break
    unresolved = bool(truth and not present and unknown > checked)
    context()["cache"].commit()
    return count, int(present), int(present or unresolved), int(unresolved)


def report_coverage(db, cfg):
    root = Path(cfg.output_dir)
    blocked = excluded_rows(validation_exclusions(db))
    modes = [("all_sources_unexcluded", "train", False, set()),
             ("heldout_removed_train", "train", False, blocked),
             ("heldout_removed_all", "all", False, blocked),
             ("external_only", "train", True, set())]
    rows = []
    query_sql = ("SELECT m.id,m.split3,m.mass_sum/m.n,m.ik14,t.fast_key FROM molecules m "
                 "LEFT JOIN train_structures t ON t.raw=m.smiles WHERE m.split3!='train' "
                 "UNION ALL SELECT id,'test',mass_sum/n,NULL,NULL FROM tests")
    total = db.execute("SELECT COUNT(*) FROM molecules WHERE split3!='train'").fetchone()[0]
    total += db.execute("SELECT COUNT(*) FROM tests").fetchone()[0]
    schema = pa.schema([("molecule_id", pa.string()), ("split", pa.string()), ("configuration", pa.string()),
                        ("ppm", pa.int32()), ("candidates", pa.int64()), ("oracle_lower", pa.float64()),
                        ("oracle_upper", pa.float64()), ("unresolved", pa.int64())])
    with pq.ParquetWriter(root/"query_coverage.parquet", schema, compression="zstd") as writer:
        for mol_id, split, mass, truth, key in tqdm(db.execute(query_sql), total=total, desc="Molecule-level oracle audit"):
            true_ids = excluded_rows({key}) if key else set()
            for name, scope, external, excluded in modes:
                context()["scope"] = scope
                for ppm in cfg.REPORT_PPM:
                    count, lower, upper, unresolved = scan_oracle_query(
                        mass, ppm, truth, true_ids, excluded, external, cfg.ORACLE_CANON_TOP_N)
                    rows.append((mol_id, split, name, ppm, count, lower if split != "test" else None,
                                 upper if split != "test" else None, unresolved))
            if len(rows) >= cfg.BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, r)) for r in rows], schema=schema))
                rows.clear()
                resource_guard(cfg)
        if rows:
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, r)) for r in rows], schema=schema))
    context()["scope"] = cfg.EXCLUSION_SCOPE
    summaries = aggregate_query_reports(root/"query_coverage.parquet", db, cfg)
    return {"external": external_coverage(db), "queries": summaries,
            "identity_note": "fast-key pool sizes; top-256 tautomer audit gives oracle interval",
            "leaky_diagnostic": "all_sources_unexcluded includes held-out train-file structures",
            "actual_db_has_train_source": not cfg.EXCLUDE_TRAIN_FROM_DB}


def aggregate_query_reports(path, db, cfg):
    db.execute("DROP TABLE IF EXISTS report_counts")
    db.execute("CREATE TABLE report_counts(split TEXT,mode TEXT,ppm INT,n INT,lo REAL,hi REAL,u INT)")
    for batch in pq.ParquetFile(path).iter_batches(cfg.BATCH_ROWS):
        frame = batch.to_pandas().drop(columns="molecule_id")
        db.executemany("INSERT INTO report_counts VALUES (?,?,?,?,?,?,?)", frame.itertuples(index=False, name=None))
    db.execute("CREATE INDEX report_group ON report_counts(split,mode,ppm,n)")
    result = []
    groups = db.execute("SELECT DISTINCT split,mode,ppm FROM report_counts").fetchall()
    for split, mode, ppm in groups:
        args = (split, mode, ppm)
        where = " WHERE split=? AND mode=? AND ppm=?"
        n, mean, maximum, empty, lo, hi, unresolved = db.execute(
            "SELECT COUNT(*),AVG(n),MAX(n),AVG(n=0),AVG(lo),AVG(hi),SUM(u) FROM report_counts" + where, args).fetchone()
        row = dict(split=split, configuration=mode, ppm=ppm, queries=n, count_mean=mean, count_max=maximum, empty_fraction=empty)
        for label, q in (("count_median", 0.5), ("count_p90", 0.9), ("count_p99", 0.99)):
            pos = (n - 1) * q
            i, j = math.floor(pos), math.ceil(pos)
            values = db.execute("SELECT n FROM report_counts" + where + " ORDER BY n LIMIT ? OFFSET ?",
                                args + (j - i + 1, i)).fetchall()
            row[label] = values[0][0] + (values[-1][0] - values[0][0]) * (pos - i)
        if split != "test":
            row.update(oracle_mrr25_lower_bound=lo, oracle_mrr25_upper_bound=hi, unresolved_queries=unresolved)
        result.append(row)
    db.commit()
    return result


def id_hash(db, split):
    digest = hashlib.sha256()
    for row in db.execute("SELECT id FROM molecules WHERE split3=? ORDER BY id", (split,)):
        digest.update((row[0] + "\n").encode())
    return digest.hexdigest()


def create_synthetic_inputs(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC", "CCCO", "CCOC", "CCS"]
    roles = ["train", "train", "train", "train", "val_model", "val_model", "val_rank", "val_rank"]
    train, splits = [], []
    for i, (smi, role) in enumerate(zip(smiles, roles)):
        mol = Chem.MolFromSmiles(smi)
        mass = float(rdMolDescriptors.CalcExactMolWt(mol))
        ik14 = Chem.MolToInchiKey(mol)[:14]
        for _ in range(60):
            train.append(dict(molecule_id=f"smoke_{i}", smiles=smi, neutral_mass=mass,
                              precursor_mz=mass+1.007276, adduct="[M+H]+", polarity="+"))
        splits.append(dict(molecule_id=f"smoke_{i}", split3=role, ik14=ik14))
    tests = [dict(molecule_id="test_a", neutral_mass=60.057514876),
             dict(molecule_id="test_b", neutral_mass=46.041864812)]
    for name, rows in (("train_clean", train), ("splits", splits), ("test_clean", tests)):
        pq.write_table(pa.Table.from_pylist(rows), root/f"{name}.parquet")
    print(f"Synthetic smoke inputs: {len(train)} train spectra; molecule counts={len(splits)}/2")
    return root


def validate_artifacts(cfg):
    state = context()
    n = state["meta"]["rows"]
    file = state["parquet"]
    assert file.metadata.num_rows == n
    assert set(DB_COLUMNS).issubset(file.schema_arrow.names)
    assert state["fp"].dtype == np.uint8 and state["fp"].shape == (n, state["meta"]["fp_bytes"])
    assert len(state["mass"]) == len(state["order"]) == len(state["formula_ids"]) == n
    previous = -np.inf
    known = 0
    for i in range(0, n, cfg.BATCH_ROWS):
        masses = state["mass"][i:i+cfg.BATCH_ROWS]
        assert masses[0] >= previous and np.all(np.diff(masses) >= 0)
        previous = masses[-1]
        assert np.array_equal(state["order"][i:i+len(masses)], np.arange(i, i+len(masses)))
        known += int(np.count_nonzero(state["valid"][i:i+len(masses)]))
    assert state["offsets"][0] == 0 and state["offsets"][-1] == n
    aliases = state["index"].execute("SELECT COUNT(*) FROM aliases WHERE row_id<0 OR row_id>=?", (n,)).fetchone()[0]
    assert aliases == 0
    if n:
        sample = fetch_rows([0]).iloc[0]
        assert 0 in set(query_by_mass(float(sample.mass), 0).row_id)
        assert 0 in set(query_by_formula(sample.formula).row_id)
        assert get_fingerprints([0]).shape == (1, state["meta"]["fp_bytes"])
    for name in state["meta"]["required_files"]:
        assert (state["root"]/name).is_file(), name
    print(f"Sanity checks passed: {n:,} rows, {known:,} materialized fingerprints")
    return {"rows": n, "fp_materialized": known, "sanity": "passed"}
'''


NOTEBOOK_CELLS = []


def md(source):
    NOTEBOOK_CELLS.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    NOTEBOOK_CELLS.append(nbf.v4.new_code_cell(source.strip()))


def add_module_cells(source):
    tree = ast.parse(source)
    lines = source.strip().splitlines()
    tree = ast.parse("\n".join(lines))
    chunks, current = [], []
    previous = 0
    for node in tree.body:
        segment = lines[previous:node.end_lineno]
        previous = node.end_lineno
        if len(current) + len(segment) > 57:
            if current:
                chunks.append("\n".join(current).strip())
            current = segment
        else:
            current += segment
    if current:
        chunks.append("\n".join(current).strip())
    for i, chunk in enumerate(chunks):
        if len(chunk.splitlines()) > 57:
            raise ValueError(f"Module block exceeds cell limit: {chunk[:100]}")
        ast.parse(chunk)
        magic = "%%writefile /kaggle/working/candidates.py" if i == 0 else "%%writefile -a /kaggle/working/candidates.py"
        code(magic + "\n" + chunk + "\n\n")


def build_notebook(output):
    NOTEBOOK_CELLS.clear()
    md(r"""
# 04 — Indexed Candidate Structure Database

Build a formula- and neutral-mass-indexed structure database from cleaned train
SMILES and attached external databases (COCONUT, LOTUS, NPAtlas, ChEBI, HMDB).
No runtime downloads. Visible test structures are never used to drop valid candidates.

**Inputs:** `train_clean.parquet`, `test_clean.parquet`, `splits.parquet` via `find_file`;
Stage 02's `evalkit.py` must already exist at `/kaggle/working/evalkit.py`.
Auto-discovery supports uncompressed files (.tsv, .csv, .smi, .sdf, .parquet) and
gzip-compressed streams (.tsv.gz, .csv.gz, .sdf.gz) as well as zip archives (.zip).

**Outputs:** `/kaggle/working/artifacts/candidates/`: `db.parquet`, `fp_packed.npy`,
`mass_sorted_idx.npy`, `db_metadata.json`, `coverage.json`, `manifest.json`, and `candidates.py`.
The importable module is written via `%%writefile /kaggle/working/candidates.py`.

**Resources:** CPU only, 4 worker processes, peak RAM < 7.5 GB, zero GPU memory.
SQLite external sorting, bounded Arrow batches and memory-mapped arrays support 10M+ rows.
4,096-bit Morgan + padded 167-bit MACCS use 533 bytes/row. Pre-computed for queries (±10 ppm)
plus all train molecules; missing entries are computed lazily into a disk-backed SQLite cache.

**Identity Contract & Tautomer Strategy:** Candidate pools uniformly use
`evalkit.fast_key` for indexing and bulk deduplication. Tautomer canonicalization is deferred:
known train canonical `ik14` is preserved; `finalize_candidates(ranked, top_n=256, limit=25)`
runs `canon_ik14` only on the top-256 ranked candidates per query.
""")
    code(r"""
from dataclasses import dataclass, asdict
from pathlib import Path
import os, time
DATA_DIR = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"

@dataclass
class CFG:
    stage: str = "candidates"
    seed: int = 42
    SMOKE_TEST: bool = False
    data_dir: str = DATA_DIR
    input_root: str = "/kaggle/input"
    output_dir: str = "/kaggle/working/artifacts/candidates"
    FULL_DB_FP: bool = False
    EXCLUDE_TRAIN_FROM_DB: bool = False
    EXCLUSION_SCOPE: str = "train"
    FP_WINDOW_PPM: float = 10.0
    REPORT_PPM: tuple = (5, 10)
    ORACLE_CANON_TOP_N: int = 256
    BENCHMARK_ROWS: int = 20_000
    MAX_DB_ROWS: int = 10_000_000
    WORKERS: int = 4
    CHUNK_ROWS: int = 256
    BATCH_ROWS: int = 8192
    RAM_GB: float = 7.2
    MIN_FREE_DISK_GB: float = 0.75
    MAX_MATERIALIZED_ROWS: int = 100_000
    time_budget_s: float = 28_800
    RESUME: bool = True

cfg = CFG()
if cfg.SMOKE_TEST:
    cfg.output_dir += "_smoke"
assert cfg.WORKERS == 4 and 0 < cfg.RAM_GB < 8
assert cfg.FP_WINDOW_PPM >= max(cfg.REPORT_PPM)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
T_START = time.perf_counter()
print(asdict(cfg))
""")
    code(r"""
import csv, gc, hashlib, importlib, json, multiprocessing as mp, random, sys
from datetime import datetime, timezone
_max_csv = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_csv)
        break
    except OverflowError:
        _max_csv = int(_max_csv / 10)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from tqdm.auto import tqdm
try:
    from rdkit import Chem, rdBase
except ImportError as exc:
    raise ImportError("RDKit is mandatory; attach an offline environment with RDKit") from exc
import torch
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
print(f"Python {sys.version.split()[0]} | RDKit {rdBase.rdkitVersion} | Arrow {pa.__version__}")
print(f"NumPy {np.__version__} | pandas {pd.__version__} | torch {torch.__version__}")
print(f"RAM available: {psutil.virtual_memory().available/1e9:.2f} GB; CPU-only stage")
print(f"GPUs available: {torch.cuda.device_count()}; allocated GPU memory: 0 GB")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, f"{props.total_memory/1e9:.2f} GB")
""")
    md(r"""
## Utilities and Reusable Module

Write `candidates.py` to `/kaggle/working/candidates.py` via `%%writefile`.
Standardize molecules via RDKit: retain largest fragment, neutralize formal charges,
strip isotopes, remove stereochemistry, sanitize, drop heavy atoms > 150, drop disallowed
elements outside the bio-organic and natural product set (C, H, N, O, S, P, halogens, bio-metals, and trace metalloids).
Bulk de-duplication across external sources uses `evalkit.fast_key`.
""")
    add_module_cells(MODULE_SOURCE)
    code(r"""
import importlib.util
WORK = Path("/kaggle/working")
sys.path.insert(0, str(WORK))
module_path = WORK / "candidates.py"
spec = importlib.util.spec_from_file_location("candidates", module_path)
candidates = importlib.util.module_from_spec(spec)
sys.modules["candidates"] = candidates
spec.loader.exec_module(candidates)
assert Path(candidates.__file__).resolve() == module_path.resolve()
evalkit = candidates.load_evalkit()
print("evalkit:", evalkit.__file__)
print("FP_SPEC:", evalkit.FP_SPEC)
find_file, find_manifest = candidates.find_file, candidates.find_manifest
SchemaAdapter = candidates.SchemaAdapter
mrr_at_25 = evalkit.mrr_at_25
print("Module source loaded; no attached Python files were imported.")
""")
    md(r"""
## Inputs, Schema Checks and External Discovery

Locate core files via recursive discovery `find_file`: `train_clean.parquet`,
`test_clean.parquet`, `splits.parquet`.
Auto-discover attached external datasets under `/kaggle/input/` matching raw (.smi, .tsv, .csv, .sdf, .parquet)
or gzip-compressed streams (.tsv.gz, .csv.gz, .sdf.gz) and zip archives (.zip).
""")
    code(r"""
if cfg.SMOKE_TEST:
    smoke_root = candidates.create_synthetic_inputs(Path(cfg.output_dir)/"synthetic_inputs")
    roots = [smoke_root]
else:
    roots = None
core = {name: find_file(name, hint="eda" if "clean" in name else "eval_splits", roots=roots)
        for name in ("train_clean.parquet", "test_clean.parquet", "splits.parquet")}
schema_path = None if cfg.SMOKE_TEST else find_file("schema.json", hint="eda", required=False)
schema = json.loads(schema_path.read_text()) if schema_path else None
sources = [] if cfg.SMOKE_TEST else candidates.discover_sources(cfg.input_root, core.values())
if not sources:
    print("No external structure databases found. Building DB from train SMILES only.")
if not sources and cfg.EXCLUDE_TRAIN_FROM_DB:
    print("EXCLUDE_TRAIN_FROM_DB=True: no sources remain; a valid empty DB will be saved.")
""")
    code(r"""
signature = {"cfg": asdict(cfg), "core": {}, "sources": sources,
             "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
             "evalkit_sha256": hashlib.sha256(Path(evalkit.__file__).read_bytes()).hexdigest()}
for name, path in core.items():
    stat = path.stat()
    signature["core"][name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
build_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
db = candidates.initialize_stage(cfg)
previous = candidates.state_get(db, "build_id")
if previous and (not cfg.RESUME or previous != build_id):
    raise ValueError("Output contains a different build or RESUME=False; choose a fresh output_dir")
candidates.state_set(db, "build_id", build_id)
candidates.ingest_metadata(db, core["train_clean.parquet"], "train", cfg, schema)
candidates.ingest_metadata(db, core["test_clean.parquet"], "test", cfg, schema)
candidates.ingest_splits(db, core["splits.parquet"], cfg)
for table in ("molecules", "tests"):
    rows, spread = db.execute(f"SELECT COUNT(*),MAX((hi-lo)/(mass_sum/n)*1e6) FROM {table}").fetchone()
    print(f"{table}: {rows:,} molecules; largest observed-mass spread={spread} ppm")
print("Split sizes:", db.execute("SELECT split3,COUNT(*) FROM molecules GROUP BY split3").fetchall())
""")
    md(r"""
## Benchmark, Standardize, and Deduplicate

Measure bulk standardization throughput on 20k rows without tautomer enumeration.
Train and external structures are standardized and deduplicated by `evalkit.fast_key`.
Candidate filtering bounds are derived strictly from the TRAIN neutral mass range:
[min_train_mass - 100, max_train_mass + 100] Da, preventing hidden-test mass purges.
""")
    code(r"""
pool = mp.get_context("spawn").Pool(cfg.WORKERS, initializer=candidates.init_worker,
                                    maxtasksperchild=128)
chem_projection = candidates.benchmark_standardization(
    sources, db, pool, sorted(candidates.BASE_ELEMENTS | candidates.EXTRA_ELEMENTS), cfg)
allowed = candidates.prepare_train(db, pool, cfg)
external_stats = candidates.ingest_external(db, sources, pool, allowed, cfg)
mass_cap = candidates.apply_mass_cap(db, cfg)
if not candidates.state_get(db, "indexes_done"):
    n_rows = candidates.build_identity_groups(db)
    widths, fp_bytes = candidates.build_mass_files(db, cfg, n_rows)
    candidates.build_disk_indexes(db, cfg, n_rows)
    candidates.state_set(db, "n_rows", n_rows)
    candidates.state_set(db, "indexes_done", True)
else:
    n_rows = candidates.state_get(db, "n_rows")
    widths, fp_bytes = candidates.fp_layout()
    print(f"Resume: {n_rows:,} indexed rows")
print(f"Final DB: {n_rows:,} rows; {fp_bytes} packed bytes/row; {n_rows*fp_bytes/1e9:.2f} GB logical FP size")
""")
    md(r"""
## Selective Packed Fingerprints and Runtime Projection

`fp_needed.npy` marks rows inside the query mass window (±10 ppm) plus all train molecules.
Fingerprints follow `FP_SPEC`: binary Morgan r=2 (4,096 bits) + MACCS (167 bits),
packed into 533 uint8 bytes per structure. Rows missing at build time are populated
lazily via `get_fingerprints(row_ids)` into a disk-backed SQLite cache.
""")
    code(r"""
selected = candidates.mark_fp_rows(db, cfg, n_rows)
fp_projection = candidates.benchmark_fingerprints(cfg, pool, n_rows, selected)
fingerprints_written = candidates.write_fingerprints(cfg, pool, n_rows, fp_bytes)
pool.close()
pool.join()
del pool
gc.collect()
torch.cuda.empty_cache()
print(f"Build RSS: {candidates.resource_guard(cfg)/1e9:.3f} GB; elapsed {time.perf_counter()-T_START:.1f}s")
""")
    md(r"""
## Retrieval API and Validation Exclusions

`query_by_mass` searches inclusive neutral mass windows in Da/ppm.
`query_by_formula` searches exact Hill formula ranges.
`finalize_candidates(ranked, top_n=256, limit=25)` computes `canon_ik14` on the top-256
ranked candidates per query for tautomer-canonical deduplication.
Missing fingerprints are lazily computed and cached on demand.
""")
    code(r"""
root = Path(cfg.output_dir)
required_files = ["db.parquet", "fp_packed.npy", "fp_valid.npy", "fp_needed.npy", "mass.npy",
                  "mass_sorted_idx.npy", "row_group_offsets.npy", "formula_row_ids.npy",
                  "formula_offsets.npy", "index.sqlite", "candidates.py"]
source_flags = {"train": 1, **{s["dataset"]: s["source_mask"] for s in sources}}
metadata = {"rows": n_rows, "build_id": build_id, "fp_spec": evalkit.FP_SPEC,
            "fp_bytes": fp_bytes, "component_bits": widths, "bitorder": "little",
            "allowed_elements": allowed, "source_flags": source_flags,
            "row_order": "ascending exact neutral mass, then representative fast_key",
            "ik14": "uniform fast_key for candidate pools; canon_ik14 computed on top-256 per query",
            "exclusion_default": cfg.EXCLUSION_SCOPE, "mass_cap": mass_cap,
            "max_materialized_rows": cfg.MAX_MATERIALIZED_ROWS, "required_files": required_files}
(root/"db_metadata.json").write_text(json.dumps(metadata, indent=2))
(root/"candidates.py").write_text(module_path.read_text(), encoding="utf-8")
candidates.configure(cfg.output_dir, exclusion_scope=cfg.EXCLUSION_SCOPE)
query_by_mass = candidates.query_by_mass
query_by_formula = candidates.query_by_formula
exclude_keys = candidates.validation_exclusions(db)
if n_rows:
    example = candidates.fetch_rows([0]).iloc[0]
    hits = query_by_mass(float(example.mass), 10, exclude_keys=exclude_keys)
    print("Example mass retrieval:", hits.shape)
    display(hits.head())
    print("Example formula retrieval:", query_by_formula(example.formula, exclude_keys).shape)
""")
    md(r"""
## Coverage, Candidate Counts, and Oracle MRR@25 Ceiling

Audit candidate pool coverage by molecule_id at 5/10 ppm across `val_model`, `val_rank`,
and test under four isolation configurations.
Oracle MRR@25 ceiling verifies that the true structure is present within the mass window.
""")
    code(r"""
metrics = candidates.report_coverage(db, cfg)
display(pd.DataFrame(metrics["queries"]))
print("External coverage:", metrics["external"])
print("Oracle estimate uses confirmed-hit lower bound and unresolved tautomer upper bound.")
(root/"coverage.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
""")
    md(r"""
## Save Artifacts, Manifest, and Sanity Checks

Verify and save all candidate database artifacts to `/kaggle/working/artifacts/candidates/`:
`db.parquet`, `fp_packed.npy`, `mass_sorted_idx.npy`, `db_metadata.json`, `coverage.json`, `manifest.json`.
""")
    code(r"""
elapsed = time.perf_counter() - T_START
heldout_hashes = {s: candidates.id_hash(db, s) for s in ("val_model", "val_rank")}
sanity = candidates.validate_artifacts(cfg)
(root/"config.json").write_text(json.dumps(asdict(cfg), indent=2))
manifest = {"stage": cfg.stage, "created": datetime.now(timezone.utc).isoformat(),
            "cfg": asdict(cfg), "lib_versions": {"python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion, "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": pa.__version__, "torch": torch.__version__},
            "metrics": metrics, "trained_on_split": None, "training_note": "No fitted model",
            "heldout_ids_hash": heldout_hashes, "fp_spec": evalkit.FP_SPEC,
            "source_flags": source_flags, "discovered_sources": sources,
            "train_chemistry": candidates.state_get(db, "train_rejections"),
            "external_chemistry": external_stats, "mass_cap": mass_cap,
            "projections": {"standardization": chem_projection, "fingerprints": fp_projection},
            "runtime_seconds": elapsed, "build_id": build_id, "sanity": sanity,
            "identity_contract": metadata["ik14"], "inputs": signature["core"]}
db.commit()
db.close()
manifest["files"] = {p.name: p.stat().st_size for p in sorted(root.iterdir()) if p.is_file()}
(root/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
assert json.loads((root/"manifest.json").read_text())["heldout_ids_hash"] == heldout_hashes
assert pq.ParquetFile(root/"query_coverage.parquet").schema_arrow.names[0] == "molecule_id"
print(f"Saved to {root}; elapsed={elapsed/60:.1f} min; files={len(manifest['files'])}")
if elapsed > cfg.time_budget_s:
    print("Soft time budget exceeded; algorithm and database scope retained.")
""")
    md(r"""
## Complete Generated Source

The module `/kaggle/working/candidates.py` is printed below for reuse in downstream stages.
""")
    code('print(Path("/kaggle/working/candidates.py").read_text(encoding="utf-8"))')
    nb = nbf.v4.new_notebook(cells=NOTEBOOK_CELLS)
    nb.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                       language_info={"name": "python", "version": "3.10.0"})
    nbf.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert len(cell.source.splitlines()) < 60, (i, len(cell.source.splitlines()))
            source = cell.source.split("\n", 1)[1] if cell.source.startswith("%%writefile") else cell.source
            ast.parse(source, filename=f"cell_{i}")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    delivery = Path("/mnt/user-data/outputs")
    if delivery.is_dir() and delivery.resolve() != output.parent.resolve():
        nbf.write(nb, delivery / output.name)
    print(f"Created {output}: {len(nb.cells)} cells; nbformat + per-cell AST validation passed")
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("04_candidate_db.ipynb"))
    args = parser.parse_args()
    build_notebook(args.output)
