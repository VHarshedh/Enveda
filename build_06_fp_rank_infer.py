"""Build the self-contained Kaggle Stage 06 notebook with stdlib only."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(s).strip()))

    def code(s):
        s = textwrap.dedent(s).strip()
        py_lines = "\n".join(l for l in s.splitlines() if not l.strip().startswith(("!", "%")))
        if py_lines.strip():
            ast.parse(py_lines)
        cells.append(dict(cell_type='code', metadata={}, source=s, outputs=[], execution_count=None))

    md('''
    # Stage 06 — Fingerprint candidate ranking

    Bridge Stage 05 molecule-level predictions and the Stage 04 mass-sorted database.
    Run all cells in order on Kaggle with Stage 02/04/05 outputs attached. CPU is
    sufficient; no network access is needed. Sampled tuning uses **val_model only**;
    val_rank is held out from this scorer's tuning for Stage 09 (not cross-validated
    out-of-fold neural predictions). Stage 05 selects its model on val_model;
    Stage 06 also reserves val_rank labels until after freezing its configuration.

    Outputs: three top-100 feature tables, a validated top-25 submission, frozen
    config, manifest, and `/kaggle/working/fprank.py` for Stage 10.
    ''')
    code('''
    !pip install -q rdkit
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    from pathlib import Path
    import importlib.util, sys, json, hashlib, time, gc
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    WORK = Path('/kaggle/working')
    OUT = WORK / 'artifacts/fprank'
    OUT.mkdir(parents=True, exist_ok=True)
    STARTED = time.time()
    # Optional overrides; discovery prefers working files, then newest mtime.
    DB_DIR = None
    PRED_DIR = None
    DATA_DIR = None
    SPLIT_DIR = None
    ''')
    md('''
    ## Stage and import modules
    Embedded snapshots make this notebook self-contained. All four modules are
    written to and imported strictly from `/kaggle/working/`. The candidate module
    checks the database fingerprint schema against evalkit before retrieval.
    ''')
    sources = {name: (ROOT / name).read_text(encoding='utf-8')
               for name in ('evalkit.py', 'candidates.py', 'fprank.py')}
    if (ROOT / 'spec2fp.py').is_file():
        sources['spec2fp.py'] = (ROOT / 'spec2fp.py').read_text(encoding='utf-8')
    else:
        tree = ast.parse((ROOT / 'build_05_spec2fp_train.py').read_text(encoding='utf-8'))
        sources['spec2fp.py'] = next(ast.literal_eval(n.value) for n in tree.body
                                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPEC2FP_SOURCE' for t in n.targets))
    for filename, source in sources.items():
        ast.parse(source)
        code(f"(WORK / {filename!r}).write_text({source!r}, encoding='utf-8')")
    code('''
    def import_work(name):
        path = WORK / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == path.resolve()
        return module
    evalkit = import_work('evalkit')
    candidates = import_work('candidates')
    spec2fp = import_work('spec2fp')
    fprank = import_work('fprank')

    def locate(filename, directory=None):
        if directory is not None:
            path = Path(directory) / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        paths = sorted({p.resolve() for root in (WORK, Path('/kaggle/input'))
                        for p in root.rglob(filename) if p.is_file()},
                       key=lambda p: (p.is_relative_to(WORK.resolve()), p.stat().st_mtime, str(p)),
                       reverse=True)
        if not paths:
            raise FileNotFoundError(f'Missing {filename} in /kaggle/working or /kaggle/input')
        return paths[0]

    def heldout_exclusions(split_path, train_path):
        def read_metadata(path, wanted):
            available = pq.ParquetFile(path).schema_arrow.names
            return pd.read_parquet(path, columns=[c for c in wanted if c in available])
        split_meta = read_metadata(split_path, ['molecule_id', 'split3', 'ik14'])
        train_meta = read_metadata(train_path, ['molecule_id', 'smiles', 'ik14'])
        split_meta['molecule_id'] = split_meta.molecule_id.astype(str)
        train_meta['molecule_id'] = train_meta.molecule_id.astype(str)
        held = split_meta.loc[split_meta.split3.ne('train')]
        held_train = train_meta.loc[train_meta.molecule_id.isin(held.molecule_id)]
        held_smiles = held_train.smiles.dropna().unique()
        keys = set()
        for s in held_smiles:
            mol = candidates.standardize_mol(s)
            if mol is not None:
                keys.add(evalkit.fast_key(mol))
            keys.add(evalkit.fast_key(s))
        for frame in (held, held_train):
            if 'ik14' in frame:
                keys.update(frame.ik14.dropna().astype(str))
        return {key for key in keys if key}

    db_root = locate('db.parquet', DB_DIR).parent
    pred_root = locate('fp_pred_val_model.npy', PRED_DIR).parent
    pred_manifest = json.loads((pred_root / 'manifest.json').read_text())
    assert pred_manifest['trained_on_split'] == 'train', 'Stage 05 split contract mismatch'
    metadata = candidates.configure(db_root, exclusion_scope='train', row_group_cache=2)
    # Compact Stage 04 datasets can omit build.sqlite.
    if (db_root / 'build.sqlite').is_file():
        with candidates.open_sql(db_root / 'build.sqlite', readonly=True) as db:
            exclude_keys = candidates.validation_exclusions(db)
    else:
        exclude_keys = heldout_exclusions(locate('splits.parquet', SPLIT_DIR),
                                          locate('train_clean.parquet', DATA_DIR))
    print('Database:', db_root, 'held-out keys:', len(exclude_keys))
    ''')
    md('''
    ## Align observed query masses with prediction rows
    Only metadata columns are read, never peak arrays or the complete compound DB.
    Observed neutral masses are averaged across spectra per molecule, matching
    Stage 04. No ground-truth formula or structure-derived mass is used to retrieve
    validation candidates. ID files are authoritative for prediction alignment.
    ''')
    code('''
    def read_queries(path):
        columns = pq.ParquetFile(path).schema_arrow.names
        wanted = ['molecule_id', 'neutral_mass', 'precursor_mz', 'adduct', 'polarity']
        frame = pd.read_parquet(path, columns=[c for c in wanted if c in columns])
        frame['molecule_id'] = frame.molecule_id.astype(str)
        frame['neutral_mass'] = candidates.observed_mass(frame)
        # Conflicting spectrum adducts can produce extreme mass outliers.
        return frame.groupby('molecule_id', sort=False, as_index=False).neutral_mass.median()

    train_path = locate('train_clean.parquet', DATA_DIR)
    test_path = locate('test_clean.parquet', DATA_DIR)
    split_path = locate('splits.parquet', SPLIT_DIR)
    splits = pd.read_parquet(split_path, columns=['molecule_id', 'split3'])
    splits['molecule_id'] = splits.molecule_id.astype(str)
    splits = splits.drop_duplicates()
    assert not splits.molecule_id.duplicated().any(), 'Conflicting splits'
    train_queries = read_queries(train_path).merge(splits, on='molecule_id', how='left', validate='one_to_one')
    assert train_queries.split3.notna().all(), 'Missing split assignments'
    test_queries = read_queries(test_path)
    test_queries['split3'] = 'test'

    def load_split(name):
        ids = pd.read_parquet(pred_root / f'fp_pred_ids_{name}.parquet')
        ids['molecule_id'] = ids.molecule_id.astype(str)
        base = test_queries if name == 'test' else train_queries.loc[train_queries.split3.eq(name)]
        assert set(ids.molecule_id) == set(base.molecule_id), f'{name}: prediction ID mismatch'
        queries = ids[['molecule_id']].merge(base, on='molecule_id', how='left',
                                            sort=False, validate='one_to_one')
        fp = np.load(pred_root / f'fp_pred_{name}.npy', mmap_mode='r')
        elem = np.load(pred_root / f'elem_pred_{name}.npy', mmap_mode='r')
        maccs_path = pred_root / f'maccs_pred_{name}.npy'
        if not maccs_path.is_file():
            try:
                maccs_path = locate(f'maccs_pred_{name}.npy', PRED_DIR)
            except Exception:
                maccs_path = None
        maccs = np.load(maccs_path, mmap_mode='r') if maccs_path is not None and Path(maccs_path).is_file() else None
        if maccs is not None:
            print(f'{name}: loaded MACCS predictions {maccs.shape}')
        else:
            print(f'{name}: no MACCS predictions found; scoring with Morgan only')
        fprank.validate_inputs(queries, fp, elem, maccs=maccs)
        return queries, fp, elem, maccs

    truth_by_smiles = {}
    fast_truth_by_smiles = {}
    def truth_for(queries, include_fast=False):
        frame = pd.read_parquet(train_path, columns=['molecule_id', 'smiles'])
        frame['molecule_id'] = frame.molecule_id.astype(str)
        frame = frame[frame.molecule_id.isin(queries.molecule_id)].dropna(subset=['smiles'])
        # A molecule_id may have multiple spectra with slight SMILES variations (kekule vs aromatic, etc.).
        # Pick the consensus (most frequent) valid SMILES per molecule_id to guarantee one structure per ID.
        smiles_counts = frame.groupby(['molecule_id', 'smiles']).size().reset_index(name='count')
        best_smiles = smiles_counts.sort_values(['molecule_id', 'count'], ascending=[True, False]).drop_duplicates(subset=['molecule_id'])
        truth = queries[['molecule_id']].merge(best_smiles[['molecule_id', 'smiles']], on='molecule_id', how='left', validate='one_to_one')
        assert truth.smiles.notna().all(), 'Missing truth SMILES for some validation queries'
        for s in tqdm(truth.smiles.unique(), desc='Unique truth keys'):
            if s not in truth_by_smiles:
                truth_by_smiles[s] = evalkit.canon_ik14(s)
            if include_fast and s not in fast_truth_by_smiles:
                mol = candidates.standardize_mol(s)
                fast_truth_by_smiles[s] = evalkit.fast_key(mol) if mol is not None else evalkit.fast_key(s)
        keys = [truth_by_smiles[s] for s in truth.smiles]
        assert all(keys), 'Invalid validation truth'
        if include_fast:
            return keys, [fast_truth_by_smiles[s] for s in truth.smiles]
        return keys
    ''')
    md('''
    ## Tune on val_model and freeze
    The 3×3 ppm/temperature grid samples up to **5,000 val_model queries** with
    NumPy RNG seed 42. Retrieve each sample's 20-ppm pool once, plus the wider
    fallback when a smaller grid window is sparse. Compute all three temperature
    feature sets together, retain only compact arrays, and run coordinate ascent
    entirely in NumPy. No SQLite writes, Parquet reads, or canonicalization occur
    during weight search. Runtime depends on candidate density and the host.

    Tuning uses a fast_key/known-ik14 **MRR@25 proxy**, without tautomer deduplication;
    full canonical deduplicated validation metrics are reported after freezing.
    Three coordinate-ascent sweeps try 0, 0.5×, 0.75×, 1.25×, and 2× weights
    (the last has a 0.1 floor). Equal improvements prefer smaller weight changes.
    Report both overall proxy MRR and conditional MRR on the fixed subset whose
    truth appears in the strict 20-ppm pool; fallback-only hits are excluded from
    that conditional subset. Zero-hit queries remain in the overall denominator.

    NB is divided by 4096 and negative element L1 by 10 before applying weights;
    raw values are retained for Stage 09. Stage 05 probabilities have already had
    its positive-weight correction applied. This stage tunes temperature, without
    claiming to fit per-bit Platt slopes. Optional pre-fitted slopes/intercepts and
    reliability weights are supported by the module and must come from val_model.
    Provenance counts curated-source bits after removing the training bit.
    ''')
    code('''
    q_model, fp_model, elem_model, maccs_model = load_split('val_model')
    truth_model, fast_truth_model = truth_for(q_model, include_fast=True)
    q_model['truth_fast_key'] = fast_truth_model
    tuning_cfg = dict(fprank.DEFAULT_CONFIG, exclude_keys=exclude_keys)
    frozen = fprank.tune(q_model, fp_model, elem_model, truth_model, tuning_cfg,
                         sweeps=3, max_tune_queries=5000, maccs=maccs_model)
    frozen['database_build_id'] = metadata['build_id']
    frozen['element_order'] = list(fprank.ELEMENTS)
    config_path = OUT / 'fprank_config.json'
    config_path.write_text(json.dumps(frozen, indent=2))
    # Reload the serialized contract; no further tuning after this point.
    frozen = json.loads(config_path.read_text())
    del fp_model, elem_model, maccs_model
    gc.collect()
    print('Frozen sampled tuning diagnostics:', {k: frozen[k] for k in
          ('tuning_queries', 'hit_queries_20ppm', 'oracle_20ppm_recall',
           'overall_proxy_mrr25', 'covered_hit_mrr25')})
    ''')
    md('''
    ## Stream ranked candidate tables
    Every mass-window candidate is scored. If fewer than 25 survive filtering,
    widen to the larger of ±50 ppm and ±0.05 Da. Keep a 256-row shortlist, then
    canonicalize only until 25 unique canonical hits are collected. Fill ranks
    26–100 using fast-key deduplication; their `canon_ik14` is null and
    `identity_kind` is `fast_only`. Stage 09 must not treat those tail identities
    as tautomer-canonical. Sparse windows may produce fewer than 100 rows.
    If both windows are empty, score the nearest 50 eligible database rows so
    test molecules retain predictions even outside the normal mass tolerance.
    `scoring_source_mask` records the mask used for scoring; `source_mask` may
    combine provenance when tautomer collisions are merged.
    Lazy chemistry uses bounded memory caches and WAL/NORMAL SQLite persistence,
    committing every 5,000 writes and at tuning/split boundaries, not per query.

    Mass sorting improves mmap and row-group locality; fallback windows and lazy
    fingerprint generation can still cause nonsequential reads. A 2048-row bit
    batch occupies about 34 MB; total RSS also includes Python, Torch, database
    row groups, and OS page cache, so a hard <2 GB total is not guaranteed.
    `oracle_top100_recall` measures canonical-or-standardized-fast-key matches in the emitted top-100
    shortlist, including fallback candidates; it is not full database membership.
    `mrr25_all` includes all validation queries, including empty/missing shortlists.
    `mrr25_given_in_pool` conditions on a top-100 truth-key match. Tail identities
    remain fast/known keys, so this coverage diagnostic is a proxy for tautomer matches.
    ''')
    code('''
    summaries = {}
    test_predictions = {}
    val_model_truth = dict(zip(q_model.molecule_id, truth_model))
    val_model_fast_truth = dict(zip(q_model.molecule_id, fast_truth_model))
    output_schema = fprank.output_schema()
    for name in ('val_model', 'val_rank', 'test'):
        queries, fp, elem, maccs = load_split(name)
        run_cfg = dict(frozen)
        if name != 'test':
            run_cfg['exclude_keys'] = exclude_keys
        path = OUT / f'cand_fp_{name}.parquet'
        writer, rows, found, rr = None, 0, set(), {}
        pending, pending_rows = [], 0
        if name == 'val_model':
            truth, fast_truth = val_model_truth, val_model_fast_truth
        elif name == 'val_rank':
            canonical_keys, fast_keys = truth_for(queries, include_fast=True)
            truth = dict(zip(queries.molecule_id, canonical_keys))
            fast_truth = dict(zip(queries.molecule_id, fast_keys))
        else:
            truth, fast_truth = {}, {}
        in_pool = dict.fromkeys(truth, False)
        rr = dict.fromkeys(truth, 0.0)
        try:
            iterator = fprank.iter_rank_candidates(queries, fp, elem, run_cfg, maccs_preds=maccs)
            for frame in tqdm(iterator, total=len(queries), desc=f'Ranking {name}'):
                if frame.empty:
                    continue
                qid = frame.molecule_id.iloc[0]
                found.add(qid)
                table = pa.Table.from_pandas(frame, schema=output_schema, preserve_index=False, safe=True)
                if writer is None:
                    writer = pq.ParquetWriter(path, output_schema, compression='zstd')
                pending.append(table)
                pending_rows += len(frame)
                if pending_rows >= 8192:
                    writer.write_table(pa.concat_tables(pending))
                    pending, pending_rows = [], 0
                rows += len(frame)
                if name == 'test':
                    test_predictions[qid] = ';'.join(frame.smiles.head(25))
                else:
                    hits = np.flatnonzero(frame.ik14.head(25).to_numpy() == truth[qid])
                    rr[qid] = 1.0/(int(hits[0])+1) if len(hits) else 0.0
                    in_pool[qid] = bool(
                        frame.ik14.head(100).eq(truth[qid]).any()
                        or frame.fast_key.head(100).eq(fast_truth[qid]).any()
                    )
        finally:
            candidates.flush_cache()
            if writer is not None:
                if pending:
                    writer.write_table(pa.concat_tables(pending))
                writer.close()
        if writer is None:
            raise RuntimeError(f'{name}: no candidates for any query; check database and masses')
        summaries[name] = dict(queries=len(queries), rows=rows, empty_queries=len(queries)-len(found))
        if name != 'test':
            covered, total_rr = sum(in_pool.values()), sum(rr.values())
            conditional = sum(rr[qid] for qid in truth if in_pool[qid]) / max(covered, 1)
            summaries[name].update(mrr25=total_rr/max(len(queries), 1),
                mrr25_all=total_rr/max(len(queries), 1), in_pool_queries=covered,
                oracle_top100_recall=covered/max(len(queries), 1),
                mrr25_given_in_pool=conditional, mrr25_when_in_top100=conditional)
        print(name, summaries[name])
        del fp, elem, maccs
        gc.collect()
    ''')
    md('''
    ## Submission validation and provenance
    Empty mass windows use the nearest 50 eligible rows as a final coverage fallback.
    A database with no eligible structures is an explicit error. Submission ID
    completeness is checked before writing. Configuration stays frozen for val_rank
    and test. Stage 10 must configure the same candidate DB before calling the API.
    ''')
    code('''
    submission = test_queries[['molecule_id']].copy()
    submission['smiles'] = submission.molecule_id.map(test_predictions)
    missing = submission.loc[submission.smiles.isna(), 'molecule_id'].tolist()
    assert not missing, f'Coverage failure after nearest-row fallback: {missing[:20]}'
    valid, errors, warnings, stats = evalkit.validate_submission(submission, test_queries.molecule_id)
    assert valid, errors
    submission.to_csv(OUT / 'submission_fprank.csv', index=False)
    print(stats, warnings)

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()

    vm_ids = train_queries.loc[train_queries.split3.eq('val_model'), 'molecule_id'].tolist()
    vr_ids = train_queries.loc[train_queries.split3.eq('val_rank'), 'molecule_id'].tolist()
    manifest = dict(stage='fprank', tuned_on='val_model', trained_on_split='train',
                    heldout_ids_hash=hashlib.sha256(''.join(sorted(vm_ids + vr_ids)).encode()).hexdigest(),
                    exclusion_scope='train',
                    database_build_id=metadata['build_id'], fp_spec=evalkit.FP_SPEC,
                    inputs=dict(database=str(db_root), predictions=str(pred_root),
                                train=str(train_path), test=str(test_path), splits=str(split_path)),
                    summaries=summaries, submission_validation=stats,
                    elapsed_seconds=time.time()-STARTED,
                    upstream_manifest_sha256=sha256(pred_root / 'manifest.json'),
                    split_file_sha256=sha256(split_path),
                    modules={n: sha256(WORK / n) for n in
                             ('evalkit.py', 'candidates.py', 'spec2fp.py', 'fprank.py')})
    manifest['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                         for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Stage 06 complete:', OUT)
    ''')
    for i, cell in enumerate(cells):
        cell['id'] = f'stage06-{i:03d}'
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                         language_info=dict(name='python', version='3.11')),
                nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    nb_dict = build_notebook()
    nb_str = json.dumps(nb_dict, indent=1, ensure_ascii=False)
    for fname in ('06_fp_rank_infer.ipynb', '06_fp_rank_infer_2.ipynb', '06_fp_rank_infer_3.ipynb',
                  '06_fp_rank_infer_4.ipynb'):
        out_path = ROOT / fname
        out_path.write_text(nb_str, encoding='utf-8')
        print(f"Generated {out_path}")
