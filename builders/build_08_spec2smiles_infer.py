"""Build and statically validate the offline Stage 08 notebook."""
import ast
import json
import textwrap
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s):
        s = textwrap.dedent(s).strip()
        ast.parse(s)
        assert len(s.splitlines()) < 60, s[:100]
        cells.append(nbf.v4.new_code_cell(s))

    md('''
    # Stage 08 — diverse spectrum-to-SMILES inference
    Reuse the Stage 07 v2 checkpoint **without retraining**. Run beam 32 / return 32,
    plus 50 nucleus samples (p=0.9, T=0.8), at molecule level on two isolated GPU
    subprocesses. Only batch size changes on OOM. GPU allocator cap: 11 GB/process;
    host RSS budgets: parent 2 GB + two 3.5 GB workers. No soft time cutoff.

    Attach clean train/test, splits, Stage 07 checkpoint/vocabulary/config and Stage 05
    element arrays with ID tables. Optional: formula DB and Stage 06 candidate tables.
    **Before Run All, place the original Stage 02 `evalkit.py` at
    `/kaggle/working/evalkit.py`. It is read only from that exact path.**
    All other Python modules are embedded; RDKit/torch must be preinstalled. Internet OFF.
    This checkpoint uses SMILES, so SELFIES and its wheel are not needed.

    Outputs under `/kaggle/working/artifacts/spec2smiles_infer`: all three `cand_gen_*`
    tables, `submission_gen.csv`, formula ranks, query diagnostics, metrics and manifest.
    Scores are higher-is-better. Runtime is measured, including an initial 200-query
    projection; stage-10 estimates cover this generator component, not other stages.
    `SMOKE_TEST=True` runs a small synthetic CPU verification without attached inputs.
    ''')
    code('''
    CFG = dict(seed=42, SMOKE_TEST=False, work='/kaggle/working',
        out='/kaggle/working/artifacts/spec2smiles_infer',
        cache='/kaggle/working/spec2smiles_infer_cache', resume=True,
        beam=32, num_return=32, samples=50, top_p=0.9, temperature=0.8,
        length_penalty=0.7, max_length=128, batch_size=4, ppm=10., relax_ppm=50.,
        gpu_memory_gb=11., parent_ram_gb=2., worker_ram_gb=3.5,
        time_budget_s=12*3600, stage10_generator_budget_s=8*3600,
        alpha_grid=[0., 0.1, 0.25, 0.5, 1.],
        formula_weight_grid=[0., 0.1, 0.25], prior_weight=0.15, presence_weight=0.5,
        formula_max_counts={}, queries_per_part=100,
        elements_order=['C','H','N','O','S','P','F','Cl','Br','I'])
    ''')
    code('''
    import gc, hashlib, importlib, json, os, subprocess, sys, time
    from collections import Counter
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    import torch
    from rdkit import Chem
    from tqdm.auto import tqdm
    START = time.perf_counter()
    WORK, OUT, CACHE = [Path(CFG[k]) for k in ('work', 'out', 'cache')]
    for directory in (WORK, OUT, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    print(dict(torch=torch.__version__, rdkit=Chem.rdBase.rdkitVersion,
        numpy=np.__version__, pandas=pd.__version__, gpu_count=torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, round(p.total_memory / 1e9, 2), 'GB')
    assert 25 <= CFG['beam'] <= 50 and 25 <= CFG['num_return'] <= 50
    assert CFG['samples'] == 50 and CFG['relax_ppm'] == 50
    ''')
    md('''
    ## Embedded model and inference code
    The saved v2 model has **no trained formula-prefix/no-formula tokens**. Its native
    predicted-element conditioning is retained for one beam pass and one sampling pass.
    Formula enumeration/ranking still runs independently, and its score can rerank
    generated structures (weight selected on val_model). No invented tokens or retraining.
    A different formula-prefix checkpoint fails explicitly instead of silently ignoring it.

    Spectra are pooled by a sqrt(peak-count)-weighted mean of valid encoded positions,
    pooled embeddings and condition vectors. Formula search covers the train-observed
    maximum count domain, optionally enlarged with `formula_max_counts`, with train-only
    1st–99th percentile element/C ratios. This empirical domain can exclude novel formulas;
    formula recall reports that limitation. DB frequencies never set chemical bounds.
    ''')
    for name in ('spec2fp.py', 'spec2smiles.py', 'spec2smiles_infer.py'):
        source = (ROOT / 'src' / name).read_text(encoding='utf-8')
        ast.parse(source)
        code(f"module_path = WORK / {name!r}\nmodule_path.write_text('', encoding='utf-8')")
        lines = source.splitlines(keepends=True)
        for i in range(0, len(lines), 45):
            fragment = '\n'.join(repr(s) for s in lines[i:i+45])
            code("fragment = (\n" + fragment + "\n)\nwith module_path.open('a', encoding='utf-8') as stream:\n    stream.write(fragment)")
    code('''
    sys.path.insert(0, str(WORK))
    importlib.invalidate_caches()
    for name in ('spec2fp', 'spec2smiles', 'spec2smiles_infer'):
        sys.modules.pop(name, None)
    import spec2smiles_infer as infer
    infer.seed_all(CFG['seed'])
    if not CFG['SMOKE_TEST']:
        evalkit_path = infer.find_file('evalkit.py', roots=[str(WORK)])
        assert evalkit_path.resolve() == (WORK / 'evalkit.py').resolve()
        evalkit = infer.load_evalkit(WORK)
        print('Read-only evalkit:', evalkit.__file__)
    ''')
    md('''
    ## Synthetic CPU smoke test
    Exercises both full-width decoders, invalid/unfinished handling, mass rejection,
    conditional relaxation, canonical aggregation, formula ranking and metrics. It does
    not claim to validate the real trained checkpoint or GPU throughput.
    ''')
    code('''
    if CFG['SMOKE_TEST']:
        class SmokeDecoder(torch.nn.Module):
            def decode(self, tokens, memory, mask):
                logits = torch.full((*tokens.shape, 8), -20.)
                target = [4, 4, 5, 2][min(tokens.shape[1]-1, 3)]
                logits[:, -1, target] = 10.
                return logits
        cfg = dict(CFG, max_length=8)
        memory, mask = torch.zeros(2, 4, 8), torch.zeros(2, 4, dtype=torch.bool)
        model = SmokeDecoder()
        beams = infer.decode_beam(model, memory, mask, cfg)
        samples = infer.decode_sample(model, memory, mask, cfg)
        assert len(beams) == 2 and all(len(s) == 50 for s in samples)
        assert all(s[0][0][:4] == [4, 4, 5, 2] for s in samples)
        mass = 46.041864812
        paths = [('CCO', -1., 'beam'), ('OCC', -2., 'sample'),
                 ('not_smiles', -3., 'sample'), ('C', -4., 'sample')]
        rows, diag = infer.postprocess('q', paths, [(mass, '[M+H]+')], [], cfg)
        assert len(rows) == 1 and rows[0]['count'] == 2 and diag['valid'] == 3
        relaxed, info = infer.postprocess('q', paths, [(mass*(1+30e-6), 'x')], [], cfg)
        assert relaxed and info['no_strict_survivor']
        empty, info = infer.postprocess('q', paths, [(100., 'x')], [], cfg)
        assert not empty and info['no_survivor']
        stats = dict(allowed=['C','H','O'], bounds=dict(C=3,H=10,O=2),
            ratios=dict(C=[1,1], H=[0,4], O=[0,2]), prior={'C2H6O': 5})
        pred = np.array([2,6,0,1,0,0,0,0,0,0])
        ranked = infer.rank_formulas([(mass, 'x')], pred, stats, cfg)
        assert ranked[0]['formula'] == 'C2H6O'
        table = infer.scored(pd.DataFrame(rows), 0.25)
        assert infer.ranking_metrics(table, {'q': rows[0]['ik14'], 'missing': 'x'})['mrr25'] == .5
        table.to_parquet(OUT / 'smoke_candidates.parquet', index=False)
        print('CPU smoke checks passed; production pipeline skipped.')
    ''')
    md('## Discover inputs and verify schemas / split coverage')
    code('''
    if not CFG['SMOKE_TEST']:
        paths = {key: str(infer.find_file(name)) for key, name in dict(
            train='train_clean.parquet', test='test_clean.parquet', splits='splits.parquet',
            model='spec2smiles_model.pt').items()}
        for key, filename in [('vocab','smiles_vocab.json'),('config','spec2smiles_config.json')]:
            paths[key] = str(infer.find_file(filename, hint=str(Path(paths['model']).parent)))
        for name in ('val_model', 'val_rank', 'test'):
            paths[f'elem_{name}'] = str(infer.find_file(f'elem_pred_{name}.npy'))
            paths[f'ids_{name}'] = str(infer.find_file(f'fp_pred_ids_{name}.parquet',
                hint=str(Path(paths[f'elem_{name}']).parent)))
        db_path = infer.find_file('db.parquet', required=False)
        schema_path = infer.find_file('schema.json', required=False)
        if schema_path:
            schema_metadata = json.loads(schema_path.read_text())
            print('Loaded schema.json; normalized required columns verified below:', schema_path)
        required = ['molecule_id','mz','inten','precursor_mz','adduct','ce','instrument','polarity']
        train_adapter = infer.SchemaAdapter(paths['train'], required + ['smiles'])
        test_adapter = infer.SchemaAdapter(paths['test'], required)
        splits = pd.read_parquet(paths['splits'])
        assert {'molecule_id','split3'} <= set(splits)
        splits['molecule_id'] = splits.molecule_id.astype(str)
        assert not splits.molecule_id.duplicated().any()
        assert set(splits.split3) == {'train','val_model','val_rank'}
        split_map = splits.set_index('molecule_id').split3
        train_ids = set()
        for frame in train_adapter.batches(['molecule_id']):
            train_ids.update(frame.molecule_id)
        assert train_ids == set(split_map.index), 'Split/train molecule coverage mismatch'
        test_ids = set()
        for frame in test_adapter.batches(['molecule_id']):
            test_ids.update(frame.molecule_id)
        assert not (test_ids & train_ids), 'Train/test molecule IDs overlap'
        ids = {name: sorted(splits.loc[splits.split3.eq(name), 'molecule_id'])
               for name in ('val_model','val_rank')}
        ids['test'] = sorted(test_ids)
        for name in ids:
            infer.prediction_map(paths, name, ids[name], CFG)
        saved_cfg = json.loads(Path(paths['config']).read_text())
        assert saved_cfg.get('elements_order', CFG['elements_order']) == CFG['elements_order']
        assert not saved_cfg.get('FULL_TRAIN', False), 'Full-train checkpoint contaminates validation'
        print({name: len(values) for name, values in ids.items()})
    ''')
    md('''
    ## Train-only formula priors and two query shards
    Known adducts determine neutral masses. Unknown adducts are logged and use polarity
    hypotheses. Filtering accepts the ppm window of any retained hypothesis. Formula
    sanity includes nonnegative integer DBE, generalized nitrogen parity and train ratios.
    Array rows are joined by their accompanying molecule IDs, never by parquet row order.
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        formula_stats = infer.build_formula_stats(train_adapter, split_map, db_path, CFG)
        paths['formula_stats'] = str(OUT / 'formula_stats.json')
        Path(paths['formula_stats']).write_text(json.dumps(formula_stats))
        print('Formula bounds:', formula_stats['bounds'])
        assignment = {mid: (name, i % 2, (i // 2) // CFG['queries_per_part'])
                      for name, values in ids.items() for i, mid in enumerate(values)}
        writers = {}
        parts = {name: {str(s): [] for s in (0,1)} for name in ids}
        try:
            for adapter in (train_adapter, test_adapter):
                for frame in adapter.batches(required + ['neutral_mass']):
                    frame = frame.loc[frame.molecule_id.isin(assignment)].copy()
                    frame['ce'] = pd.to_numeric(frame.ce, errors='coerce').fillna(0.)
                    frame['instrument'] = frame.instrument.fillna('<unk>').astype(str)
                    frame['adduct'] = frame.adduct.fillna('<unk>').astype(str)
                    frame['polarity'] = frame.polarity.fillna('<unk>').astype(str)
                    groups = frame.molecule_id.map(assignment)
                    for key, group in frame.groupby(groups, sort=False):
                        name, shard, part = key
                        folder = CACHE / name / str(shard)
                        folder.mkdir(parents=True, exist_ok=True)
                        table = pa.Table.from_pandas(group, preserve_index=False)
                        if key not in writers:
                            part_path = folder / f'{part:05d}.parquet'
                            parts[name][str(shard)].append(str(part_path))
                            writers[key] = pq.ParquetWriter(part_path, table.schema, compression='zstd')
                        writers[key].write_table(table)
                    infer.check_ram(CFG['parent_ram_gb'] * 1e9)
        finally:
            for writer in writers.values():
                writer.close()
        input_hashes = {key: infer.sha256(value) for key, value in paths.items()}
        source_hashes = {name: infer.sha256(WORK / name) for name in
                         ('spec2fp.py','spec2smiles.py','spec2smiles_infer.py','evalkit.py')}
        run_hash = hashlib.sha256(json.dumps(dict(cfg=CFG, inputs=input_hashes,
            sources=source_hashes), sort_keys=True).encode()).hexdigest()
        for name in parts:
            for shard in parts[name]:
                parts[name][shard].sort()
        job = dict(cfg=CFG, paths=paths, ids=ids, parts=parts, run_hash=run_hash)
        job_path = OUT / 'job.json'
        job_path.write_text(json.dumps(job))
        del assignment, writers, formula_stats
        gc.collect()
    ''')
    md('''
    ## Launch exactly two isolated inference workers
    Two subprocesses receive disjoint molecule lists and independent CUDA contexts.
    With one GPU they run sequentially on that GPU; normal Kaggle T4x2 runs concurrently.
    Every 100-query part is checkpointed with a configuration/input/source hash. A matching
    completed part resumes safely. No beam/sampling reductions or time-budget truncation.
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        gpu_count = torch.cuda.device_count()
        assert gpu_count >= 1, 'Production inference requires GPU; use SMOKE_TEST on CPU'
        def launch(shard):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(shard if gpu_count >= 2 else 0),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                PYTHONHASHSEED=str(CFG['seed']))
            log = (OUT / f'worker_{shard}.log').open('w')
            process = subprocess.Popen([sys.executable, str(WORK / 'spec2smiles_infer.py'),
                str(job_path), str(shard)], env=env, stdout=log, stderr=subprocess.STDOUT)
            return process, log
        workers, launch_started = {}, time.perf_counter()
        workers[0] = launch(0)
        if gpu_count >= 2:
            workers[1] = launch(1)
        projected = False
        try:
            while True:
                if gpu_count == 1 and workers[0][0].poll() == 0 and 1 not in workers:
                    workers[1] = launch(1)
                for shard, (proc, log) in workers.items():
                    if proc.poll() not in (None, 0):
                        raise RuntimeError(f'Worker {shard} failed; inspect worker_{shard}.log')
                pilot = [OUT / f'val_model_shard{s}_00000.done.json' for s in (0,1)]
                if not projected and all(p.exists() for p in pilot):
                    reports = [json.loads(p.read_text()) for p in pilot]
                    rates = [r['queries']/max(r['seconds'], 1e-9) for r in reports]
                    rate = sum(rates) if gpu_count >= 2 else 2 / sum(1/r for r in rates)
                    projection = dict(pilot_queries=sum(r['queries'] for r in reports),
                        generator_test_seconds=len(ids['test'])/rate, gpu_count=gpu_count,
                        includes='encoding, generation, formula search, postprocessing, shard writes')
                    (OUT / 'runtime_projection.json').write_text(json.dumps(projection, indent=2))
                    print('Initial stage-10 generator projection:', projection, flush=True)
                    projected = True
                completed = list(OUT.glob('*.done.json'))
                print('Completed query parts:', len(completed), 'elapsed min:',
                      round((time.perf_counter()-launch_started)/60, 1), flush=True)
                if len(workers) == 2 and all(p.poll() == 0 for p, _ in workers.values()):
                    break
                time.sleep(20)
        finally:
            for proc, log in workers.values():
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                log.close()
        measured_inference_seconds = time.perf_counter() - launch_started
    ''')
    md('''
    ## Merge, tune alpha on val_model, and evaluate
    Raw duplicate hits are retained across beam and sampling; max and mean length-normalized
    model log-probabilities are recorded. `score = max_log_prob + alpha*log(count)` plus
    an optional formula-score term. Alpha/formula weight use val_model only; val_rank is
    evaluated afterward without tuning. All ranks use tautomer-canonical InChIKey14.
    Canonicalization that exhausts its 1000-tautomer/transform guard is rejected and counted.
    Validity is valid RDKit parses / returned beam hypotheses plus all sampling attempts;
    unfinished samples count as invalid. Mass consistency is strict mass hits / valid parses.
    Uniqueness is unique survivors / surviving raw hits, after the conditional 50-ppm relaxation.
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        raw, diagnostics, formula_tables = {}, {}, {}
        for name in ids:
            expected_parts = [OUT / f'{name}_shard{s}_{Path(part).stem}' for s in (0,1)
                for part in job['parts'][name][str(s)]]
            assert all(p.with_suffix('.done.json').exists() for p in expected_parts)
            raw[name] = pd.concat([pd.read_parquet(p.with_suffix('.parquet'))
                                   for p in expected_parts], ignore_index=True)
            diagnostics[name] = pd.concat([pd.read_parquet(p.with_suffix('.diag.parquet'))
                                           for p in expected_parts], ignore_index=True)
            formula_tables[name] = pd.concat([pd.read_parquet(p.with_suffix('.formula.parquet'))
                                              for p in expected_parts], ignore_index=True)
            assert not diagnostics[name].molecule_id.duplicated().any()
            assert set(diagnostics[name].molecule_id) == set(ids[name])
            diagnostics[name].to_parquet(OUT / f'gen_diagnostics_{name}.parquet', index=False)
            formula_tables[name].to_parquet(OUT / f'formula_rank_{name}.parquet', index=False)
        truths, true_formulas = {}, {}
        label_sets = {name: {} for name in ('val_model','val_rank')}
        for frame in train_adapter.batches(['molecule_id','smiles']):
            for row in frame.itertuples():
                name = split_map[row.molecule_id]
                if name in label_sets:
                    label_sets[name].setdefault(row.molecule_id, Counter()).update([str(row.smiles)])
        for name, labels in label_sets.items():
            truths[name], true_formulas[name] = {}, {}
            for mid, choices in tqdm(labels.items(), desc=f'Canonical truth {name}'):
                smi = sorted(choices, key=lambda s: (-choices[s], s))[0]
                result = infer.canonical_record(smi)
                assert result and len(result) == 4, f'Cannot canonicalize truth {mid}'
                truths[name][mid], true_formulas[name][mid] = result[1], result[3]
            assert set(truths[name]) == set(ids[name])
        trials = []
        for alpha in CFG['alpha_grid']:
            for weight in CFG['formula_weight_grid']:
                result = infer.ranking_metrics(infer.scored(raw['val_model'], alpha, weight), truths['val_model'])
                trials.append(dict(alpha=alpha, formula_weight=weight, **result))
        tuning = pd.DataFrame(trials).sort_values(['mrr25','alpha','formula_weight'],
                                                ascending=[False,True,True])
        best = tuning.iloc[0]
        tuning.to_csv(OUT / 'alpha_tuning_val_model.csv', index=False)
        print('Selected on val_model:', best.to_dict())
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        metrics, comparisons, final = {}, [], {}
        for name in ids:
            final[name] = infer.scored(raw[name], float(best.alpha), float(best.formula_weight))
            final[name].to_parquet(OUT / f'cand_gen_{name}.parquet', index=False)
            d = diagnostics[name]
            metrics[name] = dict(queries=len(ids[name]),
                validity_pct=100*int(d.valid.sum())/max(int(d.attempted.sum()),1),
                mass_consistency_pct=100*int(d.strict_mass_hits.sum())/max(int(d.valid.sum()),1),
                uniqueness= float(d.unique.sum())/max(int(d.kept_hits.sum()),1),
                canonicalization_limited=int(d.canonicalization_limited.sum()),
                relaxed_queries=int(d.no_strict_survivor.sum()), empty_queries=int(d.no_survivor.sum()))
            if name in truths:
                metrics[name].update(infer.ranking_metrics(final[name], truths[name]))
                ranks = formula_tables[name]
                matched = ranks.loc[ranks.formula.eq(ranks.molecule_id.map(true_formulas[name]))]
                metrics[name]['formula_top1'] = matched.loc[matched['rank'].le(1)].molecule_id.nunique()/len(ids[name])
                metrics[name]['formula_top3'] = matched.loc[matched['rank'].le(3)].molecule_id.nunique()/len(ids[name])
                comparisons.append(dict(split=name, method='gen', **metrics[name]))
                fp_path = infer.find_file(f'cand_fp_{name}.parquet', required=False)
                if fp_path:
                    fp = pd.read_parquet(fp_path)
                    fp['molecule_id'] = fp.molecule_id.astype(str)
                    order, ascending = ('rank', True) if 'rank' in fp else ('score', False)
                    fp = fp.sort_values(['molecule_id', order], ascending=[True, ascending])
                    fp = fp.groupby('molecule_id', sort=False).head(25).copy()
                    fp['ik14'] = fp.smiles.map(lambda s: infer.canonical_record(str(s)))
                    fp = fp.loc[fp.ik14.map(lambda r: r is not None and len(r) == 4)].copy()
                    fp['ik14'] = fp.ik14.map(lambda r: r[1])
                    comparisons.append(dict(split=name, method='fp',
                        **infer.ranking_metrics(fp, truths[name])))
                else:
                    print(f'FP comparison unavailable: cand_fp_{name}.parquet not attached')
        comparison = pd.DataFrame(comparisons)
        comparison.to_csv(OUT / 'gen_vs_fp_metrics.csv', index=False)
        print(comparison.to_string(index=False))
        print('Generator diagnostics:', json.dumps(metrics, indent=2))
    ''')
    md('''
    ## Submission and artifact checks
    Generator outputs cannot mathematically guarantee a mass-consistent survivor for
    every query. Both empty stages are flagged. If even 50 ppm yields no candidate,
    the notebook saves all candidates/diagnostics and an explicitly incomplete submission,
    then fails the final submission check. It never inserts aspirin, labels, or a
    mass-inconsistent guess. Stage 10 can combine those queries with retrieval candidates.
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        prediction = {mid: ';'.join(group.head(25).smiles)
                      for mid, group in final['test'].groupby('molecule_id', sort=False)}
        submission = pd.DataFrame(dict(molecule_id=ids['test'],
            smiles=[prediction.get(mid, '') for mid in ids['test']]))
        submission.to_csv(OUT / 'submission_gen.csv', index=False)
        valid, errors, warnings, submission_stats = evalkit.validate_submission(submission, ids['test'])
        reports = {name: [json.loads(p.read_text()) for p in OUT.glob(f'{name}_shard*.done.json')
                         if json.loads(p.read_text())['run_hash'] == run_hash] for name in ids}
        timing = {name: dict(queries=sum(r['queries'] for r in rs),
            worker_seconds=sum(r['seconds'] for r in rs),
            peak_gpu_gb=max((r['peak_gpu_gb'] for r in rs), default=0.)) for name, rs in reports.items()}
        shard_costs = [sum(json.loads(p.read_text())['seconds'] for p in OUT.glob(f'test_shard{s}_*.done.json')
            if json.loads(p.read_text())['run_hash'] == run_hash) for s in (0,1)]
        generator_cost = max(shard_costs) if gpu_count >= 2 else sum(shard_costs)
        runtime = dict(measured_launch_wall_seconds=measured_inference_seconds,
            total_notebook_seconds=time.perf_counter()-START, per_split=timing,
            projected_stage10_generator_seconds=generator_cost,
            projected_stage10_generator_gpu_hours=timing['test']['worker_seconds']/3600,
            exceeds_stage10_generator_budget=generator_cost > CFG['stage10_generator_budget_s'],
            note='Worker-based estimate excludes other stage-10 models; resume may reuse prior timings')
        print('Measured runtime and stage-10 component cost:', json.dumps(runtime, indent=2))
        (OUT / 'gen_metrics.json').write_text(json.dumps(dict(splits=metrics,
            selected=dict(alpha=float(best.alpha), formula_weight=float(best.formula_weight)),
            runtime=runtime, submission_valid=bool(valid), errors=errors), indent=2))
        heldout_hash = hashlib.sha256('\\n'.join(sorted(ids['val_model']+ids['val_rank'])).encode()).hexdigest()
        manifest = dict(stage='08_spec2smiles_infer', created=pd.Timestamp.now(tz='UTC').isoformat(),
            cfg=CFG, run_hash=run_hash, trained_on_split='train (upstream checkpoint)',
            heldout_ids_hash=heldout_hash, selected_on='val_model', reported_on='val_rank',
            conditioning='native predicted elements; formula prefix unsupported by Stage 07 v2',
            fp_spec=getattr(evalkit,'FP_SPEC',None), input_sha256=input_hashes,
            source_sha256=source_hashes, metrics=metrics, runtime=runtime,
            lib_versions=dict(torch=torch.__version__, rdkit=Chem.rdBase.rdkitVersion),
            submission_valid=bool(valid), files={p.name: dict(bytes=p.stat().st_size,
                sha256=infer.sha256(p)) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
        for name in ids:
            frame = final[name]
            assert {'molecule_id','smiles','ik14','score','count','source'} <= set(frame)
            assert frame.source.eq('gen').all() and frame['count'].ge(1).all()
            assert not frame.duplicated(['molecule_id','ik14']).any()
            assert frame.ppm_err.le(CFG['relax_ppm']).all()
        assert valid, f'Incomplete generator submission; see diagnostics and fuse retrieval in Stage 10: {errors}'
        print('Stage 08 complete:', OUT)
    ''')
    md('''
    ## Run settings and interpretation
    Use Kaggle **T4 ×2, Internet OFF**. Adjust `beam` / `num_return` together within
    25–50, or increase `batch_size` for throughput; OOM retries halve only the batch.
    `alpha_grid` and `formula_weight_grid` are selected exclusively on val_model.
    `formula_max_counts` expands the empirical enumeration bounds explicitly.
    Run times are workload-dependent; no real checkpoint/GPU benchmark is embedded.
    The val_model score is a tuning score; val_rank is reported separately.

    The cells above contain the complete embedded module source. The appendix below
    prints it for archival inspection, as required by the project notebook rules.
    ''')
    code('''
    if not CFG['SMOKE_TEST']:
        for module_name in ('spec2fp.py', 'spec2smiles.py', 'spec2smiles_infer.py'):
            print(f'===== {module_name} =====')
            print((WORK / module_name).read_text(encoding='utf-8'))
    ''')
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')))
    nbf.validate(nb)
    return nb


if __name__ == '__main__':
    target = ROOT / 'notebooks' / '08_spec2smiles_infer.ipynb'
    nbf.write(build(), target)
    print(target)
