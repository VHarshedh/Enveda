"""Build the self-contained Stage 07 Kaggle notebook; every code cell is <59 lines."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def md(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip()))

    def code(source):
        source = textwrap.dedent(source).strip()
        ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('!')))
        assert len(source.splitlines()) < 59
        cells.append(dict(cell_type='code', metadata={}, source=source, outputs=[], execution_count=None))

    md('''
    # Stage 07 — Generative Spectrum-to-SMILES Transformer
    Generates novel structures alongside Stage 03/06 retrieval. Attach Stage 01 clean
    data, Stage 02 splits/evalkit, and a **train-only** Stage 05/05b checkpoint.
    Select two T4 GPUs. Trains for 10 epochs under a 10h total wall-clock budget (reserving
    1.2h for batched beam search and submission validation); epoch one freezes the transferred encoder.
    Predicted 180-dim substructure features (element counts, MACCS keys, heavy atom & ring counts)
    condition both training and inference. Generated beams undergo Self-Spec2FP discriminative
    reranking (Morgan & MACCS Tanimoto, elemental L1, ppm error) and tautomer-canonical deduplication.
    Raw spectra are streamed to compact disk-backed records; K=4 randomized SMILES variants.
    ''')
    code('''
    !pip install -q rdkit
    import gc, hashlib, json, os, random, shutil, sys, time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    import torch
    from torch import nn
    from rdkit import Chem, RDLogger
    WORK = Path('/kaggle/working')
    OUT = WORK / 'artifacts/spec2smiles'
    CACHE = WORK / 'spec2smiles_cache'
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORK))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    RDLogger.DisableLog('rdApp.*')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'Training requires a Kaggle GPU accelerator'
    print('GPUs:', torch.cuda.device_count())
    START = time.time()

    def check_ram():
        rss = psutil.Process().memory_info().rss / 1e9
        if rss >= 24.0:
            raise MemoryError(f'Host RSS {rss:.2f} GB exceeds 24 GB budget')
        return rss

    def find_file(filename, root=Path('/kaggle/input'), hint=''):
        hits = list(Path(root).rglob(filename))
        if not hits:
            raise FileNotFoundError(f'{filename} not found under {root}')
        hits.sort(key=lambda p: (bool(hint and hint in str(p)), p.stat().st_mtime, str(p)),
                  reverse=True)
        return hits[0]
    ''')
    # Literal source fragments are AST-valid Python; no %%writefile magic cells.
    for name in ('spec2fp.py', 'spec2smiles.py', 'evalkit.py'):
        source = (ROOT / name).read_text(encoding='utf-8')
        ast.parse(source)
        code(f"module_path = WORK / {name!r}\nmodule_path.write_text('', encoding='utf-8')")
        lines = source.splitlines(keepends=True)
        for start in range(0, len(lines), 42):
            literals = '\n'.join(repr(line) for line in lines[start:start+42])
            code("fragment = (\n" + literals + "\n)\nwith module_path.open('a', encoding='utf-8') as f:\n    f.write(fragment)")
    code('''
    encoder_path = find_file('spec2fp_model.pt')
    encoder_config_path = find_file('spec2fp_config.json', encoder_path.parent)
    upstream = json.loads(encoder_config_path.read_text())
    if upstream.get('training', {}).get('FULL_TRAIN', False):
        raise ValueError('Full-train encoder contaminates validation; attach train-only weights')
    import importlib
    for name in ('evalkit', 'spec2fp', 'spec2smiles'):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    import evalkit, spec2fp, spec2smiles as s2s
    train_path = find_file('train_clean.parquet')
    test_path = find_file('test_clean.parquet')
    split_path = find_file('splits.parquet')
    splits = pd.read_parquet(split_path, columns=['molecule_id', 'split3'])
    splits['molecule_id'] = splits.molecule_id.astype(str)
    assert not splits.molecule_id.duplicated().any()
    assert set(splits.split3) == {'train', 'val_model', 'val_rank'}
    split_map = splits.set_index('molecule_id').split3
    cfg = dict(encoder=upstream['model'], max_length=128, decoder_layers=4,
               d_model=320, nhead=8, dim_feedforward=1280, dropout=0.1,
               norm_first=True, top_k_peaks=128, mz_dim=64, epochs=10,
               time_budget_s=10.0 * 3600,
               lr_enc=8e-5, lr_dec=4e-4, batch_size=256, warmup_fraction=0.05, ema_decay=0.999,
               label_smoothing=0.05, seed=42, element_conditioning='predicted',
               elements_order=spec2fp.ELEMENTS_ORDER)
    cfg['top_k_peaks'] = upstream.get('training', {}).get('TOP_K_PEAKS', 128)
    cfg['mz_dim'] = (cfg['encoder'].get('feat_dim', 129)-1)//2
    for key, actual in [('adduct_vocab', spec2fp.ADDUCT_VOCAB),
                        ('inst_vocab', spec2fp.INST_VOCAB),
                        ('elements_order', spec2fp.ELEMENTS_ORDER)]:
        assert upstream.get(key, actual) == actual, f'Checkpoint vocabulary mismatch: {key}'
    ''')
    md('''
    Vocabulary uses training molecules only. Invalid or >126-token targets are
    excluded rather than truncated; validation generation still covers every molecule.
    Randomized SMILES that exceed the budget or vocabulary fall back to canonical targets.
    Vocabulary size is data dependent (bracketed atoms are individual tokens).
    ''')
    code('''
    meta = pd.read_parquet(train_path, columns=['molecule_id', 'smiles'], use_threads=False)
    meta['molecule_id'] = meta.molecule_id.astype(str)
    smi_counts = meta.dropna(subset=['smiles']).groupby(['molecule_id', 'smiles'], sort=False).size().reset_index(name='cnt')
    meta = smi_counts.sort_values(['molecule_id', 'cnt'], ascending=[True, False]).drop_duplicates('molecule_id')[['molecule_id', 'smiles']].copy()
    del smi_counts
    meta['split3'] = meta.molecule_id.map(split_map)
    assert meta.split3.notna().all(), 'Missing split assignments'
    canonical = {}
    probe = s2s.SmilesTokenizer()
    for smi in meta.smiles.dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or any(a.GetSymbol() not in spec2fp.ELEMENTS_ORDER for a in mol.GetAtoms()):
            continue
        value = Chem.MolToSmiles(mol, isomericSmiles=False)
        try:
            probe.encode(value)
            canonical[smi] = value
        except ValueError:
            pass
    train_smiles = {canonical[s] for s in meta.loc[meta.split3.eq('train'), 'smiles'] if s in canonical}
    tokenizer = s2s.SmilesTokenizer().fit(sorted(train_smiles))
    tokenizer.save(OUT / 'smiles_vocab.json')
    smiles = sorted(set(canonical.values()))
    smi_index = {s: i for i, s in enumerate(smiles)}
    mol_to_canon = {r.molecule_id: canonical[r.smiles] for r in meta.itertuples() if r.smiles in canonical}
    target_map = {mid: smi_index[c_smi] for mid, c_smi in mol_to_canon.items()}
    dtype = np.uint8 if len(tokenizer.vocab) <= 256 else np.int16
    assert len(tokenizer.vocab) < 32768
    tokens = np.asarray([tokenizer.encode(s) for s in smiles], dtype=dtype)
    num_aug = 4
    tokens_aug = np.repeat(tokens[None, :, :], num_aug, axis=0)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        for k in range(num_aug):
            randomized = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=False)
            try:
                encoded = tokenizer.encode(randomized)
                if 3 not in encoded:
                    tokens_aug[k, i] = encoded
            except ValueError:
                pass
    cfg['vocab_size'] = len(tokenizer.vocab)
    cfg['target_shape'] = list(tokens.shape)
    cfg['target_bytes'] = tokens.nbytes + tokens_aug.nbytes
    cfg['excluded_unique_smiles'] = int(meta.smiles.nunique()-len(canonical))
    del canonical, smi_index, train_smiles, probe, mol_to_canon
    gc.collect()
    print('Vocabulary / target MB / RSS GB:', cfg['vocab_size'], tokens.nbytes/1e6, check_ram())
    ''')
    code('''
    paths = {name: CACHE / f'{name}.bin' for name in ('train', 'val_model')}
    streams = {name: path.open('wb') for name, path in paths.items()}
    try:
        for record_batch in pq.ParquetFile(train_path).iter_batches(batch_size=2048):
            train_df = record_batch.to_pandas()
            train_df['molecule_id'] = train_df.molecule_id.astype(str)
            train_df['split3'] = train_df.molecule_id.map(split_map)
            train_df['ce'] = pd.to_numeric(train_df.ce, errors='coerce').fillna(0)
            train_split = train_df.loc[train_df.split3.eq('train') & train_df.molecule_id.isin(target_map)]
            val_split = train_df.loc[train_df.split3.eq('val_model') & train_df.molecule_id.isin(target_map)]
            del train_df
            s2s.append_records(train_split, streams['train'], cfg, target_map)
            s2s.append_records(val_split, streams['val_model'], cfg, target_map)
            del train_split, val_split, record_batch
            check_ram()
    finally:
        for stream in streams.values():
            stream.close()
    ds_train = s2s.SpectrumDataset(paths['train'], cfg, tokenizer, smiles, tokens, augment=True, tokens_aug=tokens_aug)
    ds_val = s2s.SpectrumDataset(paths['val_model'], cfg, tokenizer, smiles, tokens)
    workers = 2 if torch.cuda.is_available() else 0
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=workers, pin_memory=bool(workers), persistent_workers=bool(workers), collate_fn=ds_train.collate)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=workers, pin_memory=bool(workers), persistent_workers=bool(workers), collate_fn=ds_val.collate)
    del target_map
    gc.collect()
    print('Training / validation spectra:', len(ds_train), len(ds_val))
    ''')
    code('''
    cfg['epochs'] = 10
    TOTAL_BUDGET_S = 10.0 * 3600
    INFER_RESERVE_S = 1.2 * 3600
    TRAIN_BUDGET_S = TOTAL_BUDGET_S - INFER_RESERVE_S
    model = s2s.Spec2SmilesModel(cfg)
    model.warm_start(encoder_path)
    model.to(device)
    enc_params = model.encoder_parameters()
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': enc_params, 'lr': cfg['lr_enc']},
                                  {'params': dec_params, 'lr': cfg['lr_dec']}], weight_decay=0.01)
    total_steps = cfg['epochs'] * len(train_loader)
    warmup_steps = max(1, int(total_steps * cfg['warmup_fraction']))
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = s2s.EMA(model, decay=cfg['ema_decay'])
    parallel = nn.DataParallel(model) if torch.cuda.device_count() >= 2 else model
    history, best = [], float('inf')
    for epoch in range(cfg['epochs']):
        ep_start = time.time()
        for param in enc_params:
            param.requires_grad_(epoch >= 1)
        train_loss = s2s.run_epoch(parallel, train_loader, device, optimizer, scaler, ema, scheduler)
        val_loss = s2s.run_epoch(ema.model, val_loader, device)
        ep_time = time.time() - ep_start
        elapsed = time.time() - START
        history.append(dict(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss,
                            ep_seconds=round(ep_time, 1), elapsed_seconds=round(elapsed, 1),
                            lrs=[g['lr'] for g in optimizer.param_groups], rss_gb=check_ram()))
        if val_loss < best:
            best = val_loss
            torch.save({'model_state_dict': ema.model.state_dict(), 'epoch': epoch+1, 'val_loss': best},
                       OUT / 'spec2smiles_model.pt')
        print(history[-1], flush=True)
        if elapsed + ep_time * 1.15 > TRAIN_BUDGET_S and (epoch + 1) >= 4:
            print(f'Time guard reached at epoch {epoch+1} ({elapsed/3600:.2f}h elapsed); proceeding to beam search.')
            break
    (OUT / 'spec2smiles_config.json').write_text(json.dumps(cfg, indent=2))
    del parallel, optimizer, scheduler, scaler, ema, train_loader, val_loader
    del ds_train, ds_val, tokens, tokens_aug, smiles, enc_params, dec_params
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = torch.load(OUT / 'spec2smiles_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    del checkpoint
    model.eval()
    ''')
    md('''
    Cross-attend uncorrupted fragment peak embeddings across each molecule's spectra,
    then decode 64 molecules together with beam width 16 and a 110-token budget. Completed
    paths are scored with Self-Spec2FP discriminative reranking (Morgan & MACCS Tanimoto,
    elemental L1 distance, ppm mass error) followed by tautomer-canonical deduplication.
    Partition each source Parquet in one streaming pass; all spectra for an ID share one partition.
    ''')
    code('''
    required = ['molecule_id', 'mz', 'inten', 'precursor_mz', 'neutral_mass',
                'adduct', 'ce', 'instrument', 'polarity']
    partitions = {name: [] for name in ('val_model', 'val_rank', 'test')}
    for source_path, is_test in ((train_path, False), (test_path, True)):
        writers = {}
        try:
            for rb in pq.ParquetFile(source_path).iter_batches(batch_size=8192, columns=required):
                frame = rb.to_pandas()
                frame['molecule_id'] = frame.molecule_id.astype(str)
                frame['ce'] = pd.to_numeric(frame.ce, errors='coerce').fillna(0)
                labels = pd.Series('test', index=frame.index) if is_test else frame.molecule_id.map(split_map)
                buckets = pd.util.hash_pandas_object(frame.molecule_id, index=False) % 64
                for (name, bucket), group in frame.groupby([labels, buckets], sort=False):
                    if name not in partitions:
                        continue
                    key = (name, int(bucket))
                    table = pa.Table.from_pandas(group, preserve_index=False)
                    if key not in writers:
                        part = CACHE / f'query_{name}_{bucket}.parquet'
                        writers[key] = pq.ParquetWriter(part, table.schema, compression='zstd')
                        partitions[name].append(part)
                    writers[key].write_table(table)
                check_ram()
        finally:
            for writer in writers.values():
                writer.close()
    del frame, group, table, rb
    gc.collect()
    ''')
    code('''
    float_cols = {'score', 'log_prob', 'nb_ll', 'cos_sim', 'tan_sim', 'maccs_cos', 'maccs_tan', 'elem_dist', 'ppm_err'}
    schema = pa.schema([(c, pa.int64() if c == 'rank' else pa.float64()
                         if c in float_cols else pa.string()) for c in s2s.COLUMNS])
    summaries = {}
    required = ['molecule_id', 'mz', 'inten', 'precursor_mz', 'neutral_mass',
                'adduct', 'ce', 'instrument', 'polarity']
    for name in ('val_model', 'val_rank', 'test'):
        truth = {}
        if name != 'test':
            truth_frame = meta.loc[meta.split3.eq(name)].drop_duplicates('molecule_id')
            truth = {r.molecule_id: evalkit.canon_ik14(r.smiles) for r in truth_frame.itertuples()}
        rows, covered, reciprocal, hits, queries = 0, 0, 0.0, 0, 0
        with pq.ParquetWriter(OUT / f'cand_gen_{name}.parquet', schema, compression='zstd') as writer:
            for part in partitions[name]:
                query = pd.read_parquet(part, columns=required, use_threads=False)
                queries += query.molecule_id.nunique()
                query['molecule_id'] = query.molecule_id.astype(str)
                query['ce'] = pd.to_numeric(query.ce, errors='coerce').fillna(0)
                candidates = s2s.generate_candidates(model, tokenizer, query, device, beam_size=16, top_k=25, mol_batch_size=64, max_len=124)
                writer.write_table(pa.Table.from_pandas(candidates, schema=schema, preserve_index=False))
                rows += len(candidates)
                covered += candidates.molecule_id.nunique()
                if truth:
                    for qid, frame in candidates.groupby('molecule_id'):
                        t_ik = truth.get(str(qid))
                        if t_ik:
                            matched = frame.loc[frame.ik14.eq(t_ik), 'rank']
                            if len(matched):
                                reciprocal += 1/int(matched.min())
                                hits += 1
                del query, candidates
                gc.collect()
                check_ram()
        summaries[name] = dict(queries=queries, candidates=rows, covered=covered,
                               empty_queries=queries-covered)
        if name != 'test':
            summaries[name].update(mrr25=reciprocal/max(queries, 1), recall25=hits/max(queries, 1))
        print(name, summaries[name])
    test_cand = pd.read_parquet(OUT / 'cand_gen_test.parquet')
    test_mol_ids = pd.read_parquet(test_path, columns=['molecule_id']).molecule_id.astype(str).unique().tolist()
    test_preds = {str(qid): ';'.join(grp.sort_values('rank').head(25)['smiles']) for qid, grp in test_cand.groupby('molecule_id', sort=False)}
    fallback_smi = 'CC(=O)Oc1ccccc1C(=O)O'
    sub_df = pd.DataFrame({'molecule_id': test_mol_ids, 'smiles': [test_preds.get(mid) or fallback_smi for mid in test_mol_ids]})
    is_valid, sub_errors, sub_warnings, sub_stats = evalkit.validate_submission(sub_df, test_mol_ids)
    assert is_valid, f'Invalid submission_spec2smiles.csv: {sub_errors}'
    sub_df.to_csv(OUT / 'submission_spec2smiles.csv', index=False)
    print('Standalone submission validation:', sub_stats)
    ''')
    code('''
    metrics = dict(best_epoch=best_epoch, best_val_loss=best, history=history,
                   splits=summaries, selection='val_model smoothed cross entropy',
                   generation='beam16, max_len124, mol_batch64, top25, self-spec2fp reranked')
    (OUT / 'spec2smiles_val_metrics.json').write_text(json.dumps(metrics, indent=2))
    for name in ('spec2smiles.py', 'spec2fp.py', 'evalkit.py'):
        shutil.copy2(WORK / name, OUT / name)

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()

    vm_ids = splits.loc[splits.split3.eq('val_model'), 'molecule_id'].tolist()
    vr_ids = splits.loc[splits.split3.eq('val_rank'), 'molecule_id'].tolist()
    manifest = dict(heldout_ids_hash=hashlib.sha256(''.join(sorted(vm_ids + vr_ids)).encode()).hexdigest(),
                    stage='spec2smiles', trained_on_split='train', selected_on='val_model',
                    untouched_evaluation='val_rank', seed=42, elapsed_seconds=time.time()-START,
                    encoder=str(encoder_path), encoder_sha256=sha256(encoder_path),
                    encoder_config_sha256=sha256(encoder_config_path),
                    split_sha256=sha256(split_path), torch_version=torch.__version__,
                    rdkit_version=Chem.rdBase.rdkitVersion, summaries=summaries,
                    submission_validation=sub_stats,
                    files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                           for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Stage 07 complete:', OUT)
    ''')
    for i, cell in enumerate(cells):
        cell['id'] = f'stage07-{i:03d}'
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python', version='3.11')))


if __name__ == '__main__':
    notebook = build_notebook()
    path = ROOT / '07_spec2smiles_train.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Generated {path} ({len(notebook["cells"])} cells)')
