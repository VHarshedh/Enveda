"""Build the self-contained Stage 07 v2 Kaggle notebook; every code cell is <59 lines."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
NOTEBOOKS = ROOT / 'notebooks'


def build_notebook():
    cells = []

    def md(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip()))

    def code(source):
        source = textwrap.dedent(source).strip()
        ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('!')))
        assert len(source.splitlines()) < 59, f'Cell has {len(source.splitlines())} lines (must be <59):\n{source}'
        cells.append(dict(cell_type='code', metadata={}, source=source, outputs=[], execution_count=None))

    md('''
    # Stage 07 v2 — Generative Spectrum-to-SMILES Transformer (5.5h Train / 11.4h Guard)
    Generates novel candidate SMILES alongside Stage 03/06 retrieval. Attach Stage 01 clean
    data, Stage 02 splits/evalkit, Stage 05b encoder weights, and an existing Stage 07
    checkpoint (`spec2smiles_model.pt` + `smiles_vocab.json`) if available.
    Select two T4 GPUs (`GPUs: 2`). Configured for a **5.5-hour training budget**
    (`TRAIN_BUDGET_S = 5.5 * 3600`), dedicating ~5.5 hours to training / sharpening fine-tuning
    (up to 6 epochs with cosine decay to 1e-6 on warm resume) and protecting inference with an
    emergency session time guard at 11.4 hours (`11.4 * 3600`) to guarantee beam search across
    all test, val_model, and val_rank molecules never truncates prematurely.
    Inference uses dual-GPU parallel bucket execution (`cuda:0` + `cuda:1`) with calibrated **beam_size = 14**.
    Spectra are balanced (<=4/mol train, <=2/mol val) to eliminate extreme replicate over-weighting.
    Generated candidates undergo early-stopping beam search (beam_size=14, top_k=25, archive=64, max_len=96),
    fast mass pre-filtering, two-stage Self-Spec2FP discriminative reranking, and speed-guarded canonicalization.
    ''')
    code('''
    !pip install -q rdkit
    import concurrent.futures, copy, gc, hashlib, json, math, os, random, shutil, sys, time
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

    def find_file(filename, root=Path('/kaggle/input'), hint='', required=True):
        hits = list(Path(root).rglob(filename))
        if not hits:
            if not required:
                return None
            raise FileNotFoundError(f'{filename} not found under {root}')
        hits.sort(key=lambda p: (bool(hint and hint in str(p)), p.stat().st_mtime, str(p)),
                  reverse=True)
        return hits[0]
    ''')
    # Literal source fragments are AST-valid Python; no %%writefile magic cells.
    for name in ('spec2fp.py', 'spec2smiles.py', 'evalkit.py'):
        source_path = (SRC / name) if (SRC / name).is_file() else (ROOT / name)
        source = source_path.read_text(encoding='utf-8')
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
    spec2smiles_ckpt = find_file('spec2smiles_model.pt', required=False)
    spec2smiles_vocab = find_file('smiles_vocab.json', required=False)
    RESUME_STAGE07 = (spec2smiles_ckpt is not None) and (spec2smiles_vocab is not None)
    print(f'RESUME_STAGE07: {RESUME_STAGE07} | ckpt: {spec2smiles_ckpt} | vocab: {spec2smiles_vocab}')
    cfg = dict(encoder=upstream['model'], max_length=128, decoder_layers=4,
               d_model=320, nhead=8, dim_feedforward=1280, dropout=0.1,
               norm_first=True, top_k_peaks=128, mz_dim=64,
               epochs=6 if RESUME_STAGE07 else 12,
               time_budget_s=5.5 * 3600,
               lr_enc=8e-6 if RESUME_STAGE07 else 8e-5,
               lr_dec=3.5e-5 if RESUME_STAGE07 else 4e-4,
               batch_size=256,
               warmup_fraction=0.10 if RESUME_STAGE07 else 0.05,
               ema_decay=0.999,
               label_smoothing=0.015 if RESUME_STAGE07 else 0.05,
               aug_prob=0.25 if RESUME_STAGE07 else 0.65,
               seed=42, element_conditioning='predicted',
               elements_order=spec2fp.ELEMENTS_ORDER)
    cfg['top_k_peaks'] = upstream.get('training', {}).get('TOP_K_PEAKS', 128)
    cfg['mz_dim'] = (cfg['encoder'].get('feat_dim', 129)-1)//2
    for key, actual in [('adduct_vocab', spec2fp.ADDUCT_VOCAB),
                        ('inst_vocab', spec2fp.INST_VOCAB),
                        ('elements_order', spec2fp.ELEMENTS_ORDER)]:
        assert upstream.get(key, actual) == actual, f'Checkpoint vocabulary mismatch: {key}'
    ''')
    md('''
    Vocabulary uses training molecules only. When warm-resuming from an attached Stage 07
    checkpoint (`RESUME_STAGE07 = True`), `smiles_vocab.json` is loaded directly to preserve
    100% token-index alignment. Target arrays and K-variant SMILES augmentations are prepared in RAM.
    ''')
    code('''
    meta = pd.read_parquet(train_path, columns=['molecule_id', 'smiles'], use_threads=False)
    meta['molecule_id'] = meta.molecule_id.astype(str)
    smi_counts = meta.dropna(subset=['smiles']).groupby(['molecule_id', 'smiles'], sort=False).size().reset_index(name='cnt')
    meta = smi_counts.sort_values(['molecule_id', 'cnt'], ascending=[True, False]).drop_duplicates('molecule_id')[['molecule_id', 'smiles']].copy()
    del smi_counts
    meta['split3'] = meta.molecule_id.map(split_map)
    assert meta.split3.notna().all(), 'Missing split assignments'
    if RESUME_STAGE07:
        tokenizer = s2s.SmilesTokenizer.load(spec2smiles_vocab)
        shutil.copy2(spec2smiles_vocab, OUT / 'smiles_vocab.json')
        print(f'Loaded smiles_vocab.json from {spec2smiles_vocab} (vocab: {len(tokenizer.vocab)})')
    else:
        train_smiles, probe = set(), s2s.SmilesTokenizer()
        for smi in meta.loc[meta.split3.eq('train'), 'smiles'].dropna().unique():
            mol = Chem.MolFromSmiles(smi)
            if mol and not any(a.GetSymbol() not in spec2fp.ELEMENTS_ORDER for a in mol.GetAtoms()):
                c_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                try:
                    probe.encode(c_smi)
                    train_smiles.add(c_smi)
                except ValueError:
                    pass
        tokenizer = s2s.SmilesTokenizer().fit(sorted(train_smiles))
        tokenizer.save(OUT / 'smiles_vocab.json')
        del train_smiles, probe
    canonical = {}
    for smi in meta.smiles.dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol and not any(a.GetSymbol() not in spec2fp.ELEMENTS_ORDER for a in mol.GetAtoms()):
            c_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            try:
                enc = tokenizer.encode(c_smi)
                if not RESUME_STAGE07 or 3 not in enc:
                    canonical[smi] = c_smi
            except ValueError:
                pass
    ''')
    code('''
    smiles = sorted(set(canonical.values()))
    smi_index = {s: i for i, s in enumerate(smiles)}
    mol_to_canon = {r.molecule_id: canonical[r.smiles] for r in meta.itertuples() if r.smiles in canonical}
    target_map = {mid: smi_index[c_smi] for mid, c_smi in mol_to_canon.items()}
    dtype = np.uint8 if len(tokenizer.vocab) <= 256 else np.int16
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
    del canonical, smi_index, mol_to_canon
    gc.collect()
    print('Vocabulary / target MB / RSS GB:', cfg['vocab_size'], tokens.nbytes/1e6, check_ram())
    ''')
    code('''
    paths = {name: CACHE / f'{name}.bin' for name in ('train', 'val_model')}
    streams = {name: path.open('wb') for name, path in paths.items()}
    train_mol_counts, val_mol_counts = {}, {}
    max_train_per_mol = 4 if RESUME_STAGE07 else 999999
    max_val_per_mol = 2 if RESUME_STAGE07 else 999999
    try:
        for record_batch in pq.ParquetFile(train_path).iter_batches(batch_size=4096):
            train_df = record_batch.to_pandas()
            train_df['molecule_id'] = train_df.molecule_id.astype(str)
            train_df['split3'] = train_df.molecule_id.map(split_map)
            train_df['ce'] = pd.to_numeric(train_df.ce, errors='coerce').fillna(0)
            train_sub = train_df.loc[train_df.split3.eq('train') & train_df.molecule_id.isin(target_map)]
            if len(train_sub) > 0 and RESUME_STAGE07:
                keep = []
                for mid in train_sub.molecule_id:
                    c = train_mol_counts.get(mid, 0)
                    if c < max_train_per_mol:
                        train_mol_counts[mid] = c + 1
                        keep.append(True)
                    else:
                        keep.append(False)
                train_sub = train_sub.iloc[keep]
            val_sub = train_df.loc[train_df.split3.eq('val_model') & train_df.molecule_id.isin(target_map)]
            if len(val_sub) > 0 and RESUME_STAGE07:
                keep = []
                for mid in val_sub.molecule_id:
                    c = val_mol_counts.get(mid, 0)
                    if c < max_val_per_mol:
                        val_mol_counts[mid] = c + 1
                        keep.append(True)
                    else:
                        keep.append(False)
                val_sub = val_sub.iloc[keep]
            del train_df
            s2s.append_records(train_sub, streams['train'], cfg, target_map)
            s2s.append_records(val_sub, streams['val_model'], cfg, target_map)
            del train_sub, val_sub, record_batch
            check_ram()
    finally:
        for stream in streams.values():
            stream.close()
    del train_mol_counts, val_mol_counts
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
    SKIP_FINETUNE = False
    TOTAL_BUDGET_S = cfg.get('time_budget_s', 5.5 * 3600)
    TRAIN_BUDGET_S = 5.5 * 3600
    model = s2s.Spec2SmilesModel(cfg)
    model.warm_start(encoder_path)
    best, best_epoch = float('inf'), 0
    if RESUME_STAGE07:
        ckpt = torch.load(spec2smiles_ckpt, map_location='cpu', weights_only=False)
        state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        best = float(ckpt.get('val_loss', 0.75130))
        best_epoch = int(ckpt.get('epoch', 8))
        shutil.copy2(spec2smiles_ckpt, OUT / 'spec2smiles_model.pt')
        print(f'Warm-resumed from {spec2smiles_ckpt} (best_val_loss={best:.5f})')
        del ckpt, state
    model.to(device)
    enc_params = model.encoder_parameters()
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    if RESUME_STAGE07:
        for param in enc_params:
            param.requires_grad_(True)
    optimizer = torch.optim.AdamW([{'params': enc_params, 'lr': cfg['lr_enc']},
                                  {'params': dec_params, 'lr': cfg['lr_dec']}], weight_decay=0.01)
    epochs = 0 if SKIP_FINETUNE else cfg['epochs']
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * cfg['warmup_fraction']))
    min_ratio = 1e-6 / max(cfg['lr_dec'], 1e-6)
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(min_ratio, min_ratio + (1.0 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = s2s.EMA(model, decay=cfg['ema_decay'])
    parallel = nn.DataParallel(model) if torch.cuda.device_count() >= 2 else model
    history = []
    if RESUME_STAGE07:
        best = s2s.run_epoch(ema.model, val_loader, device, label_smoothing=0.05)
        print(f'Evaluated apples-to-apples baseline val_loss on balanced val_loader: {best:.5f}')
    ''')
    code('''
    for epoch in range(epochs):
        ep_start = time.time()
        if not RESUME_STAGE07:
            for param in enc_params:
                param.requires_grad_(epoch >= 1)
        train_loss = s2s.run_epoch(parallel, train_loader, device, optimizer, scaler, ema, scheduler,
                                   label_smoothing=cfg['label_smoothing'])
        val_loss = s2s.run_epoch(ema.model, val_loader, device, label_smoothing=0.05)
        ep_time = time.time() - ep_start
        elapsed = time.time() - START
        history.append(dict(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss,
                            ep_seconds=round(ep_time, 1), elapsed_seconds=round(elapsed, 1),
                            lrs=[g['lr'] for g in optimizer.param_groups], rss_gb=check_ram()))
        if val_loss < best:
            best = val_loss
            best_epoch = epoch + 1
            torch.save({'model_state_dict': ema.model.state_dict(), 'epoch': best_epoch, 'val_loss': best},
                       OUT / 'spec2smiles_model.pt')
        print(history[-1], flush=True)
        if elapsed + ep_time * 1.15 > TRAIN_BUDGET_S and (epoch + 1) >= (1 if RESUME_STAGE07 else 4):
            print(f'Time guard reached at epoch {epoch+1}; proceeding to beam search.')
            break
    (OUT / 'spec2smiles_config.json').write_text(json.dumps(cfg, indent=2))
    del parallel, optimizer, scheduler, scaler, ema, train_loader, val_loader
    del ds_train, ds_val, tokens, tokens_aug, smiles, enc_params, dec_params
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = torch.load(OUT / 'spec2smiles_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', best_epoch)
    del checkpoint
    model.eval()
    ''')
    md('''
    Cross-attend uncorrupted fragment peak embeddings across each molecule's spectra,
    then decode with dual-GPU parallel bucket execution (`cuda:0` + `cuda:1`).
    Hypotheses undergo early-stopping beam search (beam_size=14, max_len=96, archive=64),
    raw SMILES string deduplication, fast mass pre-filter, top-28 Self-Spec2FP discriminative
    reranking, and tautomer-canonical deduplication. Test split runs first for immediate submission validation.
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
    n_gpus = torch.cuda.device_count()
    dev0 = torch.device('cuda:0' if n_gpus >= 1 else 'cpu')
    dev1 = torch.device('cuda:1') if n_gpus >= 2 else None
    model_0 = copy.deepcopy(model).to(dev0).eval()
    model_1 = copy.deepcopy(model).to(dev1).eval() if dev1 is not None else None
    print(f'Inference ready: {n_gpus} GPU(s) allocated (model_0: {dev0}, model_1: {dev1})')
    float_cols = {'score', 'log_prob', 'nb_ll', 'cos_sim', 'tan_sim', 'maccs_cos', 'maccs_tan', 'elem_dist', 'ppm_err'}
    schema = pa.schema([(c, pa.int64() if c == 'rank' else pa.float64() if c in float_cols else pa.string()) for c in s2s.COLUMNS])

    def process_bucket(part_path, mod, dev):
        query = pd.read_parquet(part_path, columns=required, use_threads=False)
        if len(query) == 0:
            return pd.DataFrame(columns=s2s.COLUMNS), 0
        query['molecule_id'] = query.molecule_id.astype(str)
        query['ce'] = pd.to_numeric(query.ce, errors='coerce').fillna(0)
        q_count = query.molecule_id.nunique()
        cand = s2s.generate_candidates(mod, tokenizer, query, dev, beam_size=14, top_k=25,
                                      mol_batch_size=64, max_len=96)
        return cand, q_count
    ''')
    code('''
    summaries, sub_stats = {}, {}
    for name in ('test', 'val_model', 'val_rank'):
        truth = {}
        if name != 'test':
            truth_frame = meta.loc[meta.split3.eq(name)].drop_duplicates('molecule_id')
            truth = {r.molecule_id: evalkit.canon_ik14(r.smiles) for r in truth_frame.itertuples()}
        rows, covered, reciprocal, hits, queries = 0, 0, 0.0, 0, 0
        parts, n_parts = partitions[name], len(partitions[name])
        with pq.ParquetWriter(OUT / f'cand_gen_{name}.parquet', schema, compression='zstd') as writer:
            for b_idx in range(0, n_parts, 2):
                if time.time() - START > 11.4 * 3600:
                    print(f'[{name}] Session time guard (11.4h) reached; finalizing early.', flush=True)
                    break
                p0 = parts[b_idx]
                p1 = parts[b_idx + 1] if b_idx + 1 < n_parts else None
                if p1 is not None and model_1 is not None:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        f0 = executor.submit(process_bucket, p0, model_0, dev0)
                        f1 = executor.submit(process_bucket, p1, model_1, dev1)
                        pair_results = [f0.result(), f1.result()]
                else:
                    pair_results = [process_bucket(p0, model_0, dev0)]
                    if p1 is not None:
                        pair_results.append(process_bucket(p1, model_0, dev0))
                for candidates, q_cnt in pair_results:
                    queries += q_cnt
                    if len(candidates) > 0:
                        writer.write_table(pa.Table.from_pandas(candidates, schema=schema, preserve_index=False))
                        rows += len(candidates)
                        covered += candidates.molecule_id.nunique()
                        if truth:
                            for qid, frame in candidates.groupby('molecule_id'):
                                t_ik = truth.get(str(qid))
                                if t_ik and len(frame.loc[frame.ik14.eq(t_ik), 'rank']):
                                    reciprocal += 1.0 / int(frame.loc[frame.ik14.eq(t_ik), 'rank'].min())
                                    hits += 1
                cur_p = min(b_idx + 2, n_parts)
                if cur_p % 8 == 0 or cur_p == n_parts:
                    print(f'[{name}] {cur_p}/{n_parts} partitions | elapsed={(time.time()-START)/60:.1f}m | mrr25={reciprocal/max(queries,1):.4f}', flush=True)
                    check_ram()
        summaries[name] = dict(queries=queries, candidates=rows, covered=covered, empty_queries=queries-covered)
        if name != 'test':
            summaries[name].update(mrr25=reciprocal/max(queries, 1), recall25=hits/max(queries, 1))
        print(name, summaries[name], flush=True)
        if name == 'test':
            test_cand = pd.read_parquet(OUT / 'cand_gen_test.parquet')
            test_mol_ids = pd.read_parquet(test_path, columns=['molecule_id']).molecule_id.astype(str).unique().tolist()
            test_preds = {str(qid): ';'.join(grp.sort_values('rank').head(25)['smiles']) for qid, grp in test_cand.groupby('molecule_id', sort=False)}
            fallback_smi = 'CC(=O)Oc1ccccc1C(=O)O'
            sub_df = pd.DataFrame({'molecule_id': test_mol_ids, 'smiles': [test_preds.get(mid) or fallback_smi for mid in test_mol_ids]})
            is_valid, sub_errors, sub_warnings, sub_stats = evalkit.validate_submission(sub_df, test_mol_ids)
            assert is_valid, f'Invalid submission_spec2smiles.csv: {sub_errors}'
            sub_df.to_csv(OUT / 'submission_spec2smiles.csv', index=False)
            print('Test submission validated & written immediately:', sub_stats, flush=True)
            del test_cand, sub_df
            gc.collect()
    ''')
    code('''
    metrics = dict(best_epoch=best_epoch, best_val_loss=best, history=history,
                   splits=summaries, selection='val_model smoothed cross entropy',
                   generation='beam14, max_len96, mol_batch64, top25, dual-gpu parallel, self-spec2fp reranked')
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
        cell['id'] = f'stage07v2-{i:03d}'
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python', version='3.11')))


if __name__ == '__main__':
    notebook = build_notebook()
    path = NOTEBOOKS / '07_spec2smiles_train_v2.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Generated {path} ({len(notebook["cells"])} cells)')

