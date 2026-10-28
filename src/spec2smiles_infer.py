"""Stage 08 inference runtime, embedded verbatim into the generated notebook."""
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ELEMENTS = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
EXTRA = ['Na', 'K', 'Si', 'B', 'Se']
PT = Chem.GetPeriodicTable()
MASS = {e: PT.GetMostCommonIsotopeMass(e) for e in ELEMENTS + EXTRA}
NOMINAL = {e: round(MASS[e]) for e in MASS}
ADDUCTS = {'[M+H]+': (1, 1.007276), '[M-H]-': (1, -1.007276),
    '[M+Na]+': (1, 22.989218), '[M+K]+': (1, 38.963158),
    '[M+NH4]+': (1, 18.033823), '[M+Cl]-': (1, 34.969402),
    '[M+2H]2+': (2, 2.014552), '[M-2H]2-': (2, -2.014552),
    '[M+H-H2O]+': (1, -17.003289), '[M-H2O+H]+': (1, -17.003289),
    '[M-H2O-H]-': (1, -19.017841), '[M+FA-H]-': (1, 44.998201),
    '[M+CH2O2-H]-': (1, 44.998201), '[M+HAc-H]-': (1, 59.013851),
    '[M+2Na-H]+': (1, 44.971160), '[M]+': (1, -0.000549)}
RAW_COLS = ['molecule_id', 'smiles', 'ik14', 'log_prob', 'mean_log_prob',
            'count', 'formula_score', 'ppm_err', 'mass_relaxed']


def find_file(pattern, hint='', required=True, roots=None):
    roots = roots or ['/kaggle/input', '/kaggle/working']
    if pattern == 'evalkit.py':
        path = Path('/kaggle/working/evalkit.py')
        if not path.is_file() and required:
            raise FileNotFoundError(f'Missing {path}; modules are not imported from input datasets')
        return path if path.is_file() else None
    hits = [p for root in roots for p in Path(root).rglob(pattern) if p.is_file()]
    hits.sort(key=lambda p: (bool(hint and hint in str(p)), p.stat().st_mtime,
                             str(p)), reverse=True)
    if not hits and required:
        raise FileNotFoundError(f'Missing {pattern}; searched {roots}')
    return hits[0] if hits else None


def load_evalkit(work):
    path = Path(work) / 'evalkit.py'
    if not path.is_file():
        raise FileNotFoundError(f'Provide the Stage 02 module at {path}')
    spec = importlib.util.spec_from_file_location('evalkit', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['evalkit'] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    RDLogger.DisableLog('rdApp.*')


def check_ram(limit=10e9):
    used = psutil.Process().memory_info().rss
    if used >= limit:
        raise MemoryError(f'Process RAM {used / 1e9:.2f} GB >= {limit / 1e9} GB')
    return used


class SchemaAdapter:
    aliases = {'molecule_id': ['molecule_id', 'mol_id', 'id'],
        'smiles': ['smiles', 'SMILES', 'canonical_smiles'],
        'mz': ['mz', 'mzs', 'peak_mz'], 'inten': ['inten', 'intensity', 'intensities'],
        'precursor_mz': ['precursor_mz', 'precursor_mass', 'parent_mz'],
        'neutral_mass': ['neutral_mass', 'exact_mass'], 'adduct': ['adduct', 'adduct_type'],
        'ce': ['ce', 'collision_energy'], 'instrument': ['instrument', 'instrument_type'],
        'polarity': ['polarity', 'ion_mode'], 'formula': ['formula', 'molecular_formula']}

    def __init__(self, path, required):
        self.path = str(path)
        names = pq.ParquetFile(path).schema_arrow.names
        self.mapping = {key: next((v for v in opts if v in names), None)
                        for key, opts in self.aliases.items()}
        missing = [key for key in required if self.mapping.get(key) is None]
        if missing:
            raise ValueError(f'{path}: missing {missing}; available {names}')

    def batches(self, keys, batch_size=4096):
        mapping = {self.mapping[k]: k for k in keys if self.mapping.get(k)}
        for batch in pq.ParquetFile(self.path).iter_batches(
                batch_size=batch_size, columns=list(mapping), use_threads=False):
            frame = batch.to_pandas().rename(columns=mapping)
            if 'molecule_id' in frame:
                if frame.molecule_id.isna().any():
                    raise ValueError('Null molecule_id')
                frame['molecule_id'] = frame.molecule_id.astype(str)
            yield frame


def parse_formula(formula):
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula))
    if ''.join(e + n for e, n in parts) != str(formula):
        return {}
    result = Counter()
    for e, n in parts:
        result[e] += int(n or 1)
    return dict(result)


def formula_text(counts):
    order = ['C', 'H'] + sorted(set(counts) - {'C', 'H'})
    return ''.join(e + (str(counts[e]) if counts[e] != 1 else '')
                   for e in order if counts.get(e, 0))


@lru_cache(maxsize=100000)
def smiles_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    return formula_text(parse_formula(rdMolDescriptors.CalcMolFormula(mol)))


def mass_hypotheses(group):
    masses, unknown = [], set()
    for row in group[['precursor_mz', 'adduct', 'polarity']].drop_duplicates().itertuples():
        adduct = str(row.adduct).strip()
        if adduct not in ADDUCTS:
            unknown.add(adduct)
            neg = str(row.polarity).lower() in ('negative', '-', '-1')
            choices = ['[M-H]-', '[M+Cl]-'] if neg else ['[M+H]+', '[M+Na]+', '[M+NH4]+']
        else:
            choices = [adduct]
        for name in choices:
            charge, shift = ADDUCTS[name]
            mass = float(row.precursor_mz) * charge - shift
            if math.isfinite(mass) and mass > 0:
                masses.append((mass, name))
    if not masses:
        raise ValueError('No finite positive precursor-derived neutral mass')
    # Each adduct gives one robust mass estimate across replicate spectra.
    return [(float(np.median([m for m, a in masses if a == name])), name)
            for name in sorted({a for _, a in masses})], sorted(unknown)


def build_formula_stats(train_adapter, split_map, db_path, cfg):
    prior, seen, counts = Counter(), set(), []
    for frame in train_adapter.batches(['molecule_id', 'smiles']):
        frame = frame.loc[frame.molecule_id.map(split_map).eq('train')]
        for row in frame.drop_duplicates('molecule_id').itertuples():
            if row.molecule_id in seen:
                continue
            seen.add(row.molecule_id)
            f = smiles_formula(str(row.smiles))
            d = parse_formula(f)
            if d and not (set(d) - set(MASS)):
                prior[f] += 1
                counts.append(d)
        check_ram(cfg['parent_ram_gb'] * 1e9)
    if not counts:
        raise ValueError('No train-only formulas usable for chemical bounds')
    allowed = ELEMENTS + [e for e in EXTRA if any(d.get(e, 0) for d in counts)]
    ratios = {e: np.percentile([d.get(e, 0) / max(d.get('C', 0), 1)
                               for d in counts], [1, 99]).tolist() for e in allowed}
    # Explicit finite search domain; no prediction-dependent pruning or time truncation.
    bounds = {e: max(d.get(e, 0) for d in counts) for e in allowed}
    bounds.update(cfg.get('formula_max_counts', {}))
    if db_path:
        adapter = SchemaAdapter(db_path, [])
        key = 'formula' if adapter.mapping['formula'] else 'smiles'
        if not adapter.mapping[key]:
            raise ValueError('DB needs a formula or smiles column')
        for frame in adapter.batches([key]):
            values = frame[key].dropna().astype(str).value_counts()
            for value, freq in values.items():
                f = formula_text(parse_formula(value)) if key == 'formula' else smiles_formula(value)
                if f:
                    prior[f] += int(freq)
            check_ram(cfg['parent_ram_gb'] * 1e9)
    return dict(prior=dict(prior), allowed=allowed, ratios=ratios, bounds=bounds,
                prior_definition='one count per train molecule plus each optional DB row')


def sane_formula(d, stats):
    c = d.get('C', 0)
    if c < 1:
        return False
    h, n = d.get('H', 0), d.get('N', 0)
    hal = sum(d.get(e, 0) for e in ('F', 'Cl', 'Br', 'I', 'Na', 'K'))
    dbe2 = 2 + 2 * (c + d.get('Si', 0)) + n + d.get('P', 0) + d.get('B', 0) - h - hal
    if dbe2 < 0 or dbe2 % 2:
        return False
    # Generalized closed-shell nitrogen parity for trivalent P/B and monovalent metals.
    nominal = sum(NOMINAL[e] * v for e, v in d.items())
    if nominal % 2 != n % 2:
        return False
    return all(lo - 1e-9 <= d.get(e, 0) / c <= hi + 1e-9
               for e, (lo, hi) in stats['ratios'].items())


def enumerate_formulas(mass, stats, ppm):
    """Exhaustive bounded heavy-element search; solve integer H at each leaf."""
    tolerance = mass * ppm / 1e6
    bounds, ratios = stats['bounds'], stats['ratios']
    elements = sorted((e for e in stats['allowed'] if e not in ('C', 'H')),
                      key=lambda e: MASS[e], reverse=True)
    for c in range(1, min(bounds['C'], int((mass + tolerance) / 12)) + 1):
        d = {'C': c}
        ranges = {e: (max(0, math.ceil(ratios[e][0] * c - 1e-9)),
                      min(bounds[e], math.floor(ratios[e][1] * c + 1e-9)))
                  for e in elements + ['H']}
        if any(lo > hi for lo, hi in ranges.values()):
            continue
        suffix_min, suffix_max = [0.] * (len(elements) + 1), [0.] * (len(elements) + 1)
        suffix_min[-1], suffix_max[-1] = [x * MASS['H'] for x in ranges['H']]
        for i in range(len(elements) - 1, -1, -1):
            e = elements[i]
            suffix_min[i] = suffix_min[i+1] + ranges[e][0] * MASS[e]
            suffix_max[i] = suffix_max[i+1] + ranges[e][1] * MASS[e]

        def walk(i, remaining):
            if remaining + tolerance < suffix_min[i] or remaining - tolerance > suffix_max[i]:
                return
            if i == len(elements):
                lo = max(ranges['H'][0], math.ceil((remaining - tolerance) / MASS['H']))
                hi = min(ranges['H'][1], math.floor((remaining + tolerance) / MASS['H']))
                for h in range(lo, hi + 1):
                    d['H'] = h
                    if sane_formula(d, stats):
                        yield formula_text(d)
                return
            e = elements[i]
            lo = max(ranges[e][0], math.ceil((remaining - tolerance - suffix_max[i+1]) / MASS[e]))
            hi = min(ranges[e][1], math.floor((remaining + tolerance - suffix_min[i+1]) / MASS[e]))
            for value in range(lo, hi + 1):
                d[e] = value
                yield from walk(i+1, remaining - value * MASS[e])
            d.pop(e, None)
        yield from walk(0, mass - c * 12)


def rank_formulas(hypotheses, pred, stats, cfg):
    scores = {}
    order = cfg['elements_order']
    pred = np.asarray(pred, dtype=float)
    for mass, adduct in hypotheses:
        for formula in enumerate_formulas(mass, stats, cfg['ppm']):
            d = parse_formula(formula)
            actual = np.array([d.get(e, 0) for e in order])
            count_loss = np.mean(np.abs(actual - pred) / np.sqrt(pred + 1))
            presence_loss = np.mean(np.abs((actual > 0) - (1 - np.exp(-pred))))
            prior = math.log1p(stats['prior'].get(formula, 0))
            score = -count_loss - cfg['presence_weight'] * presence_loss + cfg['prior_weight'] * prior
            if formula not in scores or score > scores[formula]['score']:
                scores[formula] = dict(formula=formula, score=float(score), mass=mass, adduct=adduct)
    return sorted(scores.values(), key=lambda r: (-r['score'], r['formula']))


@lru_cache(maxsize=100000)
def canonical_record(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return None
    # Strict final deduplication: limited enumeration raises on exhaustion; no plain-key fallback.
    mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    te = rdMolStandardize.TautomerEnumerator()
    te.SetMaxTautomers(1000)
    te.SetMaxTransforms(1000)
    enumeration = te.Enumerate(mol)
    if enumeration.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
        return ('canonicalization_limit',)
    canon = te.PickCanonical(enumeration)
    key = Chem.MolToInchiKey(canon)[:14]
    if not key:
        return None
    return (Chem.MolToSmiles(canon, isomericSmiles=False), key,
            float(rdMolDescriptors.CalcExactMolWt(canon)),
            formula_text(parse_formula(rdMolDescriptors.CalcMolFormula(canon))))


def postprocess(mid, paths, hypotheses, ranked, cfg):
    valid, limited, records = 0, 0, []
    formula_scores = {r['formula']: r['score'] for r in ranked}
    for smi, logp, method in paths:
        parsed = Chem.MolFromSmiles(smi) if smi else None
        if parsed is None or not parsed.GetNumAtoms():
            continue
        valid += 1
        record = canonical_record(smi)
        if record and record[0] == 'canonicalization_limit':
            limited += 1
            continue
        if record is None:
            continue
        canonical, key, mass, formula = record
        ppm = min(abs(mass - m) / m * 1e6 for m, _ in hypotheses)
        records.append(dict(molecule_id=mid, smiles=canonical, ik14=key,
            log_prob=logp, ppm_err=ppm, formula_score=formula_scores.get(formula,
                min(formula_scores.values(), default=0.) - 1.)))
    strict = [r for r in records if r['ppm_err'] <= cfg['ppm']]
    relaxed = not strict
    kept = strict or [r for r in records if r['ppm_err'] <= cfg['relax_ppm']]
    output = []
    for key in sorted({r['ik14'] for r in kept}):
        hits = [r for r in kept if r['ik14'] == key]
        best = max(hits, key=lambda r: r['log_prob']).copy()
        best.update(count=len(hits), mean_log_prob=float(np.mean([r['log_prob'] for r in hits])),
                    mass_relaxed=relaxed)
        output.append(best)
    diag = dict(molecule_id=mid, attempted=len(paths), valid=valid,
        canonicalization_limited=limited, strict_mass_hits=len(strict), kept_hits=len(kept),
        unique=len(output), no_strict_survivor=relaxed, no_survivor=not output,
        formula_count=len(ranked), formula_top3=[r['formula'] for r in ranked[:3]])
    return output, diag


def autocast(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()


@torch.inference_mode()
def encode_group(model, group, device):
    """Average encoded memory at each valid position over all spectra, weighted by sqrt(peaks)."""
    import spec2smiles as s2s
    from spec2fp import collate_spectra
    memory_sum = weight_sum = pooled_sum = cond_sum = None
    total = 0.
    for row in group.itertuples(index=False):
        item = s2s.spectrum_item(row, model.cfg)
        batch = collate_spectra([item], model.cfg['top_k_peaks'], 2 * model.cfg['mz_dim'] + 1)
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            memory, mask, pooled, cond = model.encode(**batch)
        weight = math.sqrt(max(item['n_peaks'], 1))
        token_weight = (~mask).float().unsqueeze(-1) * weight
        weighted = memory.float() * token_weight
        memory_sum = weighted if memory_sum is None else memory_sum + weighted
        weight_sum = token_weight if weight_sum is None else weight_sum + token_weight
        pooled_sum = pooled.float() * weight if pooled_sum is None else pooled_sum + pooled.float() * weight
        cond_sum = cond.float() * weight if cond_sum is None else cond_sum + cond.float() * weight
        total += weight
    hypotheses, _ = mass_hypotheses(group)
    mass = float(group.neutral_mass.median()) if 'neutral_mass' in group else hypotheses[0][0]
    if not math.isfinite(mass) or mass <= 0:
        mass = hypotheses[0][0]
    with autocast(device):
        return model.conditioned_memory(memory_sum / weight_sum.clamp_min(1),
            weight_sum.squeeze(-1).eq(0), torch.tensor([mass], device=device),
            pooled_sum / total, cond_sum / total)


@torch.inference_mode()
def decode_beam(model, memory, mask, cfg, prefix=None):
    """Full batched beam without unsafe early stopping; archive all completed paths."""
    b, length, dim = memory.shape
    k, device = cfg['beam'], memory.device
    mem = memory[:, None].expand(-1, k, -1, -1).reshape(b*k, length, dim)
    masks = mask[:, None].expand(-1, k, -1).reshape(b*k, length)
    prefix = prefix or [1]
    seq = torch.tensor(prefix, device=device)[None, None].expand(b, k, -1).clone()
    scores = torch.full((b, k), -torch.inf, device=device)
    scores[:, 0] = 0
    completed = [[] for _ in range(b)]
    for step in range(len(prefix), cfg['max_length']):
        with autocast(device):
            logits = model.decode(seq.reshape(b*k, -1), mem, masks)[:, -1].float()
        logp = logits.log_softmax(-1).reshape(b, k, -1)
        logp[:, :, [0, 1, 3]] = -torch.inf
        if step == len(prefix):
            logp[:, :, 2] = -torch.inf
        combined = logp + scores[:, :, None]
        eos = combined[:, :, 2].cpu().tolist()
        seq_cpu = seq.cpu().tolist()
        for i in range(b):
            for j, score in enumerate(eos[i]):
                if math.isfinite(score):
                    ids = seq_cpu[i][j][len(prefix):] + [2]
                    completed[i].append((ids, score / len(ids) ** cfg['length_penalty']))
            completed[i] = sorted(completed[i], key=lambda p: -p[1])[:cfg['num_return']]
        combined[:, :, 2] = -torch.inf
        scores, indices = combined.flatten(1).topk(k, dim=1)
        parents, tokens = indices // logp.shape[-1], indices % logp.shape[-1]
        seq = torch.cat((seq.gather(1, parents[:, :, None].expand(-1, -1, step)), tokens[:, :, None]), 2)
    return completed


@torch.inference_mode()
def decode_sample(model, memory, mask, cfg, prefix=None):
    b, length, dim = memory.shape
    n, device = cfg['samples'], memory.device
    mem = memory[:, None].expand(-1, n, -1, -1).reshape(b*n, length, dim)
    masks = mask[:, None].expand(-1, n, -1).reshape(b*n, length)
    prefix = prefix or [1]
    seq = torch.tensor(prefix, device=device)[None].expand(b*n, -1).clone()
    scores = torch.zeros(b*n, device=device)
    lengths = torch.zeros(b*n, device=device)
    done = torch.zeros(b*n, dtype=torch.bool, device=device)
    for step in range(len(prefix), cfg['max_length']):
        with autocast(device):
            logits = model.decode(seq, mem, masks)[:, -1].float()
        original = logits.log_softmax(-1)
        logits[:, [0, 1, 3]] = -torch.inf
        if step == len(prefix):
            logits[:, 2] = -torch.inf
        values, indices = (logits / cfg['temperature']).sort(descending=True)
        probs = values.softmax(-1)
        values[(probs.cumsum(-1) - probs) >= cfg['top_p']] = -torch.inf
        chosen = torch.multinomial(values.softmax(-1), 1)
        token = indices.gather(1, chosen).squeeze(1)
        scores += original.gather(1, token[:, None]).squeeze(1) * (~done)
        lengths += ~done
        token = torch.where(done, 2, token)
        seq = torch.cat((seq, token[:, None]), 1)
        done |= token.eq(2)
        if done.all():
            break
    result = [[] for _ in range(b)]
    for i, (ids, score, size, finished) in enumerate(zip(seq.cpu().tolist(),
            scores.cpu().tolist(), lengths.cpu().tolist(), done.cpu().tolist())):
        # Unfinished draws remain explicit invalid attempts in diagnostics.
        output = ids[len(prefix):] if finished else []
        result[i // n].append((output, score / max(size, 1) ** cfg['length_penalty']))
    return result


def prediction_map(paths, name, expected, cfg):
    ids = pd.read_parquet(paths[f'ids_{name}'], columns=['molecule_id']).molecule_id.astype(str)
    values = np.load(paths[f'elem_{name}'], mmap_mode='r')
    if ids.duplicated().any() or values.shape != (len(ids), len(cfg['elements_order'])):
        raise ValueError(f'{name}: element array/ID shape mismatch or duplicated IDs')
    missing = set(expected) - set(ids)
    if missing:
        raise ValueError(f'{name}: {len(missing)} missing element predictions')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'{name}: invalid predicted element counts')
    wanted = set(expected)
    return {mid: np.array(values[i], copy=True) for i, mid in enumerate(ids) if mid in wanted}


def run_worker(job_path, shard):
    job = json.loads(Path(job_path).read_text())
    cfg, paths = job['cfg'], job['paths']
    seed_all(cfg['seed'] + shard)
    load_evalkit(cfg['work'])
    import spec2smiles as s2s
    device = torch.device('cpu' if cfg['SMOKE_TEST'] else 'cuda:0')
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95, cfg['gpu_memory_gb'] * 1e9 / total))
        torch.cuda.reset_peak_memory_stats()
    model_cfg = json.loads(Path(paths['config']).read_text())
    model = s2s.Spec2SmilesModel(model_cfg)
    checkpoint = torch.load(paths['model'], map_location='cpu', weights_only=False)
    state = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del checkpoint, state
    model.to(device).eval()
    tokenizer = s2s.SmilesTokenizer.load(paths['vocab'])
    if cfg['max_length'] > min(tokenizer.max_length, model.positions.shape[0]):
        raise ValueError('Configured sequence length exceeds trained positional/vocabulary limit')
    if model_cfg.get('formula_prefix', False):
        raise ValueError('This runtime targets Stage 07 v2; formula-prefix checkpoints need their trained adapter')
    print('Conditioning: native predicted elements; formula prefix/no-formula token unsupported', flush=True)
    stats = json.loads(Path(paths['formula_stats']).read_text())
    out = Path(cfg['out'])
    batch_size = cfg['batch_size']
    for name in ('val_model', 'val_rank', 'test'):
        parts = [Path(p) for p in job['parts'][name][str(shard)]]
        expected = job['ids'][name][shard::2]
        predictions = prediction_map(paths, name, expected, cfg)
        for part in parts:
            seed_all(cfg['seed'] + shard + int(part.stem) * 101 +
                     {'val_model': 0, 'val_rank': 100000, 'test': 200000}[name])
            stem = out / f'{name}_shard{shard}_{part.stem}'
            complete = stem.with_suffix('.done.json')
            if cfg['resume'] and complete.exists():
                if json.loads(complete.read_text())['run_hash'] == job['run_hash']:
                    continue
            frame = pd.read_parquet(part)
            groups = list(frame.groupby('molecule_id', sort=False))
            rows, diagnostics, formula_rows, index = [], [], [], 0
            started = time.perf_counter()
            while index < len(groups):
                chunk = groups[index:index + batch_size]
                attempt_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state() if device.type == 'cuda' else None
                try:
                    generated = generate_chunk(model, tokenizer, chunk, device, cfg)
                except torch.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError('OOM at batch=1; beam/sample counts preserved. Use a larger GPU.')
                    batch_size = max(1, batch_size // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.set_rng_state(attempt_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng)
                    print(f'OOM: retry batch={batch_size}, beam={cfg["beam"]}', flush=True)
                    continue
                for (mid, group), paths_generated in zip(chunk, generated):
                    hypotheses, unknown = mass_hypotheses(group)
                    ranked = rank_formulas(hypotheses, predictions[mid], stats, cfg)
                    candidates, diag = postprocess(mid, paths_generated, hypotheses, ranked, cfg)
                    diag.update(split=name, unknown_adducts=unknown)
                    rows.extend(candidates)
                    diagnostics.append(diag)
                    formula_rows.extend(dict(molecule_id=mid, rank=i+1, **r) for i, r in enumerate(ranked[:3]))
                index += len(chunk)
                check_ram(cfg['worker_ram_gb'] * 1e9)
                print(f'{name} shard={shard} {part.stem}: {index}/{len(groups)}', flush=True)
            pd.DataFrame(rows, columns=RAW_COLS).to_parquet(stem.with_suffix('.parquet'), index=False)
            pd.DataFrame(diagnostics).to_parquet(stem.with_suffix('.diag.parquet'), index=False)
            pd.DataFrame(formula_rows, columns=['molecule_id', 'rank', 'formula', 'score', 'mass', 'adduct']).to_parquet(stem.with_suffix('.formula.parquet'), index=False)
            peak = torch.cuda.max_memory_reserved() / 1e9 if device.type == 'cuda' else 0.
            if peak >= 12:
                raise MemoryError(f'GPU memory budget violated: {peak:.2f} GB')
            complete.write_text(json.dumps(dict(run_hash=job['run_hash'], queries=len(groups),
                seconds=time.perf_counter()-started, peak_gpu_gb=peak, batch_size=batch_size)))
            print(f'{name} shard={shard} {part.stem}: {len(groups)} queries; peak={peak:.2f} GB', flush=True)
            del frame, groups, rows, diagnostics, formula_rows
            gc.collect()


def generate_chunk(model, tokenizer, chunk, device, cfg):
    encoded = [encode_group(model, group, device) for _, group in chunk]
    memory = torch.cat([m for m, _ in encoded])
    mask = torch.cat([m for _, m in encoded])
    beam = decode_beam(model, memory, mask, cfg)
    sample = decode_sample(model, memory, mask, cfg)
    return [[(tokenizer.decode(ids) if ids else '', score, method)
             for method, paths in [('beam', b), ('sample', s)] for ids, score in paths]
            for b, s in zip(beam, sample)]


def scored(frame, alpha, formula_weight=0.):
    frame = frame.copy()
    frame['score'] = frame.log_prob + alpha * np.log(frame['count']) + formula_weight * frame.formula_score
    frame['source'] = 'gen'
    return frame.sort_values(['molecule_id', 'score', 'ik14'], ascending=[True, False, True])


def ranking_metrics(frame, truth):
    ranks = {}
    for mid, group in frame.groupby('molecule_id', sort=False):
        keys = group.ik14.drop_duplicates().head(25).tolist()
        if truth.get(mid) in keys:
            ranks[mid] = keys.index(truth[mid]) + 1
    n = len(truth)
    return dict(queries=n, mrr25=sum(1/r for r in ranks.values()) / max(n, 1),
                **{f'top{k}': sum(r <= k for r in ranks.values()) / max(n, 1) for k in (1, 5, 25)})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    run_worker(sys.argv[1], int(sys.argv[2]))
