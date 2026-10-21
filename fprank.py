"""Stage 06: bounded fingerprint scoring and validation-only rank tuning.

Call candidates.configure(..., exclusion_scope='train') before use. Arrays are
aligned positionally with queries_df (one row per molecule). No fitted global
state; all scoring parameters live in the JSON-serializable cfg dictionary.
"""
from functools import lru_cache
import re
import hashlib

import numpy as np
import pandas as pd

BYTE_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1,
                         bitorder='little').astype(np.float32)
ELEMENTS = ('C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I')
FEATURES = ('nb_ll', 'cos_sim', 'tan_sim', 'mass_score', 'elem_score', 'src_score', 'maccs_cos', 'maccs_tan')
DEFAULT_CONFIG = dict(ppm=15.0, temperature=1.0, sigma_ppm=5.0,
                      fallback_ppm=50.0, fallback_da=0.05, min_candidates=25,
                      batch_rows=2048, top_n=256, limit=100, eps=1e-6,
                      max_tune_queries=5000,
                      canonical_limit=25,
                      weights=[1., 1., 1., 0.1, 0.1, 0.1, 0.5, 0.5],
                      feature_scales=[4096., 1., 1., 1., 10., 1., 1., 1.])
OUTPUT_DTYPES = dict(row_id='int64', ik14='string', smiles='string', formula='string',
                     mass='float64', source_mask='uint64', fast_key='string', identity_kind='string',
                     **{name: 'float32' for name in FEATURES}, ppm_err='float64', elem_dist='float32',
                     score='float64', scoring_source_mask='uint64', canon_ik14='string',
                     molecule_id='string', rank='int64')


def output_schema():
    """Explicit schema even when a query contains only null canonical tail keys."""
    import pyarrow as pa
    return pa.schema([(name, pa.string() if dtype == 'string' else pa.from_numpy_dtype(dtype))
                      for name, dtype in OUTPUT_DTYPES.items()])


def normalize_output(frame):
    return frame.loc[:, list(OUTPUT_DTYPES)].astype(OUTPUT_DTYPES)


def config(cfg):
    result = dict(DEFAULT_CONFIG, **cfg)
    for key in ('ppm', 'temperature', 'sigma_ppm', 'fallback_ppm', 'fallback_da'):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('batch_rows', 'top_n', 'limit', 'min_candidates', 'max_tune_queries', 'canonical_limit'):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < result['eps'] < 0.5:
        raise ValueError('eps must be in (0, 0.5)')
    if result['limit'] > result['top_n']:
        raise ValueError('limit exceeds top_n')
    if result['canonical_limit'] < 25:
        raise ValueError('canonical_limit must cover all 25 submission guesses')
    for key in ('weights', 'feature_scales'):
        x = np.asarray(result[key], dtype=float)
        if x.shape == (6,):
            pad_len = len(FEATURES) - 6
            if key == 'weights':
                x = np.pad(x, (0, pad_len), constant_values=0.5)
            else:
                x = np.pad(x, (0, pad_len), constant_values=1.0)
            result[key] = x.tolist()
        elif x.shape != (len(FEATURES),) or not np.isfinite(x).all():
            raise ValueError(f'Invalid {key}')
    if np.any(np.asarray(result['feature_scales']) <= 0):
        raise ValueError('feature_scales must be positive')
    return result


@lru_cache(maxsize=32768)
def formula_counts(formula):
    parts = re.findall(r'([A-Z][a-z]?)([0-9]*)', str(formula))
    counts = np.zeros(10, dtype=np.float32)
    for symbol, number in parts:
        if symbol in ELEMENTS:
            counts[ELEMENTS.index(symbol)] += int(number or 1)
    return counts


def unpack_fingerprints(packed):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 533:
        raise ValueError('Expected uint8 packed Morgan4096 + MACCS167, width 533')
    return (BYTE_LUT[packed[:, :512]].reshape(-1, 4096),
            BYTE_LUT[packed[:, 512:]].reshape(-1, 168)[:, :167])


def score_features(frame, packed, probabilities, elements, mass, cfg, *,
                   bits=None, maccs_probabilities=None, maccs_bits=None):
    """Raw feature columns; composite NB is normalized by 4096 by default."""
    if bits is None or (maccs_probabilities is not None and maccs_bits is None):
        b, m = unpack_fingerprints(packed)
        if bits is None:
            bits = b
        if maccs_bits is None:
            maccs_bits = m
    p = np.clip(np.asarray(probabilities, dtype=np.float64), cfg['eps'], 1-cfg['eps'])
    logits = np.log(p) - np.log1p(-p)
    slope = np.asarray(cfg.get('platt_slope', 1.), dtype=float)
    intercept = np.asarray(cfg.get('platt_intercept', 0.), dtype=float)
    for value in (slope, intercept):
        if value.shape not in ((), (4096,)) or not np.isfinite(value).all():
            raise ValueError('Platt parameters must be finite scalars or 4096-vectors')
    logits = (logits * slope + intercept) / cfg['temperature']
    log_p = -np.logaddexp(0., -logits)
    log_not_p = -np.logaddexp(0., logits)
    p = np.exp(log_p).astype(np.float32)
    reliability = np.asarray(cfg.get('bit_weights', np.ones(4096)), dtype=np.float64)
    if reliability.shape != (4096,) or not np.isfinite(reliability).all() or (reliability < 0).any():
        raise ValueError('bit_weights must be 4096 finite nonnegative values')
    dot = bits @ p
    count = bits.sum(axis=1)
    nb = bits @ (reliability * logits).astype(np.float32) + np.sum(reliability * log_not_p)
    ppm = np.abs(frame.mass.to_numpy(dtype=float) - mass) / mass * 1e6
    elem = np.abs(np.stack([formula_counts(f) for f in frame.formula]) - elements).sum(axis=1)
    # Curated-source prior excludes TRAIN_BIT and uses the filtered source mask.
    src = np.array([(int(m) & ~1).bit_count() for m in frame.source_mask], dtype=float)

    if maccs_probabilities is not None:
        p_mc = np.clip(np.asarray(maccs_probabilities, dtype=np.float64), cfg['eps'], 1 - cfg['eps'])
        mc_slope = np.asarray(cfg.get('maccs_platt_slope', 1.), dtype=float)
        mc_intercept = np.asarray(cfg.get('maccs_platt_intercept', 0.), dtype=float)
        mc_logits = (np.log(p_mc) - np.log1p(-p_mc)) * mc_slope + mc_intercept
        mc_logits = mc_logits / cfg['temperature']
        mc_log_p = -np.logaddexp(0., -mc_logits)
        p_mc = np.exp(mc_log_p).astype(np.float32)
        dot_mc = maccs_bits @ p_mc
        count_mc = maccs_bits.sum(axis=1)
        norm_mc = np.linalg.norm(p_mc)
        maccs_cos = (dot_mc / np.maximum(np.sqrt(count_mc) * norm_mc, 1e-12)).astype(np.float32)
        maccs_tan = (dot_mc / np.maximum(count_mc + p_mc.sum() - dot_mc, 1e-12)).astype(np.float32)
    else:
        maccs_cos = np.zeros(len(frame), dtype=np.float32)
        maccs_tan = np.zeros(len(frame), dtype=np.float32)

    return np.column_stack((nb, dot / np.maximum(np.sqrt(count)*np.linalg.norm(p), 1e-12),
                            dot / np.maximum(count+p.sum()-dot, 1e-12),
                            -0.5*(ppm/cfg['sigma_ppm'])**2, -elem, src,
                            maccs_cos, maccs_tan)).astype(np.float32)


def validate_inputs(queries, fp, elem, maccs=None):
    if not {'molecule_id', 'neutral_mass'} <= set(queries):
        raise ValueError('queries require molecule_id and observed neutral_mass')
    if queries.molecule_id.isna().any() or queries.molecule_id.astype(str).duplicated().any():
        raise ValueError('Expected unique, non-null molecule IDs')
    if fp.shape != (len(queries), 4096) or elem.shape != (len(queries), 10):
        raise ValueError('Prediction shape/alignment mismatch')
    if maccs is not None and maccs.shape != (len(queries), 167):
        raise ValueError(f'MACCS prediction shape mismatch, expected ({len(queries)}, 167), got {maccs.shape}')
    masses = queries.neutral_mass.to_numpy(dtype=float)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError('Invalid observed neutral mass')
    # Chunk validation avoids copying an entire prediction mmap.
    for start in range(0, len(queries), 1024):
        p, e = fp[start:start+1024], elem[start:start+1024]
        if not np.isfinite(p).all() or not np.isfinite(e).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Nonfinite predictions or probabilities outside [0,1]')
        if maccs is not None:
            mc = maccs[start:start+1024]
            if not np.isfinite(mc).all() or (mc < 0).any() or (mc > 1).any():
                raise ValueError('Nonfinite MACCS predictions or probabilities outside [0,1]')


def exclusion_ids(queries, cfg):
    import candidates
    if candidates.context()['scope'] != 'train':
        raise ValueError("Configure candidates with exclusion_scope='train'")
    is_validation = ('split3' in queries and queries.split3.isin(['val_model', 'val_rank']).any())
    if is_validation and 'exclude_keys' not in cfg:
        raise ValueError('Validation requires validation_exclusions(db) in cfg[exclude_keys]')
    return candidates.excluded_rows(cfg.get('exclude_keys', ()))


def candidate_batches(mass, ppm, blocked, cfg, *, outside_ppm=None):
    import candidates
    lo, hi = candidates.mass_bounds(mass, ppm)
    state = candidates.context()
    size = min(cfg['batch_rows'], state['meta']['max_materialized_rows'])
    spans = [(lo, hi)]
    if outside_ppm is not None:
        inner_lo, inner_hi = candidates.mass_bounds(mass, outside_ppm)
        spans = [(lo, min(hi, inner_lo)), (max(lo, inner_hi), hi)]
    for first, last in spans:
        for start in range(first, last, size):
            ids = state['order'][start:min(last, start+size)]
            # Resolve exclude_keys once per call, not once per query or chunk.
            frame = candidates.filter_sources(candidates.fetch_rows(ids), excluded_ids=blocked)
            if not frame.empty:
                yield frame


def query_features(mass, p, e, blocked, cfg, *, maccs=None):
    import candidates
    count = 0
    for frame in candidate_batches(mass, cfg['ppm'], blocked, cfg):
        count += len(frame)
        yield frame, score_features(frame, candidates.get_fingerprints(frame.row_id, smiles=frame.smiles.to_numpy()),
                                     p, e, mass, cfg, maccs_probabilities=maccs)
    if count < cfg['min_candidates']:
        wide = max(cfg['ppm'], cfg['fallback_ppm'], cfg['fallback_da']/mass*1e6)
        for frame in candidate_batches(mass, wide, blocked, cfg, outside_ppm=cfg['ppm']):
            count += len(frame)
            yield frame, score_features(frame, candidates.get_fingerprints(frame.row_id, smiles=frame.smiles.to_numpy()),
                                         p, e, mass, cfg, maccs_probabilities=maccs)
    if count == 0:
        for frame in nearest_candidate_batches(mass, blocked, cfg):
            yield frame, score_features(frame, candidates.get_fingerprints(frame.row_id, smiles=frame.smiles.to_numpy()),
                                         p, e, mass, cfg, maccs_probabilities=maccs)


def nearest_candidate_batches(mass, blocked, cfg, limit=50):
    """Scan out from searchsorted until the nearest 50 eligible rows are found.

    Source filtering is identical to window retrieval. At least one eligible DB
    row is necessary for coverage; an empty/fully excluded database fails clearly.
    """
    import candidates
    state = candidates.context()
    masses, order = state['mass'], state['order']
    right = int(np.searchsorted(masses, mass))
    left, found = right-1, 0
    size = min(cfg['batch_rows'], state['meta']['max_materialized_rows'], limit)
    while found < limit and (left >= 0 or right < len(masses)):
        ids = []
        for _ in range(min(size, limit-found)):
            if left < 0 and right >= len(masses):
                break
            if right >= len(masses) or (left >= 0 and mass-masses[left] <= masses[right]-mass):
                ids.append(int(order[left]))
                left -= 1
            else:
                ids.append(int(order[right]))
                right += 1
        frame = candidates.filter_sources(candidates.fetch_rows(ids), excluded_ids=blocked)
        if not frame.empty:
            found += len(frame)
            yield frame
    if found == 0:
        raise ValueError('Candidate database has no eligible rows, including nearest-mass fallback')


def composite(features, cfg):
    w = np.asarray(cfg['weights'], dtype=float)
    s = np.asarray(cfg['feature_scales'], dtype=float)
    k = features.shape[1]
    return features @ (w[:k] / s[:k])


def top_indices(scores, row_ids, n):
    # Deterministic tie-breaking also at the shortlist boundary.
    return np.lexsort((row_ids, -scores))[:n]


def iter_rank_candidates(queries_df, fp_probs, elem_preds, cfg, maccs_preds=None):
    """Stream rows and flush the lazy chemistry transaction at each call boundary."""
    import candidates
    try:
        yield from _iter_rank_candidates(queries_df, fp_probs, elem_preds, cfg, maccs_preds=maccs_preds)
    finally:
        candidates.flush_cache()


def _iter_rank_candidates(queries_df, fp_probs, elem_preds, cfg, maccs_preds=None):
    """Yield one top-100 frame per query in ascending observed-mass order."""
    import candidates
    cfg = config(cfg)
    validate_inputs(queries_df, fp_probs, elem_preds, maccs=maccs_preds)
    blocked = exclusion_ids(queries_df, cfg)
    for i in np.argsort(queries_df.neutral_mass.to_numpy(), kind='stable'):
        query = queries_df.iloc[i]
        shortlist = None
        mc_p = maccs_preds[i] if maccs_preds is not None else None
        for frame, features in query_features(float(query.neutral_mass), fp_probs[i], elem_preds[i],
                                              blocked, cfg, maccs=mc_p):
            frame = frame.copy()
            for j, name in enumerate(FEATURES):
                frame[name] = features[:, j]
            frame['ppm_err'] = np.abs(frame.mass-float(query.neutral_mass))/float(query.neutral_mass)*1e6
            frame['elem_dist'] = -frame.elem_score
            frame['score'] = composite(features, cfg)
            frame['scoring_source_mask'] = frame.source_mask
            merged = pd.concat([shortlist, frame], ignore_index=True) if shortlist is not None else frame
            shortlist = merged.iloc[top_indices(merged.score.to_numpy(), merged.row_id.to_numpy(), cfg['top_n'])]
        if shortlist is None:
            raise RuntimeError(f'No eligible candidates for {query.molecule_id}')
        final = candidates.finalize_candidates(shortlist, top_n=cfg['top_n'], limit=cfg['limit'],
                                                canonical_limit=cfg['canonical_limit'])
        final = final.drop(columns=['member_row_ids'], errors='ignore')
        final['source_mask'] = final.source_mask.astype(np.uint64)
        final['scoring_source_mask'] = final.scoring_source_mask.astype(np.uint64)
        final['molecule_id'] = str(query.molecule_id)
        final['rank'] = np.arange(1, len(final)+1)
        yield normalize_output(final)


def rank_candidates(queries_df, fp_probs, elem_preds, cfg, maccs_preds=None):
    """Stateless Stage 10 API. Use iter_rank_candidates for bounded output RAM."""
    frames = list(iter_rank_candidates(queries_df, fp_probs, elem_preds, cfg, maccs_preds=maccs_preds))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tuning_pool(mass, blocked, cfg):
    """One 20-ppm retrieval; cache wider fallback only if a grid window needs it."""
    max_ppm = 20.0
    count, narrow_count = 0, 0
    for frame in candidate_batches(mass, max_ppm, blocked, cfg):
        count += len(frame)
        narrow_count += int((np.abs(frame.mass.to_numpy()-mass)/mass*1e6 <= 10.).sum())
        yield frame
    if narrow_count < cfg['min_candidates']:
        wide = max(max_ppm, cfg['fallback_ppm'], cfg['fallback_da']/mass*1e6)
        for frame in candidate_batches(mass, wide, blocked, cfg, outside_ppm=max_ppm):
            count += len(frame)
            yield frame
    if count == 0:
        yield from nearest_candidate_batches(mass, blocked, cfg)


def build_tuning_cache(queries, fp, elem, truth_keys, cfg, indices, maccs=None):
    """In-memory small feature arrays, no packed/unpacked fingerprints retained.

    Candidate metadata/fingerprints are read once per sampled query. Hit labels
    use existing fast/canonical keys (a proxy, without tautomer canonicalization).
    An optional query truth_fast_key supports truths with a different tautomer key.
    """
    import candidates
    blocked = exclusion_ids(queries, cfg)
    chunks = []
    for q, i in enumerate(indices):
        mass = float(queries.iloc[i].neutral_mass)
        truth = truth_keys[i]
        fast_truth = queries.iloc[i].get('truth_fast_key', truth)
        mc_i = maccs[i] if maccs is not None else None
        for frame in tuning_pool(mass, blocked, cfg):
            packed = candidates.get_fingerprints(frame.row_id, smiles=frame.smiles.to_numpy())
            bits, cand_maccs = unpack_fingerprints(packed)
            features = np.stack([score_features(frame, packed, fp[i], elem[i], mass,
                                                dict(cfg, temperature=t), bits=bits,
                                                maccs_probabilities=mc_i, maccs_bits=cand_maccs)
                                 for t in (0.7, 1., 1.3)], axis=1)
            hit = (frame.fast_key.eq(truth) | frame.fast_key.eq(fast_truth) | frame.ik14.eq(truth))
            chunks.append(dict(x=features, row_ids=frame.row_id.to_numpy(dtype=np.int64),
                               is_hit=hit.to_numpy(dtype=bool),
                               ppm_err=np.abs(frame.mass.to_numpy(dtype=float)-mass)/mass*1e6,
                               query=np.full(len(frame), q, dtype=np.int32)))
        if (q+1) % 100 == 0:
            print(f'Tuning features: {q+1}/{len(indices)} queries', flush=True)
    cache = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    cache['n_queries'] = len(indices)
    return cache


def tuning_window(cache, ppm, temperature_index, cfg):
    """Filter each query's cached pool; sparse primary windows retain fallback rows."""
    within = cache['ppm_err'] <= ppm
    counts = np.bincount(cache['query'][within], minlength=cache['n_queries'])
    keep = within | (counts[cache['query']] < cfg['min_candidates'])
    return dict(x=cache['x'][keep, temperature_index, :],
                row_ids=cache['row_ids'][keep], is_hit=cache['is_hit'][keep],
                query=cache['query'][keep], n_queries=cache['n_queries'])


def cache_reciprocal_ranks(cache, cfg):
    """Vectorized raw-key MRR@25 proxy, including misses; no I/O or full sorting.

    Count candidates ahead of the highest-scoring hit with row_id tie-breaking.
    Official canonical deduplicated MRR is measured on full outputs after freeze.
    """
    scores = composite(cache['x'], cfg)
    q, hit, ids, n = cache['query'], cache['is_hit'], cache['row_ids'], cache['n_queries']
    best = np.full(n, -np.inf)
    np.maximum.at(best, q[hit], scores[hit])
    best_ids = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    top_hits = hit & (scores == best[q])
    np.minimum.at(best_ids, q[top_hits], ids[top_hits])
    ahead = (scores > best[q]) | ((scores == best[q]) & (ids < best_ids[q]))
    ranks = 1 + np.bincount(q[ahead], minlength=n)
    valid = np.isfinite(best) & (ranks <= 25)
    rr = np.zeros(n, dtype=float)
    rr[valid] = 1. / ranks[valid]
    return rr


def cache_mrr(cache, cfg):
    rr = cache_reciprocal_ranks(cache, cfg)
    return float(rr.mean()) if len(rr) else 0.0


def tuning_diagnostics(window, cfg, hit_in_20ppm):
    """Conditional numerator and denominator use the same fixed 20-ppm hit cohort."""
    rr = cache_reciprocal_ranks(window, cfg)
    covered = int(hit_in_20ppm.sum())
    return dict(overall_proxy_mrr25=float(rr.mean()) if len(rr) else 0.0,
                covered_hit_mrr25=float(rr[hit_in_20ppm].mean()) if covered else 0.0,
                hit_queries_20ppm=covered,
                oracle_20ppm_recall=covered / max(len(rr), 1))


def tune(queries, fp, elem, truth_keys, cfg, cache_dir=None, sweeps=3,
         max_tune_queries: int = 5000, maccs=None):
    """Deterministic sample, one retrieval pass, then entirely in-memory search.

    cache_dir is retained for call compatibility and is unused. cfg may lower the
    sample cap further. No canonicalization or candidate I/O occurs during search.
    """
    cfg = config(cfg)
    validate_inputs(queries, fp, elem, maccs=maccs)
    if 'split3' not in queries or not queries.split3.eq('val_model').all():
        raise ValueError('Tuning may consume val_model only')
    if len(queries) == 0 or len(truth_keys) != len(queries):
        raise ValueError('Nonempty, aligned validation truths required')
    if not isinstance(max_tune_queries, int) or max_tune_queries < 1:
        raise ValueError('max_tune_queries must be a positive integer')
    cap = min(max_tune_queries, cfg['max_tune_queries'])
    indices = np.random.default_rng(42).choice(len(queries), min(cap, len(queries)), replace=False)
    indices = indices[np.argsort(queries.neutral_mass.to_numpy()[indices], kind='stable')]
    import candidates
    try:
        if maccs is not None:
            cache = build_tuning_cache(queries, fp, elem, truth_keys, cfg, indices, maccs=maccs)
        else:
            cache = build_tuning_cache(queries, fp, elem, truth_keys, cfg, indices)
    finally:
        candidates.flush_cache()
    hit_in_20ppm = np.zeros(cache['n_queries'], dtype=bool)
    hit_in_20ppm[cache['query'][cache['is_hit'] & (cache['ppm_err'] <= 20.)]] = True
    best, best_score, history = None, -1., []
    for ppm in (10., 15., 20.):
        for t, temperature in enumerate((0.7, 1., 1.3)):
            trial = dict(cfg, ppm=ppm, temperature=temperature)
            window = tuning_window(cache, ppm, t, cfg)
            current = cache_mrr(window, trial)
            for _ in range(sweeps):
                improved = False
                for j in range(len(trial['weights'])):
                    origin = trial['weights'][j]
                    values = (0.0, origin*0.5, origin*0.75, origin*1.25, max(origin*2.0, 0.1))
                    # Prefer the smaller step if several proposals improve MRR equally.
                    for value in sorted(set(values), key=lambda v: (abs(v-origin), v)):
                        weights = list(trial['weights'])
                        weights[j] = value
                        proposed = dict(trial, weights=weights)
                        score = cache_mrr(window, proposed)
                        if score > current + 1e-12:
                            trial, current, improved = proposed, score, True
                if not improved:
                    break
            diagnostics = tuning_diagnostics(window, trial, hit_in_20ppm)
            history.append(dict(ppm=ppm, temperature=temperature, mrr25=current,
                                weights=trial['weights'], **diagnostics))
            print('Tuning:', history[-1], flush=True)
            if current > best_score:
                best, best_score = dict(trial), current
                best_diagnostics = diagnostics
    best.pop('exclude_keys', None)
    ids = queries.iloc[indices].molecule_id.astype(str).tolist()
    best.update(tuned_on='val_model', tuning_proxy_mrr25=best_score, search_history=history,
                **best_diagnostics,
                tuning_sweeps=sweeps, covered_hit_scope='truth present in strict 20-ppm pool',
                tuning_metric='raw fast_key/ik14 MRR@25 proxy (no tautomer deduplication)',
                max_tune_queries=cap, tuning_queries=len(indices), tuning_seed=42,
                tuning_ids_hash=hashlib.sha256(''.join(sorted(ids)).encode()).hexdigest(),
                calibration='Stage05 probabilities; temperature selected on val_model')
    return best
