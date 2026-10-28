"""Numerical and pipeline contract tests without requiring RDKit/Arrow locally."""
import ast
from collections import OrderedDict
from pathlib import Path
import sqlite3
import sys
import types
import hashlib
import tempfile
import os
import itertools
import json

import numpy as np
import pandas as pd
import pytest

import fprank as fr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
BUILDERS = ROOT / 'builders'
NOTEBOOKS = ROOT / 'notebooks'

def find_path(rel_path):
    p = Path(rel_path)
    if p.is_file():
        return p
    for base in (SRC, BUILDERS, NOTEBOOKS, ROOT):
        cand = base / p.name
        if cand.is_file():
            return cand
    return p


def test_tuning_diagnostics_use_fixed_twenty_ppm_cohort():
    # q0 rank 1; q1 rank 2; q2 fallback-only rank 1; q3 missing.
    cache = dict(x=np.array([[3.], [2.], [3.], [1.]]), row_ids=np.arange(4),
                 query=np.array([0, 1, 1, 2]), is_hit=np.array([True, True, False, True]),
                 n_queries=4)
    cfg = dict(weights=[1.], feature_scales=[1.])
    result = fr.tuning_diagnostics(cache, cfg, np.array([True, True, False, False]))
    assert result['overall_proxy_mrr25'] == pytest.approx(2.5/4)
    assert result['covered_hit_mrr25'] == pytest.approx(1.5/2)
    assert result['oracle_20ppm_recall'] == .5
    assert result['hit_queries_20ppm'] == 2
    empty = fr.tuning_diagnostics(cache, cfg, np.zeros(4, dtype=bool))
    assert empty['covered_hit_mrr25'] == 0.


def test_notebook_query_mass_resists_outlier():
    from build_06_fp_rank_infer import build_notebook
    notebook = build_notebook()
    cell = next(c for c in notebook['cells'] if c['id'] == 'stage06-009')
    tree = ast.parse(cell['source'])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_queries')
    frame = pd.DataFrame(dict(molecule_id=['q']*3, neutral_mass=[100., 100., 602.6465]))
    scope = dict(pd=types.SimpleNamespace(read_parquet=lambda *a, **kw: frame.copy()),
                 pq=types.SimpleNamespace(ParquetFile=lambda p: types.SimpleNamespace(
                     schema_arrow=types.SimpleNamespace(names=list(frame.columns)))),
                 candidates=types.SimpleNamespace(observed_mass=lambda f: f.neutral_mass))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<read_queries>', 'exec'), scope)
    assert scope['read_queries']('unused').neutral_mass.iloc[0] == 100.


def test_notebook_coverage_denominators_include_missing_queries():
    from build_06_fp_rank_infer import build_notebook
    cell = next(c for c in build_notebook()['cells'] if c['id'] == 'stage06-013')
    tree = ast.parse(cell['source'])
    loop = next(n for n in tree.body if isinstance(n, ast.For))
    summary_if = next(n for n in loop.body if isinstance(n, ast.If)
                      and any(isinstance(x, ast.Name) and x.id == 'conditional' for x in ast.walk(n)))
    scope = dict(name='val_model', summaries={'val_model': {}}, queries=range(4),
                 truth=dict.fromkeys(['top1', 'top30', 'miss', 'empty']),
                 rr=dict(top1=1., top30=0., miss=0., empty=0.),
                 in_pool=dict(top1=True, top30=True, miss=False, empty=False))
    exec(compile(ast.Module(body=[summary_if], type_ignores=[]), '<summary>', 'exec'), scope)
    summary = scope['summaries']['val_model']
    assert summary['oracle_top100_recall'] == .5
    assert summary['mrr25'] == summary['mrr25_all'] == .25
    assert summary['mrr25_given_in_pool'] == summary['mrr25_when_in_top100'] == .5
    scope['in_pool'] = dict.fromkeys(scope['truth'], False)
    scope['rr'] = dict.fromkeys(scope['truth'], 0.)
    exec(compile(ast.Module(body=[summary_if], type_ignores=[]), '<summary>', 'exec'), scope)
    assert summary['mrr25_given_in_pool'] == 0.


@pytest.fixture
def candidate_api(monkeypatch):
    # Exercise the actual pure Stage 04 implementations, replacing only disk and chemistry I/O.
    tree = ast.parse(find_path('candidates.py').read_text(encoding='utf-8'))
    names = {'filter_sources', 'mass_bounds', 'finalize_candidates'}
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    api = types.ModuleType('candidates')
    api.np, api.pd, api.OrderedDict = np, pd, OrderedDict
    api.frame = pd.DataFrame(dict(row_id=[0, 1, 2, 3, 4], mass=[99.999, 100., 100.001, 100.003, 100.04],
                                 source_mask=[1, 3, 4, 8, 2], formula=['CH4']*5,
                                 fast_key=list('abcde'), smiles=['C']*5, ik14=['A', 'B', 'B', 'D', 'E'],
                                 identity_kind=['fast']*5))
    state = dict(scope='train', mass=api.frame.mass.to_numpy(), order=np.arange(5),
                 meta={'max_materialized_rows': 2}, cache=sqlite3.connect(':memory:'))
    api.context = lambda: state
    api.excluded_rows = lambda keys: {0, 1} if keys else set()
    api.fetch_rows = lambda ids: api.frame.iloc[np.asarray(ids)].copy().reset_index(drop=True)
    api.canonical_key = lambda row_id, smiles, known=None: known
    api.prefetch_canonical_keys = lambda ids: None
    api.flush_cache = lambda: None
    api.get_fingerprints = lambda ids, smiles=None: np.zeros((len(ids), 533), dtype=np.uint8)
    exec(compile(selected, 'candidates.py', 'exec'), api.__dict__)
    monkeypatch.setitem(sys.modules, 'candidates', api)
    yield api
    state['cache'].close()


def test_lut_and_scores_match_direct_equations():
    rng = np.random.default_rng(4)
    packed = rng.integers(0, 256, (7, 533), dtype=np.uint8)
    bits, maccs = fr.unpack_fingerprints(packed)
    np.testing.assert_array_equal(bits, np.unpackbits(packed[:, :512], axis=1, bitorder='little'))
    np.testing.assert_array_equal(maccs, np.unpackbits(packed[:, 512:], axis=1, bitorder='little')[:, :167])
    p = rng.uniform(.01, .99, 4096)
    frame = pd.DataFrame(dict(mass=[100.001]*7, formula=['C2H6O']*7, source_mask=[1, 2, 3, 6, 7, 8, 14]))
    cfg = fr.config({})
    x = fr.score_features(frame, packed, p, np.zeros(10), 100., cfg)
    expected_nb = (bits*np.log(p)+(1-bits)*np.log1p(-p)).sum(axis=1)
    np.testing.assert_allclose(x[:, 0], expected_nb, rtol=1e-6)
    dot = bits @ p
    np.testing.assert_allclose(x[:, 1], dot/(np.linalg.norm(bits, axis=1)*np.linalg.norm(p)), rtol=1e-6)
    np.testing.assert_allclose(x[:, 2], dot/(bits.sum(axis=1)+p.sum()-dot), rtol=1e-6)
    np.testing.assert_allclose(x[:, 3], -2.)
    np.testing.assert_allclose(x[:, 4], -9.)
    np.testing.assert_array_equal(x[:, 5], [0, 1, 1, 2, 2, 1, 3])
    # When maccs_probabilities is None, maccs similarities are zero
    np.testing.assert_allclose(x[:, 6], 0.)
    np.testing.assert_allclose(x[:, 7], 0.)
    # When maccs_probabilities is provided, scores match direct MACCS equations
    p_mc = rng.uniform(.01, .99, 167)
    x_mc = fr.score_features(frame, packed, p, np.zeros(10), 100., cfg, maccs_probabilities=p_mc)
    dot_mc = maccs @ p_mc
    expected_maccs_cos = dot_mc / (np.linalg.norm(maccs, axis=1) * np.linalg.norm(p_mc))
    expected_maccs_tan = dot_mc / (maccs.sum(axis=1) + p_mc.sum() - dot_mc)
    np.testing.assert_allclose(x_mc[:, 6], expected_maccs_cos, rtol=1e-6)
    np.testing.assert_allclose(x_mc[:, 7], expected_maccs_tan, rtol=1e-6)


def test_validation_filter_fallback_and_canonical_merge(candidate_api):
    queries = pd.DataFrame(dict(molecule_id=['q'], neutral_mass=[100.], split3=['val_rank']))
    p, e = np.full((1, 4096), .5), np.zeros((1, 10))
    with pytest.raises(ValueError, match='Validation requires'):
        fr.rank_candidates(queries, p, e, {})
    cfg = dict(exclude_keys={'held_out'}, weights=[0., 0., 0., 0., 0., 1.], batch_rows=2)
    ranked = fr.rank_candidates(queries, p, e, cfg)
    assert ranked.row_id.tolist() == [1, 3, 4]
    assert 'member_row_ids' not in ranked
    assert ranked.source_mask.dtype == np.dtype('uint64')
    assert ranked.source_mask.iloc[0] == 6
    assert ranked.scoring_source_mask.iloc[0] == 2
    assert ranked['rank'].tolist() == [1, 2, 3]
    assert set(fr.FEATURES) <= set(ranked)
    assert ranked.ppm_err.iloc[-1] == pytest.approx(400.)  # ±0.05 Da fallback


def test_cache_metric_and_reject_val_rank(candidate_api, monkeypatch):
    queries = pd.DataFrame(dict(molecule_id=['q', 'empty'], neutral_mass=[100., 1000.], split3=['val_model']*2))
    p, e = np.full((2, 4096), .5), np.zeros((2, 10))
    cfg = fr.config(dict(exclude_keys={'held_out'}))
    cache = fr.build_tuning_cache(queries, p, e, ['B', 'Z'], cfg, [0, 1])
    def no_io(*args, **kwargs):
        pytest.fail('Candidate I/O in tuning objective')
    monkeypatch.setattr(candidate_api, 'fetch_rows', no_io)
    monkeypatch.setattr(candidate_api, 'finalize_candidates', no_io)
    assert fr.cache_mrr(fr.tuning_window(cache, 15., 1, cfg), cfg) == .5
    with pytest.raises(ValueError, match='val_model only'):
        fr.tune(queries.assign(split3='val_rank'), p, e, ['B', 'Z'], cfg)


def test_query_sort_keeps_prediction_alignment(candidate_api):
    queries = pd.DataFrame(dict(molecule_id=['high', 'low'], neutral_mass=[100.04, 100.]))
    p = np.stack([np.full(4096, .9), np.full(4096, .1)])
    cfg = dict(min_candidates=1, weights=[1., 0., 0., 0., 0., 0.])
    result = fr.rank_candidates(queries, p, np.zeros((2, 10)), cfg)
    assert result.molecule_id.iloc[0] == 'low'
    for qid, probability in [('low', .1), ('high', .9)]:
        np.testing.assert_allclose(result.loc[result.molecule_id.eq(qid), 'nb_ll'], 4096*np.log1p(-probability), rtol=1e-6)


def test_empty_pools_and_invalid_predictions(candidate_api):
    q = pd.DataFrame(dict(molecule_id=['none'], neutral_mass=[1000.]))
    result = fr.rank_candidates(q, np.zeros((1, 4096)), np.zeros((1, 10)), {})
    assert not result.empty and result.molecule_id.eq('none').all()
    with pytest.raises(ValueError, match='outside'):
        fr.rank_candidates(q, np.full((1, 4096), 2.), np.zeros((1, 10)), {})


def test_formula_counts_tolerates_extra_characters():
    np.testing.assert_array_equal(fr.formula_counts('[C2H6O]+.Na'), [2, 6, 0, 1, 0, 0, 0, 0, 0, 0])
    assert not fr.formula_counts('').any()
    assert not fr.formula_counts(None).any()
    assert not fr.formula_counts('C0').any()


def test_vectorized_proxy_matches_brute_force_with_ties():
    rng = np.random.default_rng(9)
    cache = dict(x=rng.integers(0, 5, (250, 6)).astype(float),
                 row_ids=rng.permutation(250), query=np.repeat(np.arange(5), 50),
                 is_hit=rng.random(250) < .05, n_queries=6)  # include an empty query
    cfg = fr.config({})
    scores = fr.composite(cache['x'], cfg)
    expected = 0.
    for i in range(6):
        rows = np.flatnonzero(cache['query'] == i)
        ordered = rows[np.lexsort((cache['row_ids'][rows], -scores[rows]))][:25]
        hits = np.flatnonzero(cache['is_hit'][ordered])
        if len(hits):
            expected += 1/(hits[0]+1)
    assert fr.cache_mrr(cache, cfg) == pytest.approx(expected/6)


def test_tuning_samples_once_and_never_finalizes(candidate_api, monkeypatch):
    q = pd.DataFrame(dict(molecule_id=[f'q{i}' for i in range(10)], neutral_mass=[100.]*10,
                          split3=['val_model']*10))
    observed = []
    original = fr.build_tuning_cache
    def capture(*args):
        observed.append(list(args[-1]))
        return original(*args)
    monkeypatch.setattr(fr, 'build_tuning_cache', capture)
    def no_finalize(*args, **kwargs):
        pytest.fail('Canonicalization during tuning')
    monkeypatch.setattr(candidate_api, 'finalize_candidates', no_finalize)
    kwargs = dict(queries=q, fp=np.full((10, 4096), .5), elem=np.zeros((10, 10)),
                  truth_keys=['B']*10, cfg=dict(exclude_keys={'held_out'}, max_tune_queries=2),
                  max_tune_queries=3, sweeps=1)
    result = fr.tune(**kwargs)
    repeat = fr.tune(**kwargs)
    assert observed[0] == observed[1]
    assert len(observed[0]) == result['tuning_queries'] == 2
    assert result == repeat
    assert len(result['search_history']) == 9
    assert 'exclude_keys' not in result
    assert result['overall_proxy_mrr25'] == result['tuning_proxy_mrr25']
    assert result['hit_queries_20ppm'] == 2
    assert result['covered_hit_mrr25'] == result['overall_proxy_mrr25']


def test_tuning_defaults_and_smooth_coordinate_grid(candidate_api, monkeypatch):
    import inspect
    assert fr.DEFAULT_CONFIG['max_tune_queries'] == 5000
    assert inspect.signature(fr.tune).parameters['max_tune_queries'].default == 5000
    assert inspect.signature(fr.tune).parameters['sweeps'].default == 3
    q = pd.DataFrame(dict(molecule_id=['q'], neutral_mass=[100.], split3=['val_model']))
    seen = []
    def tied_score(cache, cfg):
        seen.append(tuple(cfg['weights']))
        return 0.0
    monkeypatch.setattr(fr, 'cache_mrr', tied_score)
    result = fr.tune(q, np.full((1, 4096), .5), np.zeros((1, 10)), ['B'], dict(exclude_keys=set()))
    # No strict improvement: retain the origin, while evaluating every proposed step.
    assert result['weights'] == fr.DEFAULT_CONFIG['weights']
    for j in (2, 4, 6, 7):
        origin = fr.DEFAULT_CONFIG['weights'][j]
        values = {weights[j] for weights in seen}
        assert {0., origin*.5, origin*.75, origin*1.25, max(origin*2., .1)} <= values


def test_tuning_smaller_window_gets_fallback_even_when_twenty_is_dense(candidate_api):
    candidate_api.frame['mass'] = [99.999, 100., 100.0015, 100.0019, 100.04]
    candidate_api.context()['mass'] = candidate_api.frame.mass.to_numpy()
    cfg = fr.config(dict(exclude_keys={'held_out'}, min_candidates=3))
    q = pd.DataFrame(dict(molecule_id=['q'], neutral_mass=[100.], split3=['val_model']))
    cache = fr.build_tuning_cache(q, np.full((1, 4096), .5), np.zeros((1, 10)), ['B'], cfg, [0])
    assert len(fr.tuning_window(cache, 10., 1, cfg)['row_ids']) == 4
    assert len(fr.tuning_window(cache, 20., 1, cfg)['row_ids']) == 3


def test_nearest_fallback_finds_fifty_eligible_rows(candidate_api):
    frame = pd.concat([candidate_api.frame.iloc[[0]]]*120, ignore_index=True)
    frame['row_id'] = np.arange(120)
    frame['mass'] = np.arange(120)+100.
    frame['source_mask'] = np.where(frame.row_id % 3 == 0, 1, 3)
    candidate_api.frame = frame
    state = candidate_api.context()
    state['mass'], state['order'] = frame.mass.to_numpy(), np.arange(120)
    mass = 159.3
    got = pd.concat(fr.nearest_candidate_batches(mass, set(range(120)), fr.config({})))
    eligible = frame.loc[frame.source_mask.ne(1)]
    expected = eligible.iloc[np.argsort(np.abs(eligible.mass.to_numpy()-mass))[:50]]
    assert got.row_id.tolist() == expected.row_id.tolist()
    assert got.source_mask.eq(2).all()


def test_empty_database_reports_clear_error(candidate_api):
    candidate_api.context()['mass'] = np.empty(0)
    candidate_api.context()['order'] = np.empty(0, dtype=int)
    with pytest.raises(ValueError, match='no eligible rows'):
        list(fr.nearest_candidate_batches(100., set(), fr.config({})))


def notebook_function(name, namespace):
    from build_06_fp_rank_infer import build_notebook
    for cell in build_notebook()['cells']:
        if cell['cell_type'] == 'code':
            py_lines = "\n".join(l for l in cell['source'].splitlines() if not l.strip().startswith(("!", "%")))
            if not py_lines.strip():
                continue
            for node in ast.parse(py_lines).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), namespace)
                    return namespace[name]
    raise AssertionError(name)


def test_locate_prefers_working_then_newest():
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='stage06_test_', ignore_cleanup_errors=True) as temp:
        root = Path(temp)
        work, inputs = root/'working', root/'input'
        work.mkdir()
        inputs.mkdir()
        for parent, timestamp in [(work/'old', 1), (work/'new', 2), (inputs/'newest', 3)]:
            parent.mkdir()
            path = parent/'test.txt'
            path.write_text('fixture')
            os.utime(path, (timestamp, timestamp))
        def mapped_path(value):
            return inputs if value == '/kaggle/input' else Path(value)
        locate = notebook_function('locate', dict(WORK=work, Path=mapped_path))
        assert locate('test.txt') == work/'new/test.txt'
        assert locate('test.txt', inputs/'newest') == inputs/'newest/test.txt'
        with pytest.raises(FileNotFoundError):
            locate('missing.txt')


def test_exclusions_without_build_sqlite(monkeypatch):
    splits = pd.DataFrame(dict(molecule_id=['a', 'b', 'c'], split3=['train', 'val_model', 'val_rank'],
                               ik14=['TRAIN', 'VM', None]))
    train = pd.DataFrame(dict(molecule_id=['a', 'b', 'b', 'c'], smiles=['train', 'held1', 'held1', 'held2'],
                              ik14=['TRAIN2', None, None, 'VR']))
    frames = {'splits': splits, 'train': train}
    monkeypatch.setattr(pd, 'read_parquet', lambda path, columns: frames[path][columns].copy())
    pq = types.SimpleNamespace(ParquetFile=lambda path: types.SimpleNamespace(
        schema_arrow=types.SimpleNamespace(names=list(frames[path]))))
    func = notebook_function('heldout_exclusions', dict(pd=pd, pq=pq,
        candidates=types.SimpleNamespace(standardize_mol=lambda s: 'STD_'+s),
        evalkit=types.SimpleNamespace(fast_key=lambda s: 'FAST_'+s)))
    assert func('splits', 'train') == {'FAST_held1', 'FAST_held2', 'FAST_STD_held1', 'FAST_STD_held2', 'VM', 'VR'}


def test_truth_for_handles_duplicate_molecule_ids_across_spectra(monkeypatch):
    queries = pd.DataFrame(dict(molecule_id=['m1', 'm2']))
    train = pd.DataFrame(dict(
        molecule_id=['m1', 'm1', 'm2'],
        smiles=['C1=CC=CC=C1', 'c1ccccc1', 'CCO']
    ))
    monkeypatch.setattr(pd, 'read_parquet', lambda path, columns: train[columns].copy())
    tqdm_fn = lambda it, **kwargs: it
    func = notebook_function('truth_for', dict(pd=pd, train_path='train', tqdm=tqdm_fn,
        truth_by_smiles={}, fast_truth_by_smiles={},
        evalkit=types.SimpleNamespace(canon_ik14=lambda s: 'CANON_' + s, fast_key=lambda s: 'FAST_' + s),
        candidates=types.SimpleNamespace(standardize_mol=lambda s: s)))
    keys, fast_keys = func(queries, include_fast=True)
    assert len(keys) == 2
    assert len(fast_keys) == 2


def test_notebook_builds_with_compilable_cells():
    from build_06_fp_rank_infer import build_notebook
    book = build_notebook()
    assert book['nbformat'] == 4
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            py_lines = "\n".join(l for l in cell['source'].splitlines() if not l.strip().startswith(("!", "%")))
            if py_lines.strip():
                ast.parse(py_lines)
    embedded = ast.parse(book['cells'][5]['source']).body[0].value.args[0].value
    assert embedded == find_path('fprank.py').read_text(encoding='utf-8')


def test_manifest_stage09_contract():
    from build_06_fp_rank_infer import build_notebook
    code = build_notebook()['cells'][15]['source']
    names = {'vm_ids', 'vr_ids', 'manifest'}
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    ns = dict(train_queries=pd.DataFrame(dict(molecule_id=['z', 'a', 'b'], split3=['val_rank', 'train', 'val_model'])),
              hashlib=hashlib, metadata={'build_id': 'id'}, evalkit=types.SimpleNamespace(FP_SPEC={}),
              db_root='db', pred_root=Path('pred'), train_path='train', test_path='test', split_path='splits',
              summaries={}, stats={}, time=types.SimpleNamespace(time=lambda: 0), STARTED=0,
              sha256=lambda path: 'hash', WORK=Path('.'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<manifest>', 'exec'), ns)
    assert ns['manifest']['trained_on_split'] == 'train'
    assert ns['manifest']['heldout_ids_hash'] == hashlib.sha256(b'bz').hexdigest()


@pytest.fixture
def chemistry_api(candidate_api):
    """Real SQLite/cache functions with only chemistry and Parquet I/O substituted."""
    api = candidate_api
    names = {'get_fingerprints', 'canonical_key', 'memory_cache_put', 'cache_written',
             'flush_cache', 'prefetch_canonical_keys', 'batched'}
    tree = ast.parse(find_path('candidates.py').read_text(encoding='utf-8'))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    api.itertools = itertools
    api._MEM_IK14_CACHE, api._MEM_FP_CACHE = OrderedDict(), OrderedDict()
    api.CTX = api.context()
    api.CTX.update(valid=np.zeros(5, dtype=bool), fp=np.zeros((5, 533), dtype=np.uint8),
                   cache_pending=0, cache_commit_rows=5000, fp_cache_size=100, ik14_cache_size=100)
    api.CTX['meta'].update(fp_bytes=533, max_materialized_rows=2048)
    api.CTX['cache'].execute('CREATE TABLE chemistry(row_id INTEGER PRIMARY KEY, ik14 TEXT, fp BLOB)')
    api.sql = []
    api.CTX['cache'].set_trace_callback(api.sql.append)
    api.fp_calls, api.canon_calls = [], []
    def fingerprint_chunk(rows):
        api.fp_calls.append(rows)
        return [(int(i), np.full(533, i, dtype=np.uint8)) for i, _ in rows]
    def canonicalize(smiles):
        api.canon_calls.append(smiles)
        return 'canon_'+smiles
    api.fingerprint_chunk = fingerprint_chunk
    api.Chem = types.SimpleNamespace(MolFromSmiles=lambda s: s)
    api.load_evalkit = lambda: types.SimpleNamespace(canon_ik14=canonicalize)
    exec(compile(selected, 'candidates.py', 'exec'), api.__dict__)
    return api


def test_lazy_fingerprints_batch_smiles_and_memory_hits(chemistry_api, monkeypatch):
    api = chemistry_api
    monkeypatch.setattr(api, 'fetch_rows', lambda ids: pytest.fail('Unexpected Parquet read'))
    ids = np.array([1, 3, 1], dtype=np.uint64)
    result = api.get_fingerprints(ids, smiles=['C1', 'C3', 'C1'])
    assert len(api.fp_calls) == 1 and api.fp_calls[0] == [(1, 'C1'), (3, 'C3')]
    np.testing.assert_array_equal(result[:, 0], ids)
    assert not any(s.startswith('COMMIT') for s in api.sql)
    before = len(api.sql)
    np.testing.assert_array_equal(api.get_fingerprints(ids, smiles=['C1', 'C3', 'C1']), result)
    assert len(api.sql) == before  # all overlapping-window hits served from RAM
    api.flush_cache()
    assert sum(s.startswith('COMMIT') for s in api.sql) == 1
    api._MEM_FP_CACHE.clear()
    np.testing.assert_array_equal(api.get_fingerprints(ids), result)
    assert len(api.fp_calls) == 1  # one batch SELECT reads persisted fingerprints


def test_lazy_fingerprints_fetch_missing_rows_once(chemistry_api, monkeypatch):
    api = chemistry_api
    fetches = []
    original = api.fetch_rows
    def fetch(ids):
        fetches.append(list(ids))
        return original(ids)
    monkeypatch.setattr(api, 'fetch_rows', fetch)
    api.get_fingerprints([1, 2, 4])
    assert fetches == [[1, 2, 4]]
    assert len(api.fp_calls) == 1


def test_chemistry_commit_threshold_and_bounded_cache(chemistry_api):
    api = chemistry_api
    api.CTX.update(cache_commit_rows=3, fp_cache_size=2)
    api.get_fingerprints([1, 2], smiles=['C', 'CC'])
    assert api.CTX['cache_pending'] == 2
    api.get_fingerprints([3], smiles=['CCC'])
    assert api.CTX['cache_pending'] == 0
    assert sum(s.startswith('COMMIT') for s in api.sql) == 1
    assert len(api._MEM_FP_CACHE) == 2


def test_only_twentyfive_canonical_hits_and_tail_schema(chemistry_api):
    api = chemistry_api
    frame = pd.concat([api.frame.iloc[[0]]]*120, ignore_index=True)
    frame['row_id'] = np.arange(120)
    frame['smiles'] = [str(i) for i in range(120)]
    frame['fast_key'] = [f'fast_{i}' for i in range(120)]
    frame['ik14'] = None
    frame['source_mask'] = np.array([1, 2**63+1]*60, dtype=np.uint64)
    api.CTX['ik14_cache_size'] = 200
    final = api.finalize_candidates(frame, limit=100, canonical_limit=25)
    assert len(final) == 100 and len(api.canon_calls) == 25
    assert final.canon_ik14.iloc[:25].notna().all()
    assert final.canon_ik14.iloc[25:].isna().all()
    assert final.identity_kind.iloc[25:].eq('fast_only').all()
    assert int(final.source_mask.iloc[1]) == 2**63+1
    assert sum(s.startswith('SELECT') for s in api.sql) == 1
    before = len(api.sql)
    api.finalize_candidates(frame, limit=100, canonical_limit=25)
    assert len(api.sql) == before and len(api.canon_calls) == 25
    assert not any(s.startswith('COMMIT') for s in api.sql)


def test_canonical_prefix_counts_unique_hits_not_raw_rows(chemistry_api):
    api = chemistry_api
    frame = pd.concat([api.frame.iloc[[0]]]*60, ignore_index=True)
    frame['row_id'] = np.arange(60)
    frame['smiles'] = [str(i//2) for i in range(60)]
    frame['fast_key'] = [str(i) for i in range(60)]
    frame['ik14'] = None
    result = api.finalize_candidates(frame, limit=100, canonical_limit=25)
    assert result.canon_ik14.head(25).nunique() == 25
    assert len(api.canon_calls) == 49


def test_normalized_output_preserves_high_bit_masks(candidate_api):
    candidate_api.frame['source_mask'] = np.array([1, 2**63+3, 4, 8, 2], dtype=np.uint64)
    q = pd.DataFrame(dict(molecule_id=['q'], neutral_mass=[100.], split3=['val_model']))
    out = fr.rank_candidates(q, np.full((1, 4096), .5), np.zeros((1, 10)), dict(exclude_keys={'held'}))
    row = out.loc[out.row_id.eq(1)].iloc[0]
    assert int(row.source_mask) == 2**63+6
    assert int(row.scoring_source_mask) == 2**63+2
    assert {col: str(dtype) for col, dtype in out.dtypes.items()} == fr.OUTPUT_DTYPES


def test_cache_configure_wal_normal_and_reset(chemistry_api):
    api = chemistry_api
    tree = ast.parse(find_path('candidates.py').read_text(encoding='utf-8'))
    names = {'configure', 'open_sql'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), 'candidates.py', 'exec'), api.__dict__)
    api.Path, api.json, api.sqlite3 = Path, json, sqlite3
    api.load_evalkit = lambda: types.SimpleNamespace(FP_SPEC={})
    api.map_read = lambda path: np.zeros(1)
    api.pq = types.SimpleNamespace(ParquetFile=lambda path: None)
    # Original in-memory fixture connection is kept for its fixture teardown.
    api.CTX = None
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='stage06_cache_', ignore_cleanup_errors=True) as temp:
        root = Path(temp)
        (root/'db_metadata.json').write_text(json.dumps(dict(fp_spec={}, build_id='test')))
        with sqlite3.connect(root/'index.sqlite') as db:
            db.execute('CREATE TABLE aliases(row_id INTEGER)')
        db.close()
        try:
            api.configure(root, cache_dir=root/'cache')
            assert api.CTX['cache'].execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
            assert api.CTX['cache'].execute('PRAGMA synchronous').fetchone()[0] == 1
            api._MEM_IK14_CACHE[1] = 'old'
            api._MEM_FP_CACHE[1] = b'old'
            api.CTX['cache'].execute("INSERT INTO chemistry(row_id,ik14) VALUES (1,'persist')")
            api.configure(root, cache_dir=root/'cache')
            assert not api._MEM_IK14_CACHE and not api._MEM_FP_CACHE
            assert api.CTX['cache'].execute('SELECT ik14 FROM chemistry WHERE row_id=1').fetchone()[0] == 'persist'
        finally:
            api.flush_cache()
            api.CTX['index'].close()
            api.CTX['cache'].close()
            api.CTX = None


def test_validate_inputs_maccs():
    q = pd.DataFrame(dict(molecule_id=['q1', 'q2'], neutral_mass=[100., 200.]))
    fp = np.full((2, 4096), 0.5)
    elem = np.zeros((2, 10))
    maccs_good = np.full((2, 167), 0.5)
    fr.validate_inputs(q, fp, elem, maccs=maccs_good)
    with pytest.raises(ValueError, match='MACCS prediction shape mismatch'):
        fr.validate_inputs(q, fp, elem, maccs=np.full((2, 168), 0.5))
    with pytest.raises(ValueError, match='MACCS prediction shape mismatch'):
        fr.validate_inputs(q, fp, elem, maccs=np.full((1, 167), 0.5))
    with pytest.raises(ValueError, match='outside'):
        fr.validate_inputs(q, fp, elem, maccs=np.full((2, 167), 2.0))


def test_rank_candidates_maccs_impact(candidate_api):
    queries = pd.DataFrame(dict(molecule_id=['q'], neutral_mass=[100.]))
    p = np.full((1, 4096), 0.5)
    e = np.zeros((1, 10))
    def custom_fp(ids, smiles=None):
        out = np.zeros((len(ids), 533), dtype=np.uint8)
        for i, row_id in enumerate(ids):
            if row_id == 3:
                out[i, 512] = 1  # MACCS bit 0
            elif row_id == 4:
                out[i, 512] = 2  # MACCS bit 1
        return out
    candidate_api.get_fingerprints = custom_fp
    cfg = dict(ppm=1000.0, min_candidates=1, fallback_da=0.05, weights=[0., 0., 0., 0., 0., 0., 0., 1.])
    maccs_pred = np.zeros((1, 167), dtype=np.float32)
    maccs_pred[0, 0] = 1.0
    res1 = fr.rank_candidates(queries, p, e, cfg, maccs_preds=maccs_pred)
    assert res1.iloc[0].row_id == 3
    maccs_pred[0, 0] = 0.0
    maccs_pred[0, 1] = 1.0
    res2 = fr.rank_candidates(queries, p, e, cfg, maccs_preds=maccs_pred)
    assert res2.iloc[0].row_id == 4


def test_tune_with_maccs(candidate_api):
    q = pd.DataFrame(dict(molecule_id=[f'q{i}' for i in range(4)], neutral_mass=[100.]*4,
                          split3=['val_model']*4))
    kwargs = dict(queries=q, fp=np.full((4, 4096), .5), elem=np.zeros((4, 10)),
                  truth_keys=['B']*4, cfg=dict(exclude_keys={'held_out'}, max_tune_queries=2),
                  max_tune_queries=2, sweeps=1, maccs=np.full((4, 167), .5))
    result = fr.tune(**kwargs)
    assert 'weights' in result
    assert len(result['weights']) == 8
    assert result['tuned_on'] == 'val_model'


def test_top100_recall_matches_fast_tail_and_canonical_prefix():
    from build_06_fp_rank_infer import build_notebook
    source = build_notebook()['cells'][13]['source']
    assignment = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Subscript)
                      and isinstance(n.targets[0].value, ast.Name)
                      and n.targets[0].value.id == 'in_pool')
    code = compile(ast.Module(body=[assignment], type_ignores=[]), '<top100>', 'exec')
    for rank, canonical, fast, expected in [
            (30, 'FAST_TRUTH', 'FAST_TRUTH', True),
            (5, 'CANON_TRUTH', 'OTHER', True),
            (101, 'FAST_TRUTH', 'FAST_TRUTH', False),
            (30, 'OTHER', 'OTHER', False)]:
        frame = pd.DataFrame(dict(ik14=['OTHER']*101, fast_key=['OTHER']*101))
        frame.loc[rank-1, ['ik14', 'fast_key']] = [canonical, fast]
        ns = dict(frame=frame, qid='q', truth={'q': 'CANON_TRUTH'},
                  fast_truth={'q': 'FAST_TRUTH'}, in_pool={})
        exec(code, ns)
        assert ns['in_pool']['q'] is expected


def test_saved_stage06_candidate_source_and_metrics_are_current():
    from build_06_fp_rank_infer import build_notebook
    saved = json.loads(find_path('06_fp_rank_infer.ipynb').read_text(encoding='utf-8'))
    generated = build_notebook()
    for i in (4, 12, 13):
        assert ''.join(saved['cells'][i]['source']) == generated['cells'][i]['source']
    embedded = ast.parse(''.join(saved['cells'][4]['source'])).body[0].value.args[0].value
    assert embedded == find_path('candidates.py').read_text(encoding='utf-8')
    stage04_tree = ast.parse(find_path('build_04_candidate_db.py').read_text(encoding='utf-8'))
    stage04_source = next(ast.literal_eval(n.value) for n in stage04_tree.body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                             t.id == 'MODULE_SOURCE' for t in n.targets))
    for name in ('discover_sources', 'ingest_metadata', 'configure', 'flush_cache',
                 'get_fingerprints', 'finalize_candidates'):
        definitions = [next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)
                            and n.name == name) for s in (embedded, stage04_source)]
        assert ast.dump(definitions[0]) == ast.dump(definitions[1])


def test_ingest_metadata_global_median_and_consensus_smiles():
    tree = ast.parse(find_path('candidates.py').read_text(encoding='utf-8'))
    func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ingest_metadata')
    adapter_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SchemaAdapter')

    # 4 spectra for m1: 3 normal at 300.0 with SMILES CCO, 1 outlier at 500.0 with SMILES C(=O)O
    data = pd.DataFrame(dict(
        molecule_id=['m1', 'm1', 'm1', 'm1', 'm2'],
        smiles=['CCO', 'CCO', 'C(=O)O', 'CCO', 'CCN'],
        neutral_mass=[300.0, 300.0, 500.0, 300.0, 100.0],
        precursor_mz=[301.0, 301.0, 501.0, 301.0, 101.0],
        adduct=['[M+H]+']*5,
        polarity=['+']*5
    ))

    class MockTable:
        def __init__(self, df): self.df = df
        def to_pandas(self): return self.df.copy()

    class MockParquetFile:
        def __init__(self, path):
            self.schema_arrow = types.SimpleNamespace(names=list(data.columns))
        def read(self, columns=None):
            return MockTable(data[columns] if columns else data)

    db = sqlite3.connect(':memory:')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS molecules(id TEXT PRIMARY KEY, smiles TEXT,
        mass_sum REAL, n INTEGER, lo REAL, hi REAL, split3 TEXT, ik14 TEXT);
    CREATE TABLE IF NOT EXISTS tests(id TEXT PRIMARY KEY, smiles TEXT,
        mass_sum REAL, n INTEGER, lo REAL, hi REAL, split3 TEXT);
    ''')

    scope = dict(
        pq=types.SimpleNamespace(ParquetFile=MockParquetFile),
        pd=pd, np=np, json=json,
        state_get=lambda db, k: False,
        state_set=lambda db, k, v: None,
        observed_mass=lambda f: f['neutral_mass'].to_numpy(dtype=float),
        resource_guard=lambda cfg: None,
    )
    exec(compile(ast.Module(body=[adapter_node, func_node], type_ignores=[]), '<ingest>', 'exec'), scope)
    scope['ingest_metadata'](db, 'dummy_train.parquet', 'train', types.SimpleNamespace())

    row_m1 = db.execute("SELECT smiles, mass_sum, n, lo, hi, mass_sum/n FROM molecules WHERE id='m1'").fetchone()
    assert row_m1[0] == 'CCO'
    assert row_m1[1] == 1200.0
    assert row_m1[2] == 4
    assert row_m1[3] == 300.0
    assert row_m1[4] == 500.0
    assert row_m1[5] == 300.0


