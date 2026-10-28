"""CPU checks for Stage 08's nontrivial inference contracts."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import shutil
import subprocess
import time
import hashlib

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import spec2smiles_infer as infer


class Stage08Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(ppm=10., relax_ppm=50., elements_order=infer.ELEMENTS,
            presence_weight=.5, prior_weight=.15, beam=25, num_return=25,
            samples=50, temperature=.8, top_p=.9, length_penalty=.7, max_length=8)
        infer.seed_all(42)

    def test_enumeration_matches_bruteforce(self):
        stats = dict(allowed=['C','H','N','O'], bounds=dict(C=5,H=15,N=2,O=4),
            ratios=dict(C=[1,1], H=[0,4], N=[0,2], O=[0,4]), prior={})
        for target in [46.041864812, 75.032028, 89.047679, 100.1234]:
            expected = set()
            for c, h, n, o in itertools.product(range(1,6),range(16),range(3),range(5)):
                d = dict(C=c,H=h,N=n,O=o)
                mass = sum(infer.MASS[e]*v for e,v in d.items())
                if abs(mass-target)/target*1e6 <= 10 and infer.sane_formula(d,stats):
                    expected.add(infer.formula_text(d))
            self.assertEqual(expected, set(infer.enumerate_formulas(target,stats,10.)))

    def test_duplicate_count_and_conditional_relaxation(self):
        paths = [('CCO',-1.,'beam'),('OCC',-3.,'sample'),('C',-2.,'sample'),('',-2.,'sample')]
        mass = 46.041864812
        rows, diag = infer.postprocess('q',paths,[(mass,'x')],[],self.cfg)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['count'],2)
        self.assertEqual(rows[0]['log_prob'],-1.)
        self.assertEqual(rows[0]['mean_log_prob'],-2.)
        self.assertEqual(diag['valid'],3)
        self.assertFalse(diag['no_strict_survivor'])
        rows, diag = infer.postprocess('q',paths,[(mass*(1+30e-6),'x')],[],self.cfg)
        self.assertTrue(rows[0]['mass_relaxed'])
        self.assertTrue(diag['no_strict_survivor'])
        rows, diag = infer.postprocess('q',paths,[(100.,'x')],[],self.cfg)
        self.assertEqual(rows,[])
        self.assertTrue(diag['no_survivor'])

    def test_tautomers_merge(self):
        a = infer.canonical_record('CC(=O)C')
        b = infer.canonical_record('C=C(O)C')
        self.assertEqual(a[1], b[1])

    def test_adducts_charge_and_unknown(self):
        frame = pd.DataFrame(dict(precursor_mz=[101.007276], adduct=['[M+2H]2+'], polarity=['positive']))
        masses, unknown = infer.mass_hypotheses(frame)
        self.assertAlmostEqual(masses[0][0],200., places=6)
        frame.adduct = 'unknown'
        masses, unknown = infer.mass_hypotheses(frame)
        self.assertEqual(len(masses),3)
        self.assertEqual(unknown,['unknown'])

    def test_id_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'molecule_id':['b','a']}).to_parquet(path/'ids.parquet')
            np.save(path/'elem.npy',np.array([[2]*10,[1]*10]))
            paths = dict(ids_test=str(path/'ids.parquet'),elem_test=str(path/'elem.npy'))
            result = infer.prediction_map(paths,'test',['a','b'],self.cfg)
            self.assertEqual(result['a'][0],1)
            self.assertEqual(result['b'][0],2)
            with self.assertRaises(ValueError):
                infer.prediction_map(paths,'test',['missing'],self.cfg)

    def test_metric_denominator_includes_empty_queries(self):
        frame = pd.DataFrame(dict(molecule_id=['a','a','b'],ik14=['wrong','yes','no']))
        metrics = infer.ranking_metrics(frame,{'a':'yes','b':'yes','c':'yes'})
        self.assertEqual(metrics['mrr25'],1/6)
        self.assertEqual(metrics['top1'],0)
        self.assertEqual(metrics['top5'],1/3)

    def test_full_decoders_and_unfinished_samples(self):
        class Decoder(torch.nn.Module):
            def decode(self,tokens,memory,mask):
                logits = torch.full((*tokens.shape,8),-20.)
                logits[:,-1,[4,4,5,2][min(tokens.shape[1]-1,3)]] = 10.
                return logits
        memory, mask = torch.zeros(2,4,8),torch.zeros(2,4,dtype=torch.bool)
        b = infer.decode_beam(Decoder(),memory,mask,self.cfg)
        s = infer.decode_sample(Decoder(),memory,mask,self.cfg)
        self.assertEqual([len(x) for x in b],[25,25])
        self.assertEqual([len(x) for x in s],[50,50])
        self.assertEqual(s[0][0][0][:4],[4,4,5,2])
        short = infer.decode_sample(Decoder(),memory,mask,dict(self.cfg,max_length=2))
        self.assertTrue(all(ids == [] for group in short for ids, _ in group))

    def test_notebook_all_cells_smoke(self):
        notebook = nbformat.read(ROOT/'notebooks/08_spec2smiles_infer.ipynb',as_version=4)
        nbformat.validate(notebook)
        with tempfile.TemporaryDirectory() as directory:
            namespace = {}
            for i,cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                self.assertLess(len(cell.source.splitlines()),60)
                exec(compile(cell.source,f'cell_{i}','exec'),namespace)
                if cell.source.startswith('CFG ='):
                    namespace['CFG'].update(SMOKE_TEST=True,work=directory,
                        out=str(Path(directory)/'out'),cache=str(Path(directory)/'cache'))

    def test_two_real_worker_processes_and_resume(self):
        import spec2smiles as s2s
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            out = work/'out'
            out.mkdir()
            for name in ('spec2fp.py','spec2smiles.py','spec2smiles_infer.py','evalkit.py'):
                shutil.copyfile(ROOT/'src'/name,work/name)
            cfg = dict(self.cfg,seed=42,SMOKE_TEST=True,work=str(work),out=str(out),
                cache=str(work/'cache'),resume=True,batch_size=2,worker_ram_gb=3.5,max_length=6)
            tokenizer = s2s.SmilesTokenizer(vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'C':4,'O':5})
            tokenizer.save(work/'smiles_vocab.json')
            model_cfg = dict(encoder=dict(d_model=32,n_layers=1,n_heads=4,feat_dim=17),
                decoder_layers=1,vocab_size=6,top_k_peaks=4,mz_dim=8)
            model = s2s.Spec2SmilesModel(model_cfg)
            torch.save({'model_state_dict':model.state_dict()},work/'model.pt')
            (work/'config.json').write_text(json.dumps(model_cfg))
            stats = dict(allowed=['C','H','O'],bounds=dict(C=3,H=10,O=2),
                ratios=dict(C=[1,1],H=[0,4],O=[0,2]),prior={'C2H6O':5})
            (work/'formulas.json').write_text(json.dumps(stats))
            paths = dict(model=str(work/'model.pt'),config=str(work/'config.json'),
                vocab=str(work/'smiles_vocab.json'),formula_stats=str(work/'formulas.json'))
            ids, parts = {}, {}
            for name in ('val_model','val_rank','test'):
                ids[name] = [f'{name}_a',f'{name}_b']
                parts[name] = {}
                path = work/f'ids_{name}.parquet'
                pd.DataFrame({'molecule_id':ids[name]}).to_parquet(path,index=False)
                paths[f'ids_{name}'] = str(path)
                paths[f'elem_{name}'] = str(work/f'elem_{name}.npy')
                np.save(paths[f'elem_{name}'],np.array([[2,6,0,1,0,0,0,0,0,0]]*2))
                for shard in (0,1):
                    folder = work/'cache'/name/str(shard)
                    folder.mkdir(parents=True)
                    path = folder/'00000.parquet'
                    frame = pd.DataFrame([dict(molecule_id=ids[name][shard],mz=[10.,20.],
                        inten=[.5,1.],precursor_mz=47.049140812,neutral_mass=46.041864812,
                        adduct='[M+H]+',ce=20.,instrument='qtof',polarity='positive')])
                    frame.to_parquet(path,index=False)
                    parts[name][str(shard)] = [str(path)]
            job = dict(cfg=cfg,paths=paths,ids=ids,parts=parts,run_hash='fixture')
            job_path = work/'job.json'
            job_path.write_text(json.dumps(job))
            procs = [subprocess.Popen([sys.executable,str(work/'spec2smiles_infer.py'),
                str(job_path),str(shard)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for shard in (0,1)]
            for process in procs:
                output,_ = process.communicate(timeout=90)
                self.assertEqual(process.returncode,0,output)
            markers = sorted(out.glob('*.done.json'))
            self.assertEqual(len(markers),6)
            before = {p.name:p.stat().st_mtime_ns for p in markers}
            result = subprocess.run([sys.executable,str(work/'spec2smiles_infer.py'),str(job_path),'0'],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in markers})
            # Force one OOM to exercise the real retry path, preserving decoding settings.
            job['cfg']['resume'] = False
            job_path.write_text(json.dumps(job))
            original_generate = infer.generate_chunk
            attempted = []
            def fail_once(*args):
                attempted.append((args[-1]['beam'],args[-1]['samples']))
                if len(attempted) == 1:
                    raise torch.OutOfMemoryError('synthetic OOM')
                return original_generate(*args)
            with mock.patch.object(infer,'generate_chunk',side_effect=fail_once):
                infer.run_worker(job_path,0)
            self.assertTrue(all(pair == (25,50) for pair in attempted))
            self.assertEqual(json.loads((out/'val_model_shard0_00000.done.json').read_text())['batch_size'],1)
            for name in ids:
                frames = [pd.read_parquet(out/f'{name}_shard{s}_00000.diag.parquet') for s in (0,1)]
                merged = pd.concat(frames)
                self.assertEqual(set(merged.molecule_id),set(ids[name]))
                self.assertTrue(merged.attempted.ge(50).all())
            # Execute the actual notebook merge/evaluation/export cells on worker results.
            heldout = ids['val_model'] + ids['val_rank']
            pd.DataFrame(dict(molecule_id=heldout,smiles=['CCO']*len(heldout))).to_parquet(work/'truth.parquet')
            from collections import Counter
            from tqdm.auto import tqdm
            import evalkit
            cfg.update(SMOKE_TEST=False,alpha_grid=[0.,.25],formula_weight_grid=[0.,.1],
                       stage10_generator_budget_s=8*3600)
            namespace = dict(CFG=cfg,ids=ids,OUT=out,job=job,pd=pd,np=np,infer=infer,
                Path=Path,Counter=Counter,tqdm=tqdm,json=json,hashlib=hashlib,time=time,
                torch=torch,Chem=infer.Chem,evalkit=evalkit,run_hash='fixture',
                measured_inference_seconds=1.,START=time.perf_counter(),gpu_count=2,
                input_hashes={},source_hashes={},
                train_adapter=infer.SchemaAdapter(work/'truth.parquet',['molecule_id','smiles']),
                split_map=pd.Series({mid:name for name in ('val_model','val_rank') for mid in ids[name]}))
            nb = nbformat.read(ROOT/'notebooks/08_spec2smiles_infer.ipynb',as_version=4)
            start = next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'raw, diagnostics, formula_tables = ' in c.source)
            for cell in nb.cells[start:]:
                if cell.cell_type != 'code':
                    continue
                if 'for module_name in' in cell.source:
                    continue
                try:
                    exec(cell.source,namespace)
                except AssertionError as error:
                    # Random model may have no mass-consistent survivor: this must be explicit.
                    self.assertIn('Incomplete generator submission',str(error))
            for name in ids:
                candidates = pd.read_parquet(out/f'cand_gen_{name}.parquet')
                self.assertTrue(set(['molecule_id','smiles','ik14','score','count','source']) <= set(candidates))
            self.assertTrue((out/'manifest.json').is_file())
            self.assertTrue((out/'submission_gen.csv').is_file())


if __name__ == '__main__':
    unittest.main()
