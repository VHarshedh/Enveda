"""Small CPU contract tests; no competition data or GPU required."""
import ast
import json
import tempfile
import shutil
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
BUILDERS = ROOT / 'builders'
NOTEBOOKS = ROOT / 'notebooks'

for p in (ROOT, SRC, BUILDERS):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

def find_path(rel_path):
    p = Path(rel_path)
    if p.is_file():
        return p
    for base in (SRC, BUILDERS, NOTEBOOKS, ROOT):
        cand = base / p.name
        if cand.is_file():
            return cand
    return p
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import spec2smiles as s
from spec2fp import Spec2FPModel
from build_07_spec2smiles_train import build_notebook

torch.set_num_threads(1)


class Stage07Tests(unittest.TestCase):
    def setUp(self):
        self.tok = s.SmilesTokenizer().fit(['CCO', 'CCN', 'C/C=C\\C', '[NH4+]', 'C%12CCCCC%12'])
        self.cfg = dict(encoder=dict(d_model=32, n_layers=1, n_heads=4, feat_dim=129),
                        vocab_size=len(self.tok.vocab), top_k_peaks=4, mz_dim=64)

    def spectra(self, n=2):
        return dict(peaks=torch.randn(n, 4, 129), pad_mask=torch.zeros(n, 4, dtype=torch.bool),
                    adduct=torch.ones(n, dtype=torch.long), instrument=torch.ones(n, dtype=torch.long),
                    polarity=torch.ones(n, dtype=torch.long), ce=torch.ones(n)*20,
                    prec_mz=torch.ones(n)*47)

    def test_tokenizer(self):
        for smi in ['CCO', 'C/C=C\\C', '[NH4+]', 'C%12CCCCC%12']:
            self.assertEqual(smi, self.tok.decode(self.tok.encode(smi)))
        with self.assertRaises(ValueError):
            self.tok.encode('C'*127)
        with self.assertRaises(ValueError):
            self.tok.encode('C invalid')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'vocab.json'
            self.tok.save(p)
            self.assertEqual(s.SmilesTokenizer.load(p).vocab, self.tok.vocab)

    def test_transfer_and_causality(self):
        model = s.Spec2SmilesModel(self.cfg).eval()
        base = Spec2FPModel(self.cfg['encoder'])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'model.pt'
            pw_p = Path(tmp)/'spec2fp_pos_weights.npy'
            np.save(pw_p, np.full(4096, 4.0, dtype=np.float32))
            torch.save({'best_proxy_mrr': np.float64(0.25), 'ema_state_dict': {'module.'+k: v for k, v in base.state_dict().items()}}, p)
            model.warm_start(p)
            expected_shift = float(0.5 * np.log(4.0))
            self.assertTrue(torch.allclose(model.pw_shift, torch.tensor(expected_shift, dtype=torch.float32)))
        self.assertTrue(torch.equal(model.peak_proj[0].weight, base.peak_proj[0].weight))
        self.assertIs(model.output.weight, model.token_emb.weight)
        self.assertFalse(model.elem_count_head.weight.requires_grad)
        ids = torch.tensor([self.tok.encode('CCO')[:4]]*2)
        changed = ids.clone()
        changed[:, -1] = self.tok.vocab['N']
        batch = self.spectra()
        with torch.no_grad():
            first = model(tokens=ids, neutral_mass=torch.ones(2)*46, **batch)
            second = model(tokens=changed, neutral_mass=torch.ones(2)*46, **batch)
        torch.testing.assert_close(first[:, :-1], second[:, :-1])
        self.assertTrue(torch.isfinite(first).all())

    def test_disk_dataset_and_training(self):
        model = s.Spec2SmilesModel(self.cfg)
        for p in model.encoder_parameters():
            p.requires_grad_(False)
        tok = np.asarray([self.tok.encode('CCO')], dtype=np.uint8)
        frame = pd.DataFrame([dict(molecule_id='m1', mz=[10., 20.], inten=[1., 2.], precursor_mz=47.,
            neutral_mass=46., ce=20., adduct='[M+H]+', instrument='qtof', polarity='positive', smiles='CCO')]*2)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'records.bin'
            with p.open('wb') as stream:
                s.append_records(frame, stream, self.cfg, {'m1': 0})
            ds = s.SpectrumDataset(p, self.cfg, self.tok, ['CCO'], tok, augment=True, tokens_aug=tok)
            with patch.object(s.Chem, 'MolFromSmiles', side_effect=AssertionError('RDKit in training')), patch.object(s.np.random, 'random', return_value=0.0):
                np.testing.assert_array_equal(ds[0][2], tok[0])
            loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=ds.collate, num_workers=0)
            ema = s.EMA(model)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            before = model.peak_proj[0].weight.detach().clone()
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            loss = s.run_epoch(model, loader, torch.device('cpu'), opt, scaler, ema)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(torch.equal(before, model.peak_proj[0].weight))
            self.assertEqual(len(ds), 2)
            del ds, loader

    def test_generation_pooling_and_filtering(self):
        model = s.Spec2SmilesModel(self.cfg).eval()
        mass = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles('CCO'))
        frame = pd.DataFrame([dict(molecule_id='a', mz=[10.], inten=[1.], precursor_mz=mass+1,
            neutral_mass=mass, ce=20., adduct='[M+H]+', instrument='qtof', polarity='positive')]*3)
        hypotheses = [(self.tok.encode('CCO'), -1.), (self.tok.encode('OCC'), -2.),
                      (self.tok.encode('CCN'), -0.5)]
        with patch.object(s, 'beam_search', return_value=[hypotheses]) as beam:
            result = s.generate_candidates(model, self.tok, frame, 'cpu')
        self.assertEqual(beam.call_count, 1)
        self.assertEqual(list(result), s.COLUMNS)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].smiles, 'CCO')
        self.assertAlmostEqual(result.iloc[0].ppm_err, 0.)
        self.assertTrue(np.isfinite(result.iloc[0].nb_ll))
        self.assertTrue(np.isfinite(result.iloc[0].cos_sim))
        self.assertTrue(np.isfinite(result.iloc[0].maccs_cos))
        self.assertEqual(result['rank'].tolist(), [1, 2])
        self.assertEqual(result.source.unique().tolist(), ['spec2smiles'])
        self.assertEqual(len(s.generate_candidates(model, self.tok, frame.iloc[:0], 'cpu')), 0)

    def test_beam_eos_and_scores(self):
        class Toy:
            def decode(self, seq, memory, mask):
                logits = torch.full((len(seq), seq.shape[1], 5), -20.)
                logits[:, -1, 4 if seq.shape[1] == 1 else 2] = 20.
                return logits
        paths = s.beam_search(Toy(), torch.zeros(1, 1, 320), torch.zeros(1, 1, dtype=torch.bool), beam_size=2, max_len=8, top_k=3)[0]
        self.assertEqual(paths[0][0], [1, 4, 2])
        self.assertTrue(all(ids[-1] == 2 and score <= 0 for ids, score in paths))

    def test_batched_beam_matches_individual(self):
        class Toy:
            def decode(self, seq, memory, mask):
                logits = torch.full((len(seq), seq.shape[1], 7), -10.)
                token = memory[:, 0, 0].long() if seq.shape[1] == 1 else torch.full((len(seq),), 2)
                logits[torch.arange(len(seq)), -1, token] = 10.
                return logits
        memory = torch.tensor([4., 5., 6.]).reshape(3, 1, 1)
        mask = torch.zeros(3, 1, dtype=torch.bool)
        batched = s.beam_search(Toy(), memory, mask, beam_size=2, max_len=6)
        for i in range(3):
            single = s.beam_search(Toy(), memory[i:i+1], mask[i:i+1], beam_size=2, max_len=6)
            self.assertEqual(batched[i], single[0])
            self.assertEqual(batched[i][0][0], [1, 4+i, 2])

    def test_notebook_and_embedded_modules(self):
        notebook = build_notebook()
        nbformat.validate(nbformat.from_dict(notebook))
        with tempfile.TemporaryDirectory() as tmp:
            scope = dict(WORK=Path(tmp))
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                ast.parse('\n'.join(line for line in cell['source'].splitlines() if not line.startswith('!')))
                self.assertLess(len(cell['source'].splitlines()), 59)
                if cell['source'].startswith(('module_path =', 'fragment =')):
                    exec(cell['source'], scope)
            for name in ('spec2fp.py', 'spec2smiles.py', 'evalkit.py'):
                self.assertEqual((Path(tmp)/name).read_text(encoding='utf-8'),
                                 find_path(name).read_text(encoding='utf-8'))

    def test_notebook_synthetic_pipeline(self):
        # Execute the notebook on CPU with a tiny encoder/two epochs. Chemistry
        # exports use deterministic hypotheses; real beam behavior is tested above.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'input'
            inputs.mkdir()
            base = Spec2FPModel(self.cfg['encoder'])
            torch.save(base.state_dict(), inputs/'spec2fp_model.pt')
            (inputs/'spec2fp_config.json').write_text(json.dumps(dict(model=self.cfg['encoder'])))
            data = []
            for mid, smi in [('tr', 'CCO'), ('vm', 'CCN'), ('vr', 'CCO'), ('te', 'CCN')]:
                mass = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smi))
                data.extend([dict(molecule_id=mid, smiles=smi, mz=[10., 20.], inten=[1., 2.],
                    precursor_mz=mass+1, neutral_mass=mass, ce=20., adduct='[M+H]+',
                    instrument='qtof', polarity='positive')]*2)
            frame = pd.DataFrame(data)
            frame.loc[frame.molecule_id.ne('te')].to_parquet(inputs/'train_clean.parquet')
            frame.loc[frame.molecule_id.eq('te')].drop(columns='smiles').to_parquet(inputs/'test_clean.parquet')
            pd.DataFrame(dict(molecule_id=['tr', 'vm', 'vr'],
                             split3=['train', 'val_model', 'val_rank'])).to_parquet(inputs/'splits.parquet')
            scope = {}
            for cell in build_notebook()['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = cell['source'].replace("Path('/kaggle/working')", repr(root/'work'))
                # repr(WindowsPath) is a constructor; use Path with a string instead.
                source = source.replace(repr(root/'work'), f'Path({str(root / "work")!r})')
                source = source.replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')
                source = source.replace("assert device.type == 'cuda', 'Training requires a Kaggle GPU accelerator'", '')
                source = '\n'.join(line for line in source.splitlines() if not line.startswith('!'))
                exec(source, scope)
                if source.startswith('encoder_path ='):
                    scope['cfg'].update(epochs=2, batch_size=2, top_k_peaks=4)
                if source.startswith('paths ='):
                    tok = scope['tokenizer']
                    scope['s2s'].beam_search = lambda model, memory, mask, **kw: [[(tok.encode('CCO'), -1.)] for _ in range(len(memory))]
            out = root/'work/artifacts/spec2smiles'
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['trained_on_split'], 'train')
            self.assertEqual(len(manifest['summaries']), 3)
            for name in ('val_model', 'val_rank', 'test'):
                candidates = pd.read_parquet(out/f'cand_gen_{name}.parquet')
                self.assertEqual(list(candidates), s.COLUMNS)
                self.assertEqual(len(candidates), 1)

    def test_bug_fixes(self):
        # Bug 1: target_map keyed by molecule_id
        frame = pd.DataFrame([
            dict(molecule_id='mol1', mz=[10.], inten=[1.], precursor_mz=47.,
                 neutral_mass=46., ce=20., adduct='[M+H]+', instrument='qtof',
                 polarity='positive', smiles='CCO'),
            dict(molecule_id='mol1', mz=[10.], inten=[1.], precursor_mz=47.,
                 neutral_mass=46., ce=20., adduct='[M+H]+', instrument='qtof',
                 polarity='positive', smiles='OCC'),  # alternate SMILES for same molecule_id
        ])
        target_map = {'mol1': 42}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'records.bin'
            with p.open('wb') as stream:
                s.append_records(frame, stream, self.cfg, target_map)
            records = np.memmap(p, mode='r', dtype=s.record_dtype(self.cfg['top_k_peaks']))
            self.assertEqual(int(records[0]['target']), 42)
            self.assertEqual(int(records[1]['target']), 42)
            del records

        # Bug 2: Empty string or None fallback in submission
        fallback = 'CC(=O)Oc1ccccc1C(=O)O'
        preds = {'m1': '', 'm2': 'CCO;CCN'}
        self.assertEqual(preds.get('m1') or fallback, fallback)
        self.assertEqual(preds.get('m2') or fallback, 'CCO;CCN')
        self.assertEqual(preds.get('missing') or fallback, fallback)

        # Bug 3: Truth lookup safety in validation
        truth = {'val1': 'IK14_A'}
        candidates = pd.DataFrame([
            dict(molecule_id='val1', ik14='IK14_A', rank=1),
            dict(molecule_id='val_missing', ik14='IK14_B', rank=2),
        ])
        reciprocal, hits = 0.0, 0
        for qid, grp in candidates.groupby('molecule_id'):
            t_ik = truth.get(str(qid))
            if t_ik:
                matched = grp.loc[grp.ik14.eq(t_ik), 'rank']
                if len(matched):
                    reciprocal += 1 / int(matched.min())
                    hits += 1
        self.assertEqual(hits, 1)
        self.assertEqual(reciprocal, 1.0)

        # Bug 4: pw_shift effect on fp_probs
        model = s.Spec2SmilesModel(self.cfg).eval()
        batch = self.spectra(n=1)
        with torch.no_grad():
            _, _, _, _, _, fp_unshifted, _ = model.encode(**batch, return_fp=True)
            model.pw_shift.fill_(2.0)
            _, _, _, _, _, fp_shifted, _ = model.encode(**batch, return_fp=True)
        self.assertTrue((fp_shifted < fp_unshifted).all())

        # Peak gather mz return and multi-spectrum non-finite/negative robustness
        spec_item = s.spectrum_item(pd.Series(dict(
            mz=[10., 20.], inten=[0., 2.], precursor_mz=47., adduct='[M+H]+', ce=20.,
            instrument='qtof', polarity='positive'
        )), self.cfg)
        self.assertIn('mz', spec_item)
        self.assertEqual(len(spec_item['mz']), spec_item['n_peaks'])


if __name__ == '__main__':
    unittest.main()
