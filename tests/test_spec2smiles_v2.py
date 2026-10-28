"""Unit and synthetic integration tests for Stage 07 v2 notebook and optimizations."""
import ast
import json
import tempfile
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

import nbformat
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import spec2smiles as s
from spec2fp import Spec2FPModel
from build_07_spec2smiles_train_v2 import build_notebook

torch.set_num_threads(1)


class Stage07V2Tests(unittest.TestCase):
    def setUp(self):
        self.tok = s.SmilesTokenizer().fit(['CCO', 'CCN', 'C/C=C\\C', '[NH4+]', 'C%12CCCCC%12'])
        self.cfg = dict(encoder=dict(d_model=32, n_layers=1, n_heads=4, feat_dim=129),
                        vocab_size=len(self.tok.vocab), top_k_peaks=4, mz_dim=64)

    def spectra(self, n=2):
        return dict(peaks=torch.randn(n, 4, 129), pad_mask=torch.zeros(n, 4, dtype=torch.bool),
                    adduct=torch.ones(n, dtype=torch.long), instrument=torch.ones(n, dtype=torch.long),
                    polarity=torch.ones(n, dtype=torch.long), ce=torch.ones(n)*20,
                    prec_mz=torch.ones(n)*47)

    def test_notebook_format_and_cells(self):
        nb_path = NOTEBOOKS / '07_spec2smiles_train_v2.ipynb'
        self.assertTrue(nb_path.is_file())
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        self.assertEqual(len(nb.cells), 47)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                lines = cell.source.splitlines()
                self.assertLess(len(lines), 59, f'Cell {i} has {len(lines)} lines')
                ast.parse('\n'.join(line for line in lines if not line.startswith('!')))

    def test_embedded_modules_roundtrip(self):
        notebook = build_notebook()
        with tempfile.TemporaryDirectory() as tmp:
            scope = dict(WORK=Path(tmp))
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                if cell['source'].startswith(('module_path =', 'fragment =')):
                    exec(cell['source'], scope)
            for name in ('spec2fp.py', 'spec2smiles.py', 'evalkit.py'):
                src_file = SRC / name
                self.assertEqual((Path(tmp)/name).read_text(encoding='utf-8'),
                                 src_file.read_text(encoding='utf-8'))

    def test_early_stopping_beam_search(self):
        class ToyModel:
            def decode(self, seq, memory, mask):
                # Always emit token 2 (EOS) after step 1
                logits = torch.full((len(seq), seq.shape[1], 10), -50.)
                if seq.shape[1] == 1:
                    logits[:, -1, 4] = 10.
                else:
                    logits[:, -1, 2] = 20.
                return logits

        memory = torch.zeros(2, 1, 320)
        mask = torch.zeros(2, 1, dtype=torch.bool)
        paths = s.beam_search(ToyModel(), memory, mask, beam_size=4, max_len=96, top_k=25, archive_size=64)
        self.assertEqual(len(paths), 2)
        # Verify completed paths terminated early and have EOS
        for p_list in paths:
            self.assertTrue(len(p_list) > 0)
            for ids, score in p_list:
                self.assertEqual(ids[-1], 2)
                self.assertTrue(score <= 0)

    def test_candidate_generation_two_stage(self):
        model = s.Spec2SmilesModel(self.cfg).eval()
        mass = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles('CCO'))
        frame = pd.DataFrame([dict(molecule_id='m1', mz=[10.], inten=[1.], precursor_mz=mass+1,
                                   neutral_mass=mass, ce=20., adduct='[M+H]+', instrument='qtof',
                                   polarity='positive')]*2)
        hypotheses = [(self.tok.encode('CCO'), -0.2), (self.tok.encode('OCC'), -0.5),
                      (self.tok.encode('CCN'), -0.8)]
        from unittest.mock import patch
        with patch.object(s, 'beam_search', return_value=[hypotheses]):
            cands = s.generate_candidates(model, self.tok, frame, 'cpu', beam_size=20, top_k=25,
                                          mol_batch_size=64, max_len=96)
        self.assertEqual(list(cands.columns), s.COLUMNS)
        # CCO and OCC deduplicate to same fast_key, CCN is second
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands.iloc[0].smiles, 'CCO')
        self.assertEqual(cands['rank'].tolist(), [1, 2])

    def test_notebook_synthetic_pipeline_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            inputs.mkdir()
            base = Spec2FPModel(self.cfg['encoder'])
            torch.save(base.state_dict(), inputs / 'spec2fp_model.pt')
            (inputs / 'spec2fp_config.json').write_text(json.dumps(dict(model=self.cfg['encoder'])))

            # Mock Stage 07 attached checkpoint and vocab
            dummy_s2s = s.Spec2SmilesModel(dict(encoder=self.cfg['encoder'], vocab_size=len(self.tok.vocab)))
            torch.save({'model_state_dict': dummy_s2s.state_dict(), 'epoch': 8, 'val_loss': 0.75130},
                       inputs / 'spec2smiles_model.pt')
            self.tok.save(inputs / 'smiles_vocab.json')

            data = []
            for mid, smi in [('tr', 'CCO'), ('vm', 'CCN'), ('vr', 'CCO'), ('te', 'CCN')]:
                mass = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smi))
                data.extend([dict(molecule_id=mid, smiles=smi, mz=[10., 20.], inten=[1., 2.],
                             precursor_mz=mass+1, neutral_mass=mass, ce=20., adduct='[M+H]+',
                             instrument='qtof', polarity='positive')]*3)
            frame = pd.DataFrame(data)
            frame.loc[frame.molecule_id.ne('te')].to_parquet(inputs / 'train_clean.parquet')
            frame.loc[frame.molecule_id.eq('te')].drop(columns='smiles').to_parquet(inputs / 'test_clean.parquet')
            pd.DataFrame(dict(molecule_id=['tr', 'vm', 'vr'],
                             split3=['train', 'val_model', 'val_rank'])).to_parquet(inputs / 'splits.parquet')

            scope = {}
            for cell in build_notebook()['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = cell['source'].replace("Path('/kaggle/working')", repr(root / 'work'))
                source = source.replace(repr(root / 'work'), f'Path({str(root / "work")!r})')
                source = source.replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')
                source = source.replace("assert device.type == 'cuda', 'Training requires a Kaggle GPU accelerator'", '')
                source = '\n'.join(line for line in source.splitlines() if not line.startswith('!'))
                exec(source, scope)
                if source.startswith('encoder_path ='):
                    scope['cfg'].update(epochs=1, batch_size=2, top_k_peaks=4)
                if source.startswith('paths ='):
                    tok = scope['tokenizer']
                    scope['s2s'].beam_search = lambda model, memory, mask, **kw: [[(tok.encode('CCO'), -1.)] for _ in range(len(memory))]

            out = root / 'work/artifacts/spec2smiles'
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['trained_on_split'], 'train')
            self.assertEqual(len(manifest['summaries']), 3)
            for name in ('val_model', 'val_rank', 'test'):
                candidates = pd.read_parquet(out / f'cand_gen_{name}.parquet')
                self.assertEqual(list(candidates), s.COLUMNS)
                self.assertEqual(len(candidates), 1)

            # Check submission
            sub = pd.read_csv(out / 'submission_spec2smiles.csv')
            self.assertEqual(list(sub.columns), ['molecule_id', 'smiles'])
            self.assertEqual(sub['molecule_id'].tolist(), ['te'])
            self.assertEqual(sub['smiles'].iloc[0], 'CCO')


if __name__ == '__main__':
    unittest.main()
