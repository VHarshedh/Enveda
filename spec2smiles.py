"""Stage 07: transferable spectral encoder and autoregressive SMILES decoder."""
import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import evalkit
from spec2fp import (Spec2FPModel, ELEMENTS_ORDER, tensorize_spectrum,
                     collate_spectra, parse_peak_array)

ENCODER_NAMES = ('peak_proj', 'cls_token', 'adduct_emb', 'inst_emb',
                 'pol_emb', 'ce_proj', 'prec_proj', 'encoder', 'norm')
COLUMNS = ['molecule_id', 'smiles', 'ik14', 'fast_key', 'score',
           'log_prob', 'ppm_err', 'rank', 'source']
# A raw regex needs two backslashes to match one SMILES bond backslash.
SMILES_PATTERN = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'


class SmilesTokenizer:
    def __init__(self, vocab=None, max_length=128):
        self.vocab = vocab or dict(zip(('<pad>', '<bos>', '<eos>', '<unk>'), range(4)))
        self.max_length = max_length
        self.pattern = re.compile(SMILES_PATTERN)
        self.inverse = {v: k for k, v in self.vocab.items()}

    def tokens(self, smiles):
        tokens = self.pattern.findall(smiles)
        if ''.join(tokens) != smiles:
            raise ValueError(f'Unrecognized SMILES syntax: {smiles}')
        return tokens

    def fit(self, smiles):
        found = set()
        for smi in smiles:
            found.update(self.tokens(smi))
        # Randomization can introduce ring labels absent from canonical strings.
        found.update(list('0123456789') + [f'%{i:02d}' for i in range(10, 30)])
        self.vocab.update({s: i + 4 for i, s in enumerate(sorted(found))})
        self.inverse = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, smiles):
        tokens = self.tokens(smiles)
        if len(tokens) + 2 > self.max_length:
            raise ValueError('SMILES exceeds sequence budget; never truncate molecules')
        ids = [1] + [self.vocab.get(t, 3) for t in tokens] + [2]
        return ids + [0] * (self.max_length - len(ids))

    def decode(self, ids):
        out = []
        for idx in ids:
            if int(idx) == 2:
                break
            if int(idx) not in (0, 1):
                out.append(self.inverse[int(idx)])
        return ''.join(out)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(vocab=self.vocab,
                                             max_length=self.max_length), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class Spec2SmilesModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base = Spec2FPModel(cfg['encoder'])
        for name in ENCODER_NAMES:
            setattr(self, name, getattr(base, name))
        self.elem_count_head = base.elem_count_head
        self.elem_count_head.requires_grad_(False)
        enc_dim = cfg['encoder']['d_model']
        self.memory_proj = nn.Linear(enc_dim, 320) if enc_dim != 320 else nn.Identity()
        self.condition = nn.Sequential(nn.Linear(11, 320), nn.GELU(), nn.LayerNorm(320))
        self.token_emb = nn.Embedding(cfg['vocab_size'], 320, padding_idx=0)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[0].zero_()
        layer = nn.TransformerDecoderLayer(320, 8, 1280, 0.1,
                                          batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 4, norm=nn.LayerNorm(320))
        self.output = nn.Linear(320, cfg['vocab_size'], bias=False)
        self.output.weight = self.token_emb.weight
        pos = torch.arange(128).float().unsqueeze(1)
        angles = pos * torch.exp(torch.arange(0, 320, 2).float() * (-math.log(10000)/320))
        pe = torch.zeros(128, 320)
        pe[:, 0::2], pe[:, 1::2] = angles.sin(), angles.cos()
        self.register_buffer('positions', pe)

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters() if n.split('.')[0] in ENCODER_NAMES]

    def warm_start(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('ema_state_dict', state.get('model_state_dict', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        wanted = {k: v for k, v in self.state_dict().items()
                  if k.split('.')[0] in ENCODER_NAMES + ('elem_count_head',)}
        missing = set(wanted) - set(state)
        if missing:
            raise ValueError(f'Incomplete pretrained encoder: {sorted(missing)}')
        self.load_state_dict({k: state[k] for k in wanted}, strict=False)

    def encode(self, peaks, pad_mask, adduct, instrument, polarity, ce, prec_mz):
        b = len(peaks)
        meta1 = self.adduct_emb(adduct) + self.ce_proj(ce[:, None]/100)
        meta1 = meta1 + self.prec_proj(prec_mz[:, None]/1000)
        meta2 = self.inst_emb(instrument) + self.pol_emb(polarity)
        x = torch.cat((self.cls_token.expand(b, -1, -1),
                       torch.stack((meta1, meta2), 1), self.peak_proj(peaks)), 1)
        mask = torch.cat((pad_mask.new_zeros((b, 3)), pad_mask), 1)
        x = self.encoder(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        pooled = self.norm(x[:, 0] + (x*valid).sum(1)/valid.sum(1).clamp_min(1))
        counts = self.elem_count_head(pooled).detach().clamp(0, 200)
        return self.memory_proj(self.norm(x)), mask, counts

    def conditioned_memory(self, memory, mask, neutral_mass, counts):
        cond = self.condition(torch.cat((neutral_mass[:, None]/1000, counts), -1))
        return torch.cat((cond[:, None], memory), 1), torch.cat((mask[:, :1]*False, mask), 1)

    def decode(self, tokens, memory, mask):
        x = self.token_emb(tokens)*math.sqrt(320) + self.positions[:tokens.shape[1]]
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=x.device,
                            dtype=torch.bool).triu(1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens.eq(0),
                         memory_key_padding_mask=mask)
        return self.output(x)

    def forward(self, tokens, neutral_mass, **spectra):
        memory, mask, counts = self.encode(**spectra)
        memory, mask = self.conditioned_memory(memory, mask, neutral_mass, counts)
        return self.decode(tokens, memory, mask)


def spectrum_item(row, cfg):
    return tensorize_spectrum(row.mz, row.inten, float(row.precursor_mz),
                             row.adduct, float(row.ce), row.instrument, row.polarity,
                             top_k=cfg['top_k_peaks'], mz_dim=cfg['mz_dim'])


def record_dtype(top_k):
    return np.dtype([('mz', 'f4', (top_k,)), ('inten', 'f4', (top_k,)),
                     ('n', 'i2'), ('prec', 'f4'), ('mass', 'f4'), ('ce', 'f4'),
                     ('adduct', 'i2'), ('inst', 'i2'), ('pol', 'i2'), ('target', 'i4')])


class SpectrumDataset(torch.utils.data.Dataset):
    """Raw top peaks on disk; only a single batch gets sinusoidal features."""
    def __init__(self, path, cfg, tokenizer, smiles, tokens, augment=False, tokens_aug=None):
        self.cfg, self.tokenizer = cfg, tokenizer
        self.smiles, self.tokens, self.augment = smiles, tokens, augment
        self.tokens_aug = tokens if tokens_aug is None else tokens_aug
        if self.tokens_aug.shape != tokens.shape:
            raise ValueError('Augmented targets must match canonical target shape')
        self.rows = np.memmap(path, mode='r', dtype=record_dtype(cfg['top_k_peaks']))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        target = self.tokens[r['target']].astype(np.int64)
        if self.augment and np.random.random() < 0.5:
            target = self.tokens_aug[r['target']].astype(np.int64)
        item = tensorize_spectrum(r['mz'][:r['n']], r['inten'][:r['n']], r['prec'],
                                  int(r['adduct']), r['ce'], int(r['inst']), int(r['pol']),
                                  self.cfg['top_k_peaks'], self.cfg['mz_dim'])
        return item, float(r['mass']), target

    def collate(self, rows):
        items, masses, tokens = zip(*rows)
        batch = collate_spectra(items, self.cfg['top_k_peaks'], 2*self.cfg['mz_dim']+1)
        batch['neutral_mass'] = torch.tensor(masses, dtype=torch.float32)
        return batch, torch.from_numpy(np.stack(tokens))


def append_records(frame, stream, cfg, target_map):
    from spec2fp import ADDUCT2IDX, INST2IDX, POLARITY2IDX
    records = np.zeros(len(frame), dtype=record_dtype(cfg['top_k_peaks']))
    for i, row in enumerate(frame.itertuples(index=False)):
        if not (math.isfinite(row.neutral_mass) and row.neutral_mass > 0
                and math.isfinite(row.precursor_mz) and row.precursor_mz > 0):
            raise ValueError('Training spectra require finite positive precursor/neutral masses')
        mz, inten = parse_peak_array(row.mz), parse_peak_array(row.inten)
        valid = np.isfinite(mz) & np.isfinite(inten) & (inten >= 0)
        valid &= mz <= float(row.precursor_mz) + 5
        mz, inten = mz[valid], inten[valid]
        idx = np.argsort(inten)[-cfg['top_k_peaks']:]
        idx = idx[np.argsort(mz[idx])]
        n = len(idx)
        records[i]['mz'][:n], records[i]['inten'][:n] = mz[idx], inten[idx]
        records[i]['n'], records[i]['prec'] = n, row.precursor_mz
        records[i]['mass'], records[i]['ce'] = row.neutral_mass, row.ce
        records[i]['adduct'] = ADDUCT2IDX.get(str(row.adduct), ADDUCT2IDX['<unk>'])
        records[i]['inst'] = INST2IDX.get(str(row.instrument), INST2IDX['<unk>'])
        records[i]['pol'] = POLARITY2IDX.get(str(row.polarity).lower(), POLARITY2IDX['<unk>'])
        records[i]['target'] = target_map[row.smiles]
    records.tofile(stream)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for dst, src in zip(self.model.parameters(), model.parameters()):
            dst.lerp_(src, 1-self.decay)


def run_epoch(model, loader, device, optimizer=None, scaler=None, ema=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    raw = model.module if isinstance(model, nn.DataParallel) else model
    if training and not raw.cls_token.requires_grad:
        for name in ENCODER_NAMES:
            module = getattr(raw, name)
            if isinstance(module, nn.Module):
                module.eval()
    total, count = 0.0, 0
    for batch, target in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = target.to(device)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(tokens=target[:, :-1], **batch)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    target[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SMILES loss; check spectra and checkpoint')
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    ema.update(raw)
                    if scheduler is not None:
                        scheduler.step()
        n = int(target[:, 1:].ne(0).sum())
        total, count = total + float(loss.detach())*n, count+n
    if not count:
        raise ValueError('Empty training/validation loader')
    return total/count


@torch.no_grad()
def beam_search(model, memory, mask, beam_size=12, max_len=96, top_k=25):
    """Decode B molecules together; return one list of completed paths per molecule."""
    if beam_size < 1 or max_len < 2 or max_len > 128 or top_k < 1:
        raise ValueError('Invalid beam size, sequence budget, or candidate count')
    b, length, dim = memory.shape
    k, device = beam_size, memory.device
    memory = memory[:, None].expand(-1, k, -1, -1).reshape(b*k, length, dim)
    mask = mask[:, None].expand(-1, k, -1).reshape(b*k, length)
    seq = torch.ones((b, k, 1), dtype=torch.long, device=device)
    scores = torch.full((b, k), -torch.inf, device=device)
    scores[:, 0] = 0
    archive_size = max(top_k*8, k)
    archive_scores = torch.full((b, archive_size), -torch.inf, device=device)
    archive = torch.zeros((b, archive_size, max_len), dtype=torch.long, device=device)
    for step in range(1, max_len):
        logits = model.decode(seq.reshape(b*k, -1), memory, mask)
        logp = logits[:, -1].float().log_softmax(-1).reshape(b, k, -1)
        logp[:, :, [0, 1, 3]] = -torch.inf
        if step == 1:
            logp[:, :, 2] = -torch.inf
        combined = logp + scores[:, :, None]
        vocab = combined.shape[-1]
        values, indices = combined.flatten(1).topk(min(2*k, k*vocab), dim=1)
        parents, tokens = indices // vocab, indices % vocab
        paths = torch.cat((seq.gather(1, parents[:, :, None].expand(-1, -1, step)),
                           tokens[:, :, None]), dim=2)
        eos_scores = values.masked_fill(tokens.ne(2), -torch.inf)
        padded = nn.functional.pad(paths, (0, max_len-step-1))
        archive_scores, keep = torch.cat((archive_scores, eos_scores), 1).topk(archive_size, dim=1)
        archive = torch.cat((archive, padded), 1).gather(1, keep[:, :, None].expand(-1, -1, max_len))
        scores, live = values.masked_fill(tokens.eq(2), -torch.inf).topk(k, dim=1)
        seq = paths.gather(1, live[:, :, None].expand(-1, -1, step+1))
        done = scores.max(1).values <= archive_scores[:, -1]
        scores = scores.masked_fill(done[:, None], -torch.inf)
        if not torch.isfinite(scores).any():
            break
    result = []
    for paths, values in zip(archive.cpu().tolist(), archive_scores.cpu().tolist()):
        result.append([(ids[:ids.index(2)+1], score) for ids, score in zip(paths, values)
                       if math.isfinite(score)])
    return result


@torch.no_grad()
def generate_candidates(model, tokenizer, query_df, device, beam_size=12, top_k=25, mol_batch_size=32, max_len=96):
    model = model.module if isinstance(model, nn.DataParallel) else model
    model.eval()
    device = torch.device(device)
    rows = []
    if beam_size < 1 or top_k < 1 or mol_batch_size < 1:
        raise ValueError('beam_size and top_k must be positive')
    groups = iter(query_df.groupby('molecule_id', sort=False))
    from itertools import islice
    while True:
        chunk = list(islice(groups, mol_batch_size))
        if not chunk:
            break
        memories, masks, masses = [], [], []
        for molecule_id, group in chunk:
            mass = float(group.neutral_mass.mean())
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f'Invalid neutral mass for {molecule_id}')
            mem_sum, weights, elem_sum, n = None, None, None, 0
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                for start in range(0, len(group), 32):
                    items = [spectrum_item(r, model.cfg) for r in group.iloc[start:start+32].itertuples()]
                    batch = collate_spectra(items, model.cfg['top_k_peaks'], 2*model.cfg['mz_dim']+1)
                    memory, mask, elem = model.encode(**{k: v.to(device) for k, v in batch.items()})
                    valid = (~mask).float().unsqueeze(-1)
                    ms, ws, es = (memory.float()*valid).sum(0, keepdim=True), valid.sum(0, keepdim=True), elem.float().sum(0, keepdim=True)
                    mem_sum = ms if mem_sum is None else mem_sum + ms
                    weights = ws if weights is None else weights + ws
                    elem_sum = es if elem_sum is None else elem_sum + es
                    n += len(items)
                memory, mask = model.conditioned_memory(mem_sum/weights.clamp_min(1),
                    weights.squeeze(-1).eq(0), torch.tensor([mass], device=device), elem_sum/n)
            memories.append(memory)
            masks.append(mask)
            masses.append(mass)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            hypotheses = beam_search(model, torch.cat(memories), torch.cat(masks),
                                     beam_size=beam_size, max_len=max_len, top_k=top_k)
        for (molecule_id, _), mass, paths in zip(chunk, masses, hypotheses):
            unique = {}
            for ids, log_prob in paths:
                smi = tokenizer.decode(ids)
                mol = Chem.MolFromSmiles(smi)
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                if any(a.GetSymbol() not in ELEMENTS_ORDER for a in mol.GetAtoms()):
                    continue
                key = evalkit.fast_key(mol)
                if not key:
                    continue
                ppm = abs(rdMolDescriptors.CalcExactMolWt(mol)-mass)/mass*1e6
                score = log_prob - 0.05*ppm
                candidate = dict(molecule_id=str(molecule_id), smiles=Chem.MolToSmiles(mol, isomericSmiles=False),
                                 fast_key=key, score=score, log_prob=log_prob, ppm_err=ppm, source='spec2smiles')
                if key not in unique or score > unique[key]['score']:
                    unique[key] = candidate
            for rank, candidate in enumerate(sorted(unique.values(), key=lambda x: -x['score'])[:top_k], 1):
                candidate.update(rank=rank, ik14=evalkit.canon_ik14(candidate['smiles']))
                rows.append(candidate)
    return pd.DataFrame(rows, columns=COLUMNS).astype({'rank': 'int64', 'score': 'float64',
                                                       'log_prob': 'float64', 'ppm_err': 'float64'})
