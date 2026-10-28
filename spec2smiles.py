"""Stage 07: transferable spectral encoder, 180-dim substructure conditioning, and Self-Spec2FP reranked SMILES decoder."""
import copy
import json
import math
import re
from itertools import islice
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
AUX_HEAD_NAMES = ('fp_head', 'maccs_head', 'heavy_atom_head',
                  'ring_count_head', 'elem_presence_head', 'elem_count_head')
COLUMNS = ['molecule_id', 'smiles', 'ik14', 'fast_key', 'score',
           'log_prob', 'nb_ll', 'cos_sim', 'tan_sim', 'maccs_cos', 'maccs_tan',
           'elem_dist', 'ppm_err', 'rank', 'source']

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


def mol_elem_counts(mol):
    counts = np.zeros(len(ELEMENTS_ORDER), dtype=np.float32)
    mol_h = Chem.AddHs(mol)
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ELEMENTS_ORDER:
            counts[ELEMENTS_ORDER.index(sym)] += 1.0
    return counts


class Spec2SmilesModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base = Spec2FPModel(cfg['encoder'])
        for name in ENCODER_NAMES:
            setattr(self, name, getattr(base, name))
        # Retain all pretrained Stage 05b auxiliary heads (frozen) for conditioning & reranking
        for name in AUX_HEAD_NAMES:
            head = getattr(base, name)
            head.requires_grad_(False)
            setattr(self, name, head)

        enc_dim = cfg['encoder']['d_model']
        self.memory_proj = nn.Linear(enc_dim, 320) if enc_dim != 320 else nn.Identity()
        # 180-dim conditioning: neutral_mass(1) + elem_counts(10) + maccs_probs(167) + heavy(1) + rings(1)
        self.condition = nn.Sequential(
            nn.Linear(180, 320), nn.GELU(), nn.LayerNorm(320),
            nn.Linear(320, 320), nn.GELU(), nn.LayerNorm(320)
        )
        self.pooled_proj = nn.Sequential(nn.Linear(enc_dim, 320), nn.LayerNorm(320))
        self.token_emb = nn.Embedding(cfg['vocab_size'], 320, padding_idx=0)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[0].zero_()
        layer = nn.TransformerDecoderLayer(320, 8, 1280, 0.1,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.get('decoder_layers', 4), norm=nn.LayerNorm(320))
        self.output = nn.Linear(320, cfg['vocab_size'], bias=False)
        self.output.weight = self.token_emb.weight
        pos = torch.arange(128).float().unsqueeze(1)
        angles = pos * torch.exp(torch.arange(0, 320, 2).float() * (-math.log(10000)/320))
        pe = torch.zeros(128, 320)
        pe[:, 0::2], pe[:, 1::2] = angles.sin(), angles.cos()
        self.register_buffer('positions', pe)
        self.register_buffer('pw_shift', torch.zeros(4096))

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters() if n.split('.')[0] in ENCODER_NAMES]

    def warm_start(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('ema_state_dict', state.get('model_state_dict', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        wanted = {k: v for k, v in self.state_dict().items()
                  if k.split('.')[0] in ENCODER_NAMES + AUX_HEAD_NAMES}
        missing = set(wanted) - set(state)
        if missing:
            raise ValueError(f'Incomplete pretrained encoder: {sorted(missing)}')
        self.load_state_dict({k: state[k] for k in wanted}, strict=False)
        pw_path = Path(path).parent / 'spec2fp_pos_weights.npy'
        if pw_path.is_file():
            pw = torch.from_numpy(np.load(pw_path)).float().clamp(min=1.0)
            self.pw_shift.copy_(0.5 * torch.log(pw))

    def encode(self, peaks, pad_mask, adduct, instrument, polarity, ce, prec_mz, return_fp=False):
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
        with torch.no_grad():
            counts = self.elem_count_head(pooled).detach().clamp(0, 200)
            maccs_probs = torch.sigmoid(self.maccs_head(pooled)).detach()
            heavy = self.heavy_atom_head(pooled).detach().clamp(1, 150) / 100.0
            rings = self.ring_count_head(pooled).detach().clamp(0, 20) / 10.0
            cond_vec = torch.cat((counts / 50.0, maccs_probs, heavy, rings), dim=-1)  # 179 dims
            if return_fp:
                fp_probs = torch.sigmoid(self.fp_head(pooled) - self.pw_shift).detach()
                return self.memory_proj(self.norm(x)), mask, pooled, cond_vec, counts, fp_probs, maccs_probs
        return self.memory_proj(self.norm(x)), mask, pooled, cond_vec

    def conditioned_memory(self, memory, mask, neutral_mass, pooled, cond_vec):
        full_cond = torch.cat((neutral_mass[:, None] / 1000.0, cond_vec), dim=-1)  # 180 dims
        cond_tok = self.condition(full_cond) + self.pooled_proj(pooled)
        return torch.cat((cond_tok[:, None], memory), 1), torch.cat((mask[:, :1]*False, mask), 1)

    def decode(self, tokens, memory, mask):
        x = self.token_emb(tokens)*math.sqrt(320) + self.positions[:tokens.shape[1]]
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=x.device,
                            dtype=torch.bool).triu(1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens.eq(0),
                         memory_key_padding_mask=mask)
        return self.output(x)

    def forward(self, tokens, neutral_mass, **spectra):
        memory, mask, pooled, cond_vec = self.encode(**spectra)
        memory, mask = self.conditioned_memory(memory, mask, neutral_mass, pooled, cond_vec)
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
    """Raw top peaks on disk; supports K-variant precomputed SMILES augmentation."""
    def __init__(self, path, cfg, tokenizer, smiles, tokens, augment=False, tokens_aug=None):
        self.cfg, self.tokenizer = cfg, tokenizer
        self.smiles, self.tokens, self.augment = smiles, tokens, augment
        self.tokens_aug = tokens[None, ...] if tokens_aug is None else tokens_aug
        if self.tokens_aug.ndim == 2:
            self.tokens_aug = self.tokens_aug[None, ...]
        self.rows = np.memmap(path, mode='r', dtype=record_dtype(cfg['top_k_peaks']))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        t_idx = int(r['target'])
        if self.augment and np.random.random() < 0.65:
            k = np.random.randint(0, self.tokens_aug.shape[0])
            target = self.tokens_aug[k, t_idx].astype(np.int64)
        else:
            target = self.tokens[t_idx].astype(np.int64)
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
        records[i]['adduct'] = ADDUCT2IDX.get(str(row.adduct).strip(), ADDUCT2IDX['<unk>'])
        records[i]['inst'] = INST2IDX.get(str(row.instrument).strip(), INST2IDX['<unk>'])
        records[i]['pol'] = POLARITY2IDX.get(str(row.polarity).strip().lower(), POLARITY2IDX['<unk>'])
        records[i]['target'] = target_map[str(row.molecule_id)]
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
    for name in AUX_HEAD_NAMES:
        getattr(raw, name).eval()
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
def beam_search(model, memory, mask, beam_size=16, max_len=124, top_k=25):
    """Decode B molecules together; return length-penalized completed paths."""
    if beam_size < 1 or max_len < 2 or max_len > 128 or top_k < 1:
        raise ValueError('Invalid beam size, sequence budget, or candidate count')
    b, length, dim = memory.shape
    k, device = beam_size, memory.device
    memory = memory[:, None].expand(-1, k, -1, -1).reshape(b*k, length, dim)
    mask = mask[:, None].expand(-1, k, -1).reshape(b*k, length)
    seq = torch.ones((b, k, 1), dtype=torch.long, device=device)
    scores = torch.full((b, k), -torch.inf, device=device)
    scores[:, 0] = 0
    archive_size = max(top_k * 10, 256)
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
        norm_eos_scores = eos_scores / ((step + 1) ** 0.7)
        padded = nn.functional.pad(paths, (0, max_len-step-1))
        archive_scores, keep = torch.cat((archive_scores, norm_eos_scores), 1).topk(archive_size, dim=1)
        archive = torch.cat((archive, padded), 1).gather(1, keep[:, :, None].expand(-1, -1, max_len))
        scores, live = values.masked_fill(tokens.eq(2), -torch.inf).topk(k, dim=1)
        seq = paths.gather(1, live[:, :, None].expand(-1, -1, step+1))
        live_upper = scores.max(1).values / (max_len ** 0.7)
        done = live_upper <= archive_scores[:, -1]
        scores = scores.masked_fill(done[:, None], -torch.inf)
        if not torch.isfinite(scores).any():
            break
    result = []
    for paths, values in zip(archive.cpu().tolist(), archive_scores.cpu().tolist()):
        result.append([(ids[:ids.index(2)+1], score) for ids, score in zip(paths, values)
                       if math.isfinite(score) and 2 in ids])
    return result


@torch.no_grad()
def generate_candidates(model, tokenizer, query_df, device, beam_size=16, top_k=25, mol_batch_size=64, max_len=124):
    model = model.module if isinstance(model, nn.DataParallel) else model
    model.eval()
    device = torch.device(device)
    rows = []
    if beam_size < 1 or top_k < 1 or mol_batch_size < 1:
        raise ValueError('beam_size and top_k must be positive')
    top_k_peaks = model.cfg['top_k_peaks']
    groups = iter(query_df.groupby('molecule_id', sort=False))
    while True:
        chunk = list(islice(groups, mol_batch_size))
        if not chunk:
            break
        memories, masks, masses, pred_fps, pred_maccs, pred_elems = [], [], [], [], [], []
        for molecule_id, group in chunk:
            mass = float(group.neutral_mass.median())
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f'Invalid neutral mass for {molecule_id}')
            meta_mem_sum, pooled_sum, cond_sum, elem_sum, fp_sum, maccs_sum, n = None, None, None, None, None, None, 0
            peak_tokens_list, peak_intens_list, peak_mzs_list = [], [], []
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                for start in range(0, len(group), 32):
                    sub = group.iloc[start:start+32]
                    items = [spectrum_item(r, model.cfg) for r in sub.itertuples()]
                    batch = collate_spectra(items, top_k_peaks, 2*model.cfg['mz_dim']+1)
                    memory, mask, pooled, cond_vec, counts, fp_p, maccs_p = model.encode(
                        **{k: v.to(device) for k, v in batch.items()}, return_fp=True
                    )
                    # Global tokens 0, 1, 2 (CLS, meta1, meta2) are positionally aligned across spectra
                    meta_mem = memory[:, :3].float().sum(0, keepdim=True)
                    meta_mem_sum = meta_mem if meta_mem_sum is None else meta_mem_sum + meta_mem
                    pooled_sum = pooled.float().sum(0, keepdim=True) if pooled_sum is None else pooled_sum + pooled.float().sum(0, keepdim=True)
                    cond_sum = cond_vec.float().sum(0, keepdim=True) if cond_sum is None else cond_sum + cond_vec.float().sum(0, keepdim=True)
                    elem_sum = counts.float().sum(0, keepdim=True) if elem_sum is None else elem_sum + counts.float().sum(0, keepdim=True)
                    fp_sum = fp_p.float().sum(0) if fp_sum is None else fp_sum + fp_p.float().sum(0)
                    maccs_sum = maccs_p.float().sum(0) if maccs_sum is None else maccs_sum + maccs_p.float().sum(0)
                    n += len(items)
                    for it in items:
                        peak_mzs_list.append(torch.from_numpy(it['mz']).float().to(device))
                    # Gather valid fragment peak tokens without averaging misaligned m/z indices
                    valid_peaks = ~mask[:, 3:]
                    peak_tokens_list.append(memory[:, 3:][valid_peaks].float())
                    peak_intens_list.append(batch['peaks'][..., 64].to(device)[valid_peaks].float())

                # Select top_k_peaks highest-intensity fragment embeddings across the molecule's spectra,
                # deduplicating peaks within 0.01 Da bins across replicate spectra
                all_peak_toks = torch.cat(peak_tokens_list, dim=0)
                all_peak_ints = torch.cat(peak_intens_list, dim=0)
                all_peak_mzs = torch.cat(peak_mzs_list, dim=0) if peak_mzs_list else torch.zeros(0, device=device)
                mol_peaks = torch.zeros((1, top_k_peaks, 320), dtype=torch.float32, device=device)
                mol_peak_mask = torch.ones((1, top_k_peaks), dtype=torch.bool, device=device)
                if len(all_peak_toks) > 0:
                    sorted_idx = torch.argsort(all_peak_ints, descending=True)
                    bins = (all_peak_mzs[sorted_idx] * 100).long()
                    _, unique_pos = np.unique(bins.cpu().numpy(), return_index=True)
                    dedup_idx = sorted_idx[torch.from_numpy(np.sort(unique_pos)).to(device)]
                    take = min(top_k_peaks, len(dedup_idx))
                    chosen = dedup_idx[:take]
                    mz_sort = torch.argsort(all_peak_mzs[chosen])
                    chosen = chosen[mz_sort]
                    mol_peaks[0, :take] = all_peak_toks[chosen]
                    mol_peak_mask[0, :take] = False

                mol_mem = torch.cat([meta_mem_sum / n, mol_peaks], dim=1)
                mol_mask = torch.cat([torch.zeros((1, 3), dtype=torch.bool, device=device), mol_peak_mask], dim=1)
                memory, mask = model.conditioned_memory(
                    mol_mem, mol_mask, torch.tensor([mass], device=device), pooled_sum / n, cond_sum / n
                )
            memories.append(memory)
            masks.append(mask)
            masses.append(mass)
            pred_fps.append((fp_sum / n).cpu().numpy())
            pred_maccs.append((maccs_sum / n).cpu().numpy())
            pred_elems.append((elem_sum.squeeze(0) / n).cpu().numpy())

        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            hypotheses = beam_search(model, torch.cat(memories), torch.cat(masks),
                                     beam_size=beam_size, max_len=max_len, top_k=top_k)

        for (molecule_id, _), mass, p_fp, p_mc, p_el, paths in zip(
            chunk, masses, pred_fps, pred_maccs, pred_elems, hypotheses
        ):
            unique = {}
            p_fp_clip = np.clip(p_fp, 1e-6, 1.0 - 1e-6)
            fp_logits = np.log(p_fp_clip) - np.log1p(-p_fp_clip)
            fp_log_not_p = float(np.sum(-np.logaddexp(0.0, fp_logits)))
            p_fp_sum = float(p_fp.sum())
            p_fp_norm = float(np.linalg.norm(p_fp))
            p_mc_sum = float(p_mc.sum())
            p_mc_norm = float(np.linalg.norm(p_mc))
            for ids, log_prob in paths:
                smi = tokenizer.decode(ids)
                raw_mol = Chem.MolFromSmiles(smi)
                if raw_mol is None or raw_mol.GetNumAtoms() == 0:
                    continue
                mol = evalkit.standardize_mol(raw_mol) or raw_mol
                if any(a.GetSymbol() not in ELEMENTS_ORDER for a in mol.GetAtoms()):
                    continue
                key = evalkit.fast_key(mol)
                if not key:
                    continue
                ppm = abs(rdMolDescriptors.CalcExactMolWt(mol) - mass) / mass * 1e6
                # Full Self-Spec2FP Discriminative Reranking (Morgan4096 + MACCS167 + NB + Elem)
                c_fp = evalkit.compute_morgan_fp(mol).astype(np.float32)
                c_mc = evalkit.compute_maccs_fp(mol).astype(np.float32)
                c_fp_sum = float(c_fp.sum())
                c_mc_sum = float(c_mc.sum())
                dot_fp = float(c_fp @ p_fp)
                cos_sim = dot_fp / max(math.sqrt(c_fp_sum) * p_fp_norm, 1e-8)
                tan_sim = dot_fp / max(c_fp_sum + p_fp_sum - dot_fp, 1e-8)
                nb_ll = (float(c_fp @ fp_logits) + fp_log_not_p) / 4096.0
                dot_mc = float(c_mc @ p_mc)
                maccs_cos = dot_mc / max(math.sqrt(c_mc_sum) * p_mc_norm, 1e-8)
                maccs_tan = dot_mc / max(c_mc_sum + p_mc_sum - dot_mc, 1e-8)
                elem_dist = float(np.abs(mol_elem_counts(mol) - p_el).sum())
                mass_score = -0.5 * (min(ppm, 50.0) / 7.5) ** 2

                # Hybrid generative + Stage06-aligned discriminative score
                score = (
                    log_prob
                    + 2.0 * nb_ll
                    + 1.8 * cos_sim
                    + 1.8 * tan_sim
                    + 0.6 * maccs_cos
                    + 0.6 * maccs_tan
                    + 0.40 * mass_score
                    - 0.04 * elem_dist
                )
                candidate = dict(
                    molecule_id=str(molecule_id),
                    smiles=Chem.MolToSmiles(mol, isomericSmiles=False),
                    fast_key=key, score=score, log_prob=log_prob,
                    nb_ll=nb_ll, cos_sim=cos_sim, tan_sim=tan_sim,
                    maccs_cos=maccs_cos, maccs_tan=maccs_tan,
                    elem_dist=elem_dist, ppm_err=ppm, source='spec2smiles'
                )
                if key not in unique or score > unique[key]['score']:
                    unique[key] = candidate

            # Tautomer-canonical deduplication on shortlist
            canon_seen = set()
            rank = 0
            for candidate in sorted(unique.values(), key=lambda x: -x['score']):
                ik14 = evalkit.canon_ik14(candidate['smiles'])
                if not ik14 or ik14 in canon_seen:
                    continue
                canon_seen.add(ik14)
                rank += 1
                candidate.update(rank=rank, ik14=ik14)
                rows.append(candidate)
                if rank >= top_k:
                    break

    return pd.DataFrame(rows, columns=COLUMNS).astype({
        'rank': 'int64', 'score': 'float64', 'log_prob': 'float64',
        'nb_ll': 'float64', 'cos_sim': 'float64', 'tan_sim': 'float64',
        'maccs_cos': 'float64', 'maccs_tan': 'float64',
        'elem_dist': 'float64', 'ppm_err': 'float64'
    })
