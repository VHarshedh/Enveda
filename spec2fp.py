"""spec2fp.py - Spectrum-to-Fingerprint model, tensorization, and inference.

Defines the Spec2FP Transformer, peak tensorization, and predict() API.
Import ONLY from /kaggle/working/.
"""
import json, math, sys, warnings, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ADDUCT_VOCAB = [
    "<pad>", "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+",
    "[M+H-H2O]+", "[M+FA-H]-", "[M+Cl]-", "[M]+", "[M-H2O+H]+",
    "[M-H2O-H]-", "[M+2H]2+", "[M-2H]2-", "[M+HAc-H]-",
    "[M+2Na-H]+", "[M+CH2O2-H]-", "<unk>",
]
ADDUCT2IDX = {a: i for i, a in enumerate(ADDUCT_VOCAB)}
INST_VOCAB = [
    "<pad>", "timsTOF", "qtof", "orbitrap", "FTICR", "QqQ",
    "ion_trap", "None/Unknown", "<unk>",
]
INST2IDX = {a: i for i, a in enumerate(INST_VOCAB)}
POLARITY_VOCAB = ["<pad>", "positive", "negative", "<unk>"]
POLARITY2IDX = {a: i for i, a in enumerate(POLARITY_VOCAB)}
ELEMENTS_ORDER = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
N_ELEMENTS = len(ELEMENTS_ORDER)


def parse_peak_array(val: Any) -> np.ndarray:
    """Robustly parse peak m/z or intensity into a float64 numpy array."""
    if isinstance(val, np.ndarray):
        return val.astype(np.float64)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float64)
    if isinstance(val, str):
        try:
            return np.asarray(json.loads(val), dtype=np.float64)
        except Exception:
            pass
    try:
        return np.asarray(val, dtype=np.float64)
    except Exception:
        return np.zeros(0, dtype=np.float64)


def sinusoidal_embed(values: np.ndarray, dim: int = 64) -> np.ndarray:
    """Multi-scale sinusoidal encoding resolving 0.005 Da mass defects up to 2000 Da."""
    freqs = np.logspace(
        math.log10(2 * math.pi / 2000.0),
        math.log10(2 * math.pi / 0.005),
        dim // 2, dtype=np.float32)
    angles = np.outer(values.astype(np.float32), freqs)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def tensorize_spectrum(
    mz: Any, inten: Any, precursor_mz: float,
    adduct: Any, ce: float, instrument: Any, polarity: Any,
    top_k: int = 192, mz_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """Convert a single spectrum into model-ready numpy arrays."""
    mz = parse_peak_array(mz)
    inten = parse_peak_array(inten)
    mask_valid = mz <= precursor_mz + 5.0
    mz, inten = mz[mask_valid], inten[mask_valid]
    if len(mz) == 0:
        mz = np.array([precursor_mz], dtype=np.float64)
        inten = np.array([1.0], dtype=np.float64)
    max_int = max(inten.max(), 1e-12)
    inten = np.sqrt(inten / max_int)
    if len(mz) > top_k:
        idx = np.argsort(inten)[-top_k:]
        idx = idx[np.argsort(mz[idx])]
        mz, inten = mz[idx], inten[idx]
    n = len(mz)
    mz_emb = sinusoidal_embed(mz, mz_dim)
    nl = precursor_mz - mz
    nl_emb = sinusoidal_embed(nl, mz_dim)
    log_int = np.log1p(inten).astype(np.float32).reshape(-1, 1)
    peak_feat = np.concatenate([mz_emb, log_int, nl_emb], axis=1).astype(np.float32)
    adduct_idx = int(adduct) if isinstance(adduct, (int, np.integer)) else ADDUCT2IDX.get(str(adduct).strip(), ADDUCT2IDX["<unk>"])
    inst_idx = int(instrument) if isinstance(instrument, (int, np.integer)) else INST2IDX.get(str(instrument).strip(), INST2IDX["<unk>"])
    pol_idx = int(polarity) if isinstance(polarity, (int, np.integer)) else POLARITY2IDX.get(str(polarity).strip().lower(), POLARITY2IDX["<unk>"])
    ce_val = float(ce) if not (ce is None or (isinstance(ce, float) and math.isnan(ce))) else 0.0
    return {
        "peak_feat": peak_feat, "n_peaks": n,
        "adduct_idx": adduct_idx, "inst_idx": inst_idx, "pol_idx": pol_idx,
        "ce_val": np.float32(ce_val), "prec_mz": np.float32(precursor_mz),
    }


def collate_spectra(batch: list, top_k: int = 192, feat_dim: int = 129):
    """Collate list of tensorize_spectrum outputs into batched tensors."""
    B = len(batch)
    max_len = top_k + 3
    peaks = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(B, max_len, dtype=torch.bool)
    adducts = torch.zeros(B, dtype=torch.long)
    instruments = torch.zeros(B, dtype=torch.long)
    polarities = torch.zeros(B, dtype=torch.long)
    ces = torch.zeros(B, dtype=torch.float32)
    prec_mzs = torch.zeros(B, dtype=torch.float32)
    for i, item in enumerate(batch):
        n = item["n_peaks"]
        peaks[i, :n] = torch.from_numpy(item["peak_feat"][:n])
        mask[i, :n] = False
        adducts[i] = item["adduct_idx"]
        instruments[i] = item["inst_idx"]
        polarities[i] = item["pol_idx"]
        ces[i] = float(item["ce_val"])
        prec_mzs[i] = float(item["prec_mz"])
    return {
        "peaks": peaks, "pad_mask": mask,
        "adduct": adducts, "instrument": instruments, "polarity": polarities,
        "ce": ces, "prec_mz": prec_mzs,
    }


class Spec2FPModel(nn.Module):
    """Transformer encoder mapping MS/MS spectrum to molecular fingerprint."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d, n_layers, n_heads = cfg.get("d_model", 320), cfg.get("n_layers", 5), cfg.get("n_heads", 8)
        dropout, feat_dim = cfg.get("dropout", 0.1), cfg.get("feat_dim", 129)
        fp_bits, maccs_bits = cfg.get("fp_bits", 4096), cfg.get("maccs_bits", 167)
        n_elements = cfg.get("n_elements", N_ELEMENTS)
        self.peak_proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.adduct_emb = nn.Embedding(len(ADDUCT_VOCAB), d, padding_idx=0)
        self.inst_emb = nn.Embedding(len(INST_VOCAB), d, padding_idx=0)
        self.pol_emb = nn.Embedding(len(POLARITY_VOCAB), d, padding_idx=0)
        self.ce_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())
        self.prec_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)
        self.fp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, fp_bits))
        self.maccs_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, maccs_bits))
        self.heavy_atom_head = nn.Linear(d, 1)
        self.ring_count_head = nn.Linear(d, 1)
        self.elem_presence_head = nn.Linear(d, n_elements)
        self.elem_count_head = nn.Linear(d, n_elements)

    def forward(self, peaks, pad_mask, adduct, instrument, polarity, ce, prec_mz):
        B = peaks.size(0)
        x = self.peak_proj(peaks)
        cls = self.cls_token.expand(B, -1, -1)
        meta1 = (
            self.adduct_emb(adduct)
            + self.ce_proj(ce.unsqueeze(-1) / 100.0)
            + self.prec_proj(prec_mz.unsqueeze(-1) / 1000.0)
        )
        meta2 = self.inst_emb(instrument) + self.pol_emb(polarity)
        meta = torch.stack([meta1, meta2], dim=1)
        x = torch.cat([cls, meta, x], dim=1)
        cls_meta_mask = torch.zeros(B, 3, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_meta_mask, pad_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=full_mask)
        cls_out = x[:, 0]
        valid_tok = (~full_mask).float().unsqueeze(-1)
        mean_out = (x * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp(min=1.0)
        x = self.norm(cls_out + mean_out)
        return {
            "fp_logits": self.fp_head(x), "maccs_logits": self.maccs_head(x),
            "heavy_atoms": self.heavy_atom_head(x).squeeze(-1),
            "ring_count": self.ring_count_head(x).squeeze(-1),
            "elem_presence": self.elem_presence_head(x),
            "elem_count": self.elem_count_head(x),
        }


def load_model(config_path: str, weights_path: str, device: str = "cpu"):
    """Load Spec2FP model from config + weights."""
    with open(config_path) as f:
        cfg = json.load(f)
    model = Spec2FPModel(cfg["model"])
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    if "ema_state_dict" in sd:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd["ema_state_dict"].items()})
    elif "model_state_dict" in sd:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd["model_state_dict"].items()})
    else:
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    pw_path = Path(weights_path).parent / "spec2fp_pos_weights.npy"
    if pw_path.is_file():
        try:
            model.pos_weights = np.load(pw_path)
        except Exception:
            pass
    model.eval()
    return model.to(device), cfg


@torch.no_grad()
def predict_full(model, clean_df, device="cuda", batch_size=128, top_k=192, mz_dim=64, pos_weights=None):
    """Predict Morgan-4096, MACCS-167, and element counts with streaming molecule-level pooling."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    pw = pos_weights if pos_weights is not None else getattr(raw_model, "pos_weights", None)
    pw_shift = 0.5 * torch.log(torch.as_tensor(pw, device=device, dtype=torch.float32).clamp(min=1.0)) if pw is not None else None
    feat_dim, n_rows = 2 * mz_dim + 1, len(clean_df)
    if n_rows == 0:
        return (np.zeros((0, 4096), dtype=np.float16), np.zeros((0, 167), dtype=np.float16),
                np.zeros((0, N_ELEMENTS), dtype=np.float32), [])

    mol_ids_arr = clean_df["molecule_id"].to_numpy().astype(str)
    unique_mols, inv_indices = np.unique(mol_ids_arr, return_inverse=True)
    n_unique = len(unique_mols)
    fp_sum = np.zeros((n_unique, 4096), dtype=np.float32)
    maccs_sum = np.zeros((n_unique, 167), dtype=np.float32)
    elem_sum = np.zeros((n_unique, N_ELEMENTS), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.float32)

    mz_col = clean_df["mz"].to_numpy()
    int_col = clean_df["inten"].to_numpy()
    pmz_col = clean_df["precursor_mz"].to_numpy(dtype=np.float32)
    add_col = clean_df["adduct"].to_numpy() if "adduct" in clean_df.columns else np.array(["[M+H]+"] * n_rows)
    ce_col = pd.to_numeric(clean_df["ce"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "ce" in clean_df.columns else np.zeros(n_rows, dtype=np.float32)
    inst_col = clean_df["instrument"].to_numpy() if "instrument" in clean_df.columns else np.array(["unknown"] * n_rows)
    pol_col = clean_df["polarity"].to_numpy() if "polarity" in clean_df.columns else np.array(["positive"] * n_rows)
    dev_type, is_cuda = ("cuda" if "cuda" in str(device) else "cpu"), ("cuda" in str(device))

    for s_idx in range(0, n_rows, batch_size):
        e_idx = min(s_idx + batch_size, n_rows)
        items = [
            tensorize_spectrum(
                mz_col[r], int_col[r], float(pmz_col[r]),
                str(add_col[r]), float(ce_col[r]),
                str(inst_col[r]), str(pol_col[r]), top_k=top_k, mz_dim=mz_dim)
            for r in range(s_idx, e_idx)
        ]
        bd = collate_spectra(items, top_k=top_k, feat_dim=feat_dim)
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bd.items()}
        with torch.amp.autocast(device_type=dev_type, dtype=torch.float16, enabled=is_cuda):
            out = raw_model(**bd)
        fp_logits = out["fp_logits"] - pw_shift if pw_shift is not None else out["fp_logits"]
        batch_fp = torch.sigmoid(fp_logits).float().cpu().numpy()
        batch_maccs = torch.sigmoid(out["maccs_logits"]).float().cpu().numpy()
        batch_elem = out["elem_count"].float().cpu().numpy()
        batch_inv = inv_indices[s_idx:e_idx]
        np.add.at(fp_sum, batch_inv, batch_fp)
        np.add.at(maccs_sum, batch_inv, batch_maccs)
        np.add.at(elem_sum, batch_inv, batch_elem)
        np.add.at(counts, batch_inv, 1.0)

    counts_col = np.maximum(counts, 1.0)[:, None]
    return ((fp_sum / counts_col).astype(np.float16),
            (maccs_sum / counts_col).astype(np.float16),
            (elem_sum / counts_col).astype(np.float32), unique_mols.tolist())


@torch.no_grad()
def predict(model, clean_df, device="cuda", batch_size=128, top_k=192, mz_dim=64, pos_weights=None, return_maccs=False):
    """Predict fingerprints with optional MACCS return for backward-compatibility."""
    fp_pred, maccs_pred, elem_pred, mol_ids = predict_full(
        model, clean_df, device=device, batch_size=batch_size, top_k=top_k, mz_dim=mz_dim, pos_weights=pos_weights
    )
    if return_maccs:
        return fp_pred, maccs_pred, elem_pred, mol_ids
    return fp_pred, elem_pred, mol_ids
