"""libretrieval.py - Spectral library retrieval baseline module for LC-MS/MS data.

Consolidates the spectral retrieval baseline from Stage 03 (Spectral Library Retrieval Baseline)
to enable downstream notebooks and pipelines to run fast library searches against
arbitrary query tables.
"""

from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize
import torch

try:
    from preprocess import neutral_mass_from_precursor
except ImportError:
    # Standalone fallback if preprocess.py is not yet in Python path
    _PROTON_MASS = 1.00727646677
    _ADDUCT_FALLBACK = {
        "[M+H]+": 1.00727646677,
        "[M-H]-": -1.00727646677,
        "[M+Na]+": 22.989218,
        "[M+NH4]+": 18.033823,
        "[M+K]+": 38.963158,
    }

    def neutral_mass_from_precursor(precursor_mz, adduct, polarity=None):
        adduct_str = str(adduct).strip() if adduct is not None else ""
        if adduct_str in _ADDUCT_FALLBACK:
            return float(precursor_mz) - _ADDUCT_FALLBACK[adduct_str]
        pol = str(polarity).strip().lower() if polarity is not None else "positive"
        if "neg" in pol or pol.startswith("-"):
            return float(precursor_mz) + _PROTON_MASS
        return float(precursor_mz) - _PROTON_MASS


# ── CONFIGURATION DATACLASS ──────────────────────────────────────────────────
@dataclass
class LibRetrievalConfig:
    """Hyperparameters and batching thresholds for spectral library retrieval."""
    MAX_MZ: float = 1500.0
    BIN_COARSE: float = 0.1     # 15 000 bins (0.1 Da)
    BIN_FINE: float = 0.01      # 150 000 bins (0.01 Da)
    MAX_PEAKS: int = 200        # per-spectrum peak capacity cap

    # Mass pre-filter window for modified cosine & entropy similarity
    MASS_WINDOW: float = 2.0    # Da

    # Modified cosine tolerance parameters
    DA_TOL: float = 0.01        # minimum absolute tolerance in Da
    PPM_TOL: float = 10e-6      # 10 ppm relative tolerance

    # GPU / CPU batch chunking sizes
    COS_LIB_CHUNK: int = 30000
    COS_QUERY_CHUNK: int = 5000
    MOD_COS_BATCH: int = 2048
    TOP_K: int = 100

    # Device configuration (None = auto-detect CUDA else CPU)
    device: Optional[str] = None


# ── SPECTRUM VECTORIZATION / AGGREGATION ─────────────────────────────────────
def build_query_vectors(
    df_clean: pd.DataFrame,
    cfg: Optional[LibRetrievalConfig] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Aggregate spectra per unique molecule_id and build sparse representations.

    Uses contiguously sorted integer indices and searchsorted boundaries to
    eliminate pandas groupby/iterrows overhead.
    Constructs:
      - Coarse (0.1 Da) and fine (0.01 Da) sqrt-intensity binned CSR matrices.
      - Fixed-shape padded peak tensors for the best spectrum per molecule
        (selected by highest summed root intensity).
    """
    if cfg is None:
        cfg = LibRetrievalConfig()

    n_bins_c = int(cfg.MAX_MZ / cfg.BIN_COARSE)   # 15 000
    n_bins_f = int(cfg.MAX_MZ / cfg.BIN_FINE)     # 150 000

    if len(df_clean) == 0:
        empty_coarse = csr_matrix((0, n_bins_c), dtype=np.float32)
        empty_fine = csr_matrix((0, n_bins_f), dtype=np.float32)
        return {
            "mol_ids": np.array([], dtype=str),
            "sp_coarse": empty_coarse,
            "sp_fine": empty_fine,
            "peak_mzs": np.zeros((0, cfg.MAX_PEAKS), dtype=np.float32),
            "peak_ints": np.zeros((0, cfg.MAX_PEAKS), dtype=np.float32),
            "peak_mask": np.zeros((0, cfg.MAX_PEAKS), dtype=np.bool_),
            "prec_mass": np.zeros(0, dtype=np.float32),
            "neutral_mass": np.zeros(0, dtype=np.float32),
        }

    # Deduplicate and sort molecule IDs
    mol_ids = sorted(df_clean["molecule_id"].astype(str).unique())
    n_mols = len(mol_ids)
    mol_id_to_idx = {m: i for i, m in enumerate(mol_ids)}

    # Pre-allocate peak arrays for modified cosine
    mol_peak_mzs = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.float32)
    mol_peak_ints = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.float32)
    mol_peak_mask = np.zeros((n_mols, cfg.MAX_PEAKS), dtype=np.bool_)
    mol_prec_mass = np.zeros(n_mols, dtype=np.float32)
    mol_best_score = np.full(n_mols, -1.0, dtype=np.float32)

    df_work = df_clean.copy()
    df_work["midx"] = df_work["molecule_id"].astype(str).map(mol_id_to_idx).astype(np.int32)
    df_work = df_work.sort_values("midx").reset_index(drop=True)

    # Resolve peak array and precursor m/z column names
    mz_col = "mz" if "mz" in df_work.columns else ("mzs" if "mzs" in df_work.columns else "mz_array")
    int_col = "inten" if "inten" in df_work.columns else ("intensities" if "intensities" in df_work.columns else "intensity_array")
    pmz_col = "precursor_mz" if "precursor_mz" in df_work.columns else "prec_mz"

    midx_arr = df_work["midx"].values
    mzs_arr = df_work[mz_col].values
    ints_arr = df_work[int_col].values
    prec_arr = df_work[pmz_col].values.astype(np.float32)

    boundaries = np.searchsorted(midx_arr, np.arange(n_mols + 1))

    indptr_c = [0]; indices_c = []; data_c = []
    indptr_f = [0]; indices_f = []; data_f = []

    _buf_c = np.zeros(n_bins_c, dtype=np.float32)
    _buf_f = np.zeros(n_bins_f, dtype=np.float32)

    for midx in range(n_mols):
        s_start = boundaries[midx]
        s_end = boundaries[midx + 1]

        touched_c = []
        touched_f = []

        for s_idx in range(s_start, s_end):
            mzs = np.asarray(mzs_arr[s_idx], dtype=np.float64)
            ints = np.asarray(ints_arr[s_idx], dtype=np.float64)
            prec = float(prec_arr[s_idx])

            # Filter: peaks <= precursor + 5 Da, non-negative, intensity > 0
            keep = (mzs <= prec + 5.0) & (mzs >= 0) & (ints > 0)
            mzs, ints = mzs[keep], ints[keep]
            if len(mzs) == 0:
                continue

            ints = np.sqrt(np.clip(ints, 0, None))

            # Coarse bins (0.1 Da)
            bc = np.clip((mzs / cfg.BIN_COARSE).astype(np.int32), 0, n_bins_c - 1)
            np.maximum.at(_buf_c, bc, ints)
            touched_c.append(bc)

            # Fine bins (0.01 Da)
            bf = np.clip((mzs / cfg.BIN_FINE).astype(np.int32), 0, n_bins_f - 1)
            np.maximum.at(_buf_f, bf, ints)
            touched_f.append(bf)

            # Select best spectrum per molecule by max sum of root-intensities
            s = float(ints.sum())
            if s > mol_best_score[midx]:
                mol_best_score[midx] = s
                if len(mzs) > cfg.MAX_PEAKS:
                    top = np.argsort(ints)[-cfg.MAX_PEAKS:]
                    mzs_t, ints_t = mzs[top], ints[top]
                else:
                    mzs_t, ints_t = mzs, ints
                n = len(mzs_t)
                mol_peak_mzs[midx, :] = 0
                mol_peak_ints[midx, :] = 0
                mol_peak_mask[midx, :] = False
                mol_peak_mzs[midx, :n] = mzs_t.astype(np.float32)
                mol_peak_ints[midx, :n] = ints_t.astype(np.float32)
                mol_peak_mask[midx, :n] = True
                mol_prec_mass[midx] = prec

        # Coarse sparse row: wipe only touched bins (avoids scanning 150k elements)
        if touched_c:
            uniq_c = np.unique(np.concatenate(touched_c))
            vals_c = _buf_c[uniq_c]
            nc = np.linalg.norm(vals_c)
            if nc > 0:
                vals_c /= nc
            indices_c.append(uniq_c.astype(np.int32))
            data_c.append(vals_c.astype(np.float32))
            indptr_c.append(indptr_c[-1] + len(uniq_c))
            _buf_c[uniq_c] = 0.0
        else:
            indptr_c.append(indptr_c[-1])

        # Fine sparse row: wipe only touched bins
        if touched_f:
            uniq_f = np.unique(np.concatenate(touched_f))
            vals_f = _buf_f[uniq_f]
            nf = np.linalg.norm(vals_f)
            if nf > 0:
                vals_f /= nf
            indices_f.append(uniq_f.astype(np.int32))
            data_f.append(vals_f.astype(np.float32))
            indptr_f.append(indptr_f[-1] + len(uniq_f))
            _buf_f[uniq_f] = 0.0
        else:
            indptr_f.append(indptr_f[-1])

    # Assemble L2-normalized CSR sparse matrices
    data_c_arr = np.concatenate(data_c) if data_c else np.array([], dtype=np.float32)
    indices_c_arr = np.concatenate(indices_c) if indices_c else np.array([], dtype=np.int32)
    indptr_c_arr = np.array(indptr_c, dtype=np.int32)
    sp_coarse = csr_matrix((data_c_arr, indices_c_arr, indptr_c_arr), shape=(n_mols, n_bins_c))

    data_f_arr = np.concatenate(data_f) if data_f else np.array([], dtype=np.float32)
    indices_f_arr = np.concatenate(indices_f) if indices_f else np.array([], dtype=np.int32)
    indptr_f_arr = np.array(indptr_f, dtype=np.int32)
    sp_fine = csr_matrix((data_f_arr, indices_f_arr, indptr_f_arr), shape=(n_mols, n_bins_f))

    # Resolve neutral masses
    df_first = df_work.drop_duplicates("molecule_id").set_index("molecule_id")
    if "neutral_mass" in df_first.columns:
        neutral_mass = df_first.loc[mol_ids, "neutral_mass"].values.astype(np.float32)
    else:
        neutral_mass = np.array([
            neutral_mass_from_precursor(
                float(df_first.loc[m, pmz_col]),
                str(df_first.loc[m, "adduct"]) if "adduct" in df_first.columns else "[M+H]+",
                str(df_first.loc[m, "polarity"]) if "polarity" in df_first.columns else "+",
            )
            for m in mol_ids
        ], dtype=np.float32)

    bundle = {
        "mol_ids": np.array(mol_ids),
        "sp_coarse": sp_coarse,
        "sp_fine": sp_fine,
        "peak_mzs": mol_peak_mzs,
        "peak_ints": mol_peak_ints,
        "peak_mask": mol_peak_mask,
        "prec_mass": mol_prec_mass,
        "neutral_mass": neutral_mass,
    }

    if "smiles" in df_first.columns:
        bundle["smiles"] = df_first.loc[mol_ids, "smiles"].fillna("").astype(str).values
    if "ik14" in df_first.columns:
        bundle["ik14"] = df_first.loc[mol_ids, "ik14"].fillna("").astype(str).values

    if verbose:
        print(f"Built vectors for {n_mols} molecules.")
        print(f"  sp_coarse: {sp_coarse.shape}, nnz={sp_coarse.nnz}")
        print(f"  sp_fine:   {sp_fine.shape}, nnz={sp_fine.nnz}")

    return bundle


# ── SIMILARITY KERNELS ───────────────────────────────────────────────────────
def gpu_cosine_topk(
    query_sp: csr_matrix,
    lib_sp: csr_matrix,
    top_k: int = 100,
    q_chunk: int = 5000,
    l_chunk: int = 30000,
    device: Optional[Union[str, torch.device]] = None,
    desc: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarity between sparse query and library matrices.

    Branching strategy:
      - High-dimensional fine representations (>20,000 bins): uses chunked
        sparse CSC dot products to avoid allocating multi-gigabyte dense matrices.
      - Coarse representations (<=20,000 bins): uses PyTorch dense matrix
        multiplications chunked along both query and library axes.

    Returns:
      (best_scores, best_idx) numpy arrays of shape (n_query, top_k).
    """
    n_q = query_sp.shape[0]
    n_l = lib_sp.shape[0]
    actual_k = min(top_k, n_l)

    best_scores = np.full((n_q, top_k), -1.0, dtype=np.float32)
    best_idx = np.full((n_q, top_k), -1, dtype=np.int64)

    if n_q == 0 or actual_k == 0:
        return best_scores, best_idx

    # High-dimensional branch (e.g. 150 000 bins at 0.01 Da)
    if query_sp.shape[1] > 20000:
        lib_csc = lib_sp.T.tocsc()
        for qs in range(0, n_q, q_chunk):
            qe = min(qs + q_chunk, n_q)
            sim_chunk = query_sp[qs:qe].dot(lib_csc)
            for i in range(qe - qs):
                r_start = sim_chunk.indptr[i]
                r_end = sim_chunk.indptr[i + 1]
                r_len = r_end - r_start
                if r_len == 0:
                    continue
                r_data = sim_chunk.data[r_start:r_end]
                r_cols = sim_chunk.indices[r_start:r_end]
                k = min(actual_k, r_len)
                if r_len > k:
                    part = np.argpartition(r_data, -k)[-k:]
                    sort_order = part[np.argsort(-r_data[part])]
                else:
                    sort_order = np.argsort(-r_data)
                best_scores[qs + i, :k] = r_data[sort_order]
                best_idx[qs + i, :k] = r_cols[sort_order]
        del lib_csc
        gc.collect()
        return best_scores, best_idx

    # Coarse branch (dense matmul chunking)
    if device is None:
        torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_dev = torch.device(device)

    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        Q = torch.from_numpy(query_sp[qs:qe].toarray().astype(np.float32)).to(torch_dev)

        cur_topk_scores = torch.full((qe - qs, actual_k), -1.0, device=torch_dev)
        cur_topk_idx = torch.full((qe - qs, actual_k), -1, dtype=torch.long, device=torch_dev)

        for ls in range(0, n_l, l_chunk):
            le = min(ls + l_chunk, n_l)
            L = torch.from_numpy(lib_sp[ls:le].toarray().astype(np.float32)).to(torch_dev)
            sim = Q @ L.t()  # (q_chunk, l_chunk)
            chunk_k = min(actual_k, le - ls)
            vals, idxs = sim.topk(chunk_k, dim=1)
            idxs = idxs + ls

            # Merge chunk results into running top-k
            merged_s = torch.cat([cur_topk_scores, vals], dim=1)
            merged_i = torch.cat([cur_topk_idx, idxs], dim=1)
            topk_s, sel = merged_s.topk(actual_k, dim=1)
            cur_topk_scores = topk_s
            cur_topk_idx = merged_i.gather(1, sel)
            del L, sim, vals, idxs, merged_s, merged_i

        best_scores[qs:qe, :actual_k] = cur_topk_scores.cpu().numpy()
        best_idx[qs:qe, :actual_k] = cur_topk_idx.cpu().numpy()
        del Q, cur_topk_scores, cur_topk_idx
        if torch.cuda.is_available() and torch_dev.type == "cuda":
            torch.cuda.empty_cache()

    return best_scores, best_idx


def modified_cosine_batch(
    q_mzs: torch.Tensor,
    q_ints: torch.Tensor,
    q_masks: torch.Tensor,
    l_mzs: torch.Tensor,
    l_ints: torch.Tensor,
    l_masks: torch.Tensor,
    shifts: torch.Tensor,
    da_tol: float = 0.01,
    ppm_tol: float = 10e-6,
) -> torch.Tensor:
    """Compute batched modified cosine with neutral-loss matching on PyTorch.

    Args:
      q_mzs, q_ints, q_masks: (B, MAX_PEAKS) tensors for query.
      l_mzs, l_ints, l_masks: (B, MAX_PEAKS) tensors for library candidates.
      shifts: (B,) precursor mass difference tensor (query_prec - lib_prec).
      da_tol: minimum tolerance in Da.
      ppm_tol: relative tolerance in ppm.

    Returns:
      (B,) cosine similarity scores.
    """
    # Peak difference tensor: (B, Q, 1) - (B, 1, L) -> (B, Q, L)
    diff = q_mzs.unsqueeze(2) - l_mzs.unsqueeze(1)

    # Adaptive tolerance based on mean m/z
    mean_mz = (q_mzs.unsqueeze(2).abs() + l_mzs.unsqueeze(1).abs()) / 2 + 1e-6
    tol = torch.clamp(mean_mz * ppm_tol, min=da_tol)

    # Direct peak match or neutral-loss (shifted) match
    direct = diff.abs() <= tol
    shifted = (diff - shifts.view(-1, 1, 1)).abs() <= tol
    match = direct | shifted

    # Valid non-padding pair mask
    valid = q_masks.unsqueeze(2) & l_masks.unsqueeze(1)
    match = match & valid

    # Intensity products for matched pairs
    prods = q_ints.unsqueeze(2) * l_ints.unsqueeze(1)
    prods = prods * match.float()

    # Greedy best match per query peak
    best_per_q = prods.max(dim=2).values  # (B, Q)
    matched_sum = (best_per_q * q_masks.float()).sum(dim=1)

    # Normalization
    q_norm = (q_ints**2 * q_masks.float()).sum(dim=1).sqrt()
    l_norm = (l_ints**2 * l_masks.float()).sum(dim=1).sqrt()
    denom = q_norm * l_norm + 1e-8
    return matched_sum / denom


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy across rows of a probability distribution."""
    safe = torch.clamp(p, min=1e-12)
    return -(safe * safe.log()).sum(dim=-1)


def entropy_sim_topk(
    query_sp_raw: csr_matrix,
    lib_sp_raw: csr_matrix,
    query_masses: np.ndarray,
    lib_masses: np.ndarray,
    mass_window: float = 2.0,
    top_k: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    desc: str = "entropy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute JSD-based spectral entropy similarity with mass pre-filtering.

    L1-normalizes the raw binned representations and computes 1 - JSD within
    precursor mass candidate windows.

    Returns:
      (best_scores, best_idx) numpy arrays of shape (n_query, top_k).
    """
    if device is None:
        torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_dev = torch.device(device)

    # L1-normalize to probability distributions
    q_prob = sk_normalize(query_sp_raw, norm="l1", axis=1)
    l_prob = sk_normalize(lib_sp_raw, norm="l1", axis=1)

    lib_order = np.argsort(lib_masses)
    lib_masses_s = lib_masses[lib_order]
    n_q = q_prob.shape[0]

    best_s = np.full((n_q, top_k), -1.0, dtype=np.float32)
    best_i = np.full((n_q, top_k), -1, dtype=np.int64)

    for qi in range(n_q):
        qm = query_masses[qi]
        lo = np.searchsorted(lib_masses_s, qm - mass_window, side="left")
        hi = np.searchsorted(lib_masses_s, qm + mass_window, side="right")
        cand = lib_order[lo:hi]
        if len(cand) == 0:
            continue

        q_v = torch.from_numpy(q_prob[qi].toarray().astype(np.float32)).to(torch_dev)    # (1, D)
        l_v = torch.from_numpy(l_prob[cand].toarray().astype(np.float32)).to(torch_dev)  # (C, D)

        H_q = _entropy(q_v).expand(len(cand))
        H_l = _entropy(l_v)
        H_m = _entropy((q_v + l_v) / 2)
        jsd = (2 * H_m - H_q - H_l) / math.log(4)
        sim = 1.0 - jsd

        k = min(top_k, len(cand))
        topv, topi = sim.topk(k)
        best_s[qi, :k] = topv.cpu().numpy()
        best_i[qi, :k] = cand[topi.cpu().numpy()]
        del q_v, l_v

    if torch.cuda.is_available() and torch_dev.type == "cuda":
        torch.cuda.empty_cache()

    return best_s, best_i


def _run_modified_cosine_retrieval(
    query_bundle: Dict[str, Any],
    library_bundle: Dict[str, Any],
    cfg: LibRetrievalConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute candidate window retrieval using batched modified cosine."""
    n_query = len(query_bundle["mol_ids"])
    lib_masses = library_bundle["neutral_mass"]
    lib_m_order = np.argsort(lib_masses)
    lib_m_sorted = lib_masses[lib_m_order]

    # Transfer library peak arrays to device
    l_mzs_all = torch.from_numpy(library_bundle["peak_mzs"]).to(device)
    l_ints_all = torch.from_numpy(library_bundle["peak_ints"]).to(device)
    l_mask_all = torch.from_numpy(library_bundle["peak_mask"]).to(device)
    l_prec_all = library_bundle["prec_mass"]

    # Transfer query peak arrays to device
    q_mzs_all = torch.from_numpy(query_bundle["peak_mzs"]).to(device)
    q_ints_all = torch.from_numpy(query_bundle["peak_ints"]).to(device)
    q_mask_all = torch.from_numpy(query_bundle["peak_mask"]).to(device)
    q_prec_all = query_bundle["prec_mass"]
    q_mass_all = query_bundle["neutral_mass"]

    mod_sc_list: List[np.ndarray] = []
    mod_id_list: List[np.ndarray] = []

    for qi in range(n_query):
        qm = q_mass_all[qi]
        lo = np.searchsorted(lib_m_sorted, qm - cfg.MASS_WINDOW, side="left")
        hi = np.searchsorted(lib_m_sorted, qm + cfg.MASS_WINDOW, side="right")
        cand = lib_m_order[lo:hi]

        if len(cand) == 0:
            mod_sc_list.append(np.full(cfg.TOP_K, -1.0, dtype=np.float32))
            mod_id_list.append(np.full(cfg.TOP_K, -1, dtype=np.int64))
            continue

        nc = len(cand)
        q_mzs_b = q_mzs_all[qi].unsqueeze(0).expand(nc, -1)
        q_ints_b = q_ints_all[qi].unsqueeze(0).expand(nc, -1)
        q_mask_b = q_mask_all[qi].unsqueeze(0).expand(nc, -1)

        cand_t = torch.from_numpy(cand).long().to(device)
        l_mzs_b = l_mzs_all[cand_t]
        l_ints_b = l_ints_all[cand_t]
        l_mask_b = l_mask_all[cand_t]

        shifts_b = (
            torch.tensor(q_prec_all[qi], device=device)
            - torch.from_numpy(l_prec_all[cand]).to(device)
        )

        all_sc: List[torch.Tensor] = []
        for bs in range(0, nc, cfg.MOD_COS_BATCH):
            be = min(bs + cfg.MOD_COS_BATCH, nc)
            sc = modified_cosine_batch(
                q_mzs_b[bs:be], q_ints_b[bs:be], q_mask_b[bs:be],
                l_mzs_b[bs:be], l_ints_b[bs:be], l_mask_b[bs:be],
                shifts_b[bs:be], cfg.DA_TOL, cfg.PPM_TOL,
            )
            all_sc.append(sc)

        sc_t = torch.cat(all_sc)
        k = min(cfg.TOP_K, nc)
        topv, topi = sc_t.topk(k)

        rs = np.full(cfg.TOP_K, -1.0, dtype=np.float32)
        ri = np.full(cfg.TOP_K, -1, dtype=np.int64)
        rs[:k] = topv.cpu().numpy()
        ri[:k] = cand[topi.cpu().numpy()]

        mod_sc_list.append(rs)
        mod_id_list.append(ri)

    mod_sc = np.stack(mod_sc_list) if mod_sc_list else np.empty((0, cfg.TOP_K), dtype=np.float32)
    mod_id = np.stack(mod_id_list) if mod_id_list else np.empty((0, cfg.TOP_K), dtype=np.int64)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return mod_sc, mod_id


# ── LIBRARY LOADING AND SAVING ROUTINES ──────────────────────────────────────
def load_library(artifacts_dir: str) -> Dict[str, Any]:
    """Load precomputed spectral library artifacts from disk.

    Reads:
      - lib_config.json
      - lib_index.npz
      - lib_sp_coarse.npz
      - lib_sp_fine.npz

    Returns:
      In-memory dictionary bundle containing metadata, CSR matrices, and peak tensors.
    """
    cfg_path = os.path.join(artifacts_dir, "lib_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            lib_config = json.load(f)
    else:
        lib_config = {"best_method": "cos_01", "source_tag": "lib_cos"}

    coarse_path = os.path.join(artifacts_dir, "lib_sp_coarse.npz")
    fine_path = os.path.join(artifacts_dir, "lib_sp_fine.npz")
    idx_path = os.path.join(artifacts_dir, "lib_index.npz")

    if not os.path.exists(coarse_path):
        raise FileNotFoundError(f"Missing library coarse matrix: {coarse_path}")
    if not os.path.exists(fine_path):
        raise FileNotFoundError(f"Missing library fine matrix: {fine_path}")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing library index: {idx_path}")

    sp_coarse = scipy.sparse.load_npz(coarse_path)
    sp_fine = scipy.sparse.load_npz(fine_path)
    idx_data = np.load(idx_path, allow_pickle=True)

    mol_ids = idx_data["mol_ids"]
    neutral_mass = idx_data["neutral_mass"]
    peak_mzs = idx_data["peak_mzs"]
    peak_ints = idx_data["peak_ints"]
    peak_mask = idx_data["peak_mask"]
    prec_mass = idx_data["prec_mass"]

    smiles = idx_data["smiles"] if "smiles" in idx_data else None
    ik14 = idx_data["ik14"] if "ik14" in idx_data else None

    # Fallback structure discovery if not packed directly into index
    if smiles is None or ik14 is None:
        cand_meta_paths = [
            os.path.join(artifacts_dir, "train_clean.parquet"),
            os.path.join(artifacts_dir, "..", "eda", "train_clean.parquet"),
            os.path.join(artifacts_dir, "splits.parquet"),
            os.path.join(artifacts_dir, "..", "eval_splits", "splits.parquet"),
        ]
        meta_found = False
        for mp in cand_meta_paths:
            if os.path.exists(mp):
                try:
                    df_meta = pd.read_parquet(mp)
                    if "molecule_id" in df_meta.columns:
                        df_m = df_meta.drop_duplicates("molecule_id").set_index("molecule_id")
                        if smiles is None and "smiles" in df_m.columns:
                            smiles = np.array([str(df_m.loc[m, "smiles"]) if m in df_m.index else "" for m in mol_ids])
                        if ik14 is None and "ik14" in df_m.columns:
                            ik14 = np.array([str(df_m.loc[m, "ik14"]) if m in df_m.index else str(m) for m in mol_ids])
                        meta_found = True
                        break
                except Exception:
                    pass
        if not meta_found:
            if smiles is None:
                smiles = np.array([""] * len(mol_ids))
            if ik14 is None:
                ik14 = np.array([str(m) for m in mol_ids])

    return {
        "config": lib_config,
        "sp_coarse": sp_coarse,
        "sp_fine": sp_fine,
        "mol_ids": np.array(mol_ids),
        "neutral_mass": neutral_mass,
        "peak_mzs": peak_mzs,
        "peak_ints": peak_ints,
        "peak_mask": peak_mask,
        "prec_mass": prec_mass,
        "smiles": np.array(smiles),
        "ik14": np.array(ik14),
    }


def build_library(
    df_clean: pd.DataFrame,
    cfg: Optional[LibRetrievalConfig] = None,
    best_method: str = "cos_01",
    source_tag: str = "lib_cos",
) -> Dict[str, Any]:
    """Construct a full library bundle directly from clean library spectra."""
    if cfg is None:
        cfg = LibRetrievalConfig()

    bundle = build_query_vectors(df_clean, cfg=cfg)
    bundle["config"] = {
        "best_method": best_method,
        "source_tag": source_tag,
        "params": asdict(cfg),
    }

    # Ensure smiles and ik14 exist in bundle
    df_first = df_clean.drop_duplicates("molecule_id").set_index("molecule_id")
    mol_ids = bundle["mol_ids"]

    if "smiles" not in bundle or bundle["smiles"] is None:
        if "smiles" in df_first.columns:
            bundle["smiles"] = df_first.loc[mol_ids, "smiles"].fillna("").astype(str).values
        else:
            bundle["smiles"] = np.array([""] * len(mol_ids))

    if "ik14" not in bundle or bundle["ik14"] is None:
        if "ik14" in df_first.columns:
            bundle["ik14"] = df_first.loc[mol_ids, "ik14"].fillna("").astype(str).values
        else:
            bundle["ik14"] = np.array([str(m) for m in mol_ids])

    return bundle


def save_library(
    library_bundle: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save all library artifacts to disk conforming to the Stage 03 artifact contract."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sparse matrices
    scipy.sparse.save_npz(
        os.path.join(output_dir, "lib_sp_coarse.npz"),
        library_bundle["sp_coarse"],
    )
    scipy.sparse.save_npz(
        os.path.join(output_dir, "lib_sp_fine.npz"),
        library_bundle["sp_fine"],
    )

    # 2. Dense index arrays
    idx_kwargs = {
        "mol_ids": np.array(library_bundle["mol_ids"]),
        "neutral_mass": library_bundle["neutral_mass"],
        "peak_mzs": library_bundle["peak_mzs"],
        "peak_ints": library_bundle["peak_ints"],
        "peak_mask": library_bundle["peak_mask"],
        "prec_mass": library_bundle["prec_mass"],
    }
    if "smiles" in library_bundle and library_bundle["smiles"] is not None:
        idx_kwargs["smiles"] = np.array(library_bundle["smiles"])
    if "ik14" in library_bundle and library_bundle["ik14"] is not None:
        idx_kwargs["ik14"] = np.array(library_bundle["ik14"])

    np.savez_compressed(os.path.join(output_dir, "lib_index.npz"), **idx_kwargs)

    # 3. Configuration
    cfg_data = library_bundle.get("config", {"best_method": "cos_01", "source_tag": "lib_cos"})
    with open(os.path.join(output_dir, "lib_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)


# ── MAIN INFERENCE ENTRY POINT ───────────────────────────────────────────────
def retrieve(
    queries_df: pd.DataFrame,
    library_bundle: Dict[str, Any],
    cfg: Optional[LibRetrievalConfig] = None,
    top_k: int = 100,
) -> pd.DataFrame:
    """Run spectral retrieval for arbitrary query dataframes against a library bundle.

    Executes the optimal metric recorded in library_bundle["config"]["best_method"]
    ('cos_01', 'cos_001', 'mod_cos', 'entropy') or falls back to coarse cosine.

    Returns:
      pd.DataFrame with schema contract:
        ["molecule_id", "smiles", "ik14", "score", "source"]
    """
    if cfg is None:
        cfg = LibRetrievalConfig()
    cfg.TOP_K = top_k

    cols = ["molecule_id", "smiles", "ik14", "score", "source"]
    if len(queries_df) == 0 or len(library_bundle["mol_ids"]) == 0:
        return pd.DataFrame(columns=cols)

    if cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_bundle = build_query_vectors(queries_df, cfg=cfg)

    best_method = library_bundle.get("config", {}).get("best_method", "cos_01")
    src_tag = library_bundle.get("config", {}).get(
        "source_tag",
        {"cos_01": "lib_cos", "cos_001": "lib_cos", "mod_cos": "lib_mod", "entropy": "lib_ent"}.get(best_method, "lib_cos"),
    )

    # Dispatch to appropriate similarity kernel
    if best_method == "cos_01":
        scores, indices = gpu_cosine_topk(
            query_bundle["sp_coarse"],
            library_bundle["sp_coarse"],
            top_k=cfg.TOP_K,
            q_chunk=cfg.COS_QUERY_CHUNK,
            l_chunk=cfg.COS_LIB_CHUNK,
            device=device,
            desc="cos_01",
        )
    elif best_method == "cos_001":
        scores, indices = gpu_cosine_topk(
            query_bundle["sp_fine"],
            library_bundle["sp_fine"],
            top_k=cfg.TOP_K,
            q_chunk=min(cfg.COS_QUERY_CHUNK, 2000),
            l_chunk=min(cfg.COS_LIB_CHUNK, 5000),
            device=device,
            desc="cos_001",
        )
    elif best_method == "mod_cos":
        scores, indices = _run_modified_cosine_retrieval(
            query_bundle, library_bundle, cfg, device
        )
    elif best_method == "entropy":
        scores, indices = entropy_sim_topk(
            query_bundle["sp_coarse"],
            library_bundle["sp_coarse"],
            query_masses=query_bundle["neutral_mass"],
            lib_masses=library_bundle["neutral_mass"],
            mass_window=cfg.MASS_WINDOW,
            top_k=cfg.TOP_K,
            device=device,
            desc="entropy",
        )
    else:
        # Default fallback to coarse cosine
        scores, indices = gpu_cosine_topk(
            query_bundle["sp_coarse"],
            library_bundle["sp_coarse"],
            top_k=cfg.TOP_K,
            q_chunk=cfg.COS_QUERY_CHUNK,
            l_chunk=cfg.COS_LIB_CHUNK,
            device=device,
            desc="fallback_cos",
        )

    # Build candidate DataFrame conforming to competition contract
    rows: List[Dict[str, Any]] = []
    query_ids = query_bundle["mol_ids"]
    lib_smiles = library_bundle.get("smiles")
    lib_ik14 = library_bundle.get("ik14")
    lib_ids = library_bundle["mol_ids"]

    for qi, qid in enumerate(query_ids):
        for rank in range(scores.shape[1]):
            sc = float(scores[qi, rank])
            li = int(indices[qi, rank])
            if li < 0 or sc < 0:
                break
            smi = str(lib_smiles[li]) if (lib_smiles is not None and li < len(lib_smiles)) else ""
            ik = str(lib_ik14[li]) if (lib_ik14 is not None and li < len(lib_ik14)) else (str(lib_ids[li]) if li < len(lib_ids) else "")
            rows.append({
                "molecule_id": qid,
                "smiles": smi,
                "ik14": ik,
                "score": sc,
                "source": src_tag,
            })

    return pd.DataFrame(rows, columns=cols)
