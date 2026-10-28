# Root-Cause Diagnosis of Your Interrupted 11-Hour Run

From your execution logs in `07_spec2smiles_train`:
1. **Training Convergence:**
   - **Epoch 1–8:** `val_loss` dropped steadily from `0.8553` $\to$ `0.7923` $\to$ `0.7722` $\to$ `0.7619` $\to$ `0.7563` $\to$ `0.7533` $\to$ `0.7523` $\to$ **`0.7513` (Best at Epoch 8)**.
   - **Epoch 9:** `val_loss` ticked up slightly to `0.7515`, triggering the time guard at **8.30 hours (`29,869s`)** and saving the best Epoch 8 EMA checkpoint to `artifacts/spec2smiles/spec2smiles_model.pt`.
   - **Verdict on Weights:** Your saved `spec2smiles_model.pt` checkpoint has already converged on teacher-forced cross-entropy (`0.7513`). However, a **short 2-epoch sequence-level / low-LR canonical fine-tuning pass (~1.5 hours)** with reduced label smoothing (`0.01` instead of `0.05`) and lower augmentation rate (`0.25` instead of `0.65`) will sharpen the top-1 token probabilities specifically for beam search!

2. **Why Candidate Generation (`Cell 41`) Timed Out After 3+ Hours:**
   Look at the 4 massive bottlenecks inside `generate_candidates` and `beam_search` in your interrupted run:
   - **Bottleneck A (Single-GPU Inference):** `beam_search` and `model.encode` in `generate_candidates` unwrapped `model = model.module` and ran on **only 1 of your 2 T4 GPUs (`cuda:0`)**, leaving the second T4 GPU 100% idle for 3.5 hours!
   - **Bottleneck B (14,000,000 Redundant RDKit Calls in Python Loop):** For 55,562 queries $\times$ up to 256 archive paths, the CPU loop ran `Chem.MolFromSmiles(smi)` + `evalkit.standardize_mol` + `CalcExactMolWt` + `compute_morgan_fp` + `compute_maccs_fp` on **every single raw beam** before checking if the decoded SMILES string was already seen or if its mass was even remotely close!
   - **Bottleneck C (Mass Pre-Filter Missing Before Expensive Fingerprints):** Computing 4096-bit Morgan + 167-bit MACCS fingerprints + `AddHs` elemental counts on 150+ invalid-mass candidates per molecule took ~85% of the CPU time, when a 0.01ms exact-mass check (`CalcExactMolWt`) can immediately skip full fingerprint generation for candidates with `ppm > 100` (or only compute fingerprints on the top-48 mass-filtered unique structures per query).
   - **Bottleneck D (Unnecessary 45-Minute `train.bin` Rebuild if Fine-Tuning):** Cell 38 spent **45 minutes (`2,688s`)** building `train.bin` across all 2.03M spectra. For a 2-epoch sharpening fine-tune, sampling 1–2 spectra per unique training molecule (~350k spectra instead of 2.03M replicate spectra) builds `train.bin` in **6 minutes** and runs each epoch in **8 minutes instead of 51 minutes**, while eliminating replicate-spectrum bias!

---

# Copy-Paste Prompt for Gemini 3.8 Flash High in AGY IDE

Copy the block below and paste it directly into your IDE chat with `07_spec2smiles_train.ipynb` attached:

---

```markdown
We are working on Stage 07 (`07_spec2smiles_train.ipynb`) of the Enveda CASMI 2026 Kaggle competition on dual NVIDIA T4 GPUs (`GPUs: 2`, 12-hour session limit).

### Context from Previous 11-Hour Run:
In our previous run, training ran for 9 epochs (8.3 hours), reaching its best validation loss at Epoch 8 (`val_loss = 0.75130`, saved in `spec2smiles_model.pt` and `smiles_vocab.json`, now attached as an input dataset alongside Stage 05b). However, Cell 41 (`generate_candidates` across `val_model` [27,581], `val_rank` [27,581], and `test` [400] = 55,562 molecules) did not finish within the remaining 3.5 hours of the 12-hour Kaggle session.

### Objective:
Update `07_spec2smiles_train.ipynb` so that it:
1. **Warm-Resumes from Attached `spec2smiles_model.pt`** (falling back to Stage 05b warm-start only if `spec2smiles_model.pt` is absent) and runs a **fast, high-precision 2-epoch sharpening fine-tune (~45 mins total)** to maximize `MRR@25`.
2. **Accelerates `generate_candidates` and `beam_search` by ~8x–10x** so all 55,562 molecules (`val_model`, `val_rank`, `test`) complete in **under 45 minutes** on dual T4 GPUs while further improving `MRR@25`.

Please implement all 6 specific engineering & modeling improvements below across the notebook cells:

---

### 1. Warm-Resume & Fast Molecule-Balanced Fine-Tuning (`Cells 15, 36, 37, 38, 39`)
- **Auto-detect attached Stage 07 checkpoint (`Cell 36` & `Cell 39`):**
  - Search `/kaggle/input` for an existing `spec2smiles_model.pt` and `smiles_vocab.json` via `find_file(..., required=False)` (or `try/except FileNotFoundError`).
  - If `smiles_vocab.json` is found in `/kaggle/input`, load `tokenizer = s2s.SmilesTokenizer.load(...)` directly so the vocabulary indices match `spec2smiles_model.pt` 100%, and copy it to `OUT / 'smiles_vocab.json'`.
  - If `spec2smiles_model.pt` is found in `/kaggle/input`, load its `model_state_dict` into `model` after `model.warm_start(encoder_path)` (which still loads `spec2fp_pos_weights.npy` into `self.pw_shift`!).
- **Eliminate the 45-Minute `train.bin` Rebuild & Replicate-Spectrum Over-Weighting (`Cell 38`):**
  - In `train_clean.parquet`, some molecules have 50+ replicate spectra while others have 2, causing `train.bin` to take 45 minutes to write (2.03M rows) and 51 minutes/epoch to train.
  - When warm-resuming from `spec2smiles_model.pt` (`RESUME_STAGE07 = True`), cap training spectra to at most **3 spectra per `molecule_id`** (using a counter per `molecule_id` while streaming batches) and **2 spectra per `molecule_id`** for `val_model`. This reduces `train.bin` from 2.03M to ~550k balanced spectra, cutting `Cell 38` I/O time from 45 mins to ~8 mins and epoch time from 51 mins to **~13 mins/epoch**!
- **Sharpening Fine-Tuning Schedule (`Cells 18, 36, 39`):**
  - When `RESUME_STAGE07 = True`:
    - Run `epochs = 2` (or `0` if `SKIP_FINETUNE = False` is toggled, default `epochs = 2` with `TRAIN_BUDGET_S = 1.5 * 3600`).
    - Unfreeze both encoder and decoder immediately (`lr_enc = 1.5e-5`, `lr_dec = 6e-5`, cosine decay to `1e-6`).
    - Reduce `label_smoothing` in `run_epoch` (`Cell 18`) from `0.05` to `0.015` (pass `label_smoothing=cfg.get('label_smoothing', 0.02)`) and reduce SMILES augmentation probability in `SpectrumDataset.__getitem__` (`Cell 17`) from `0.65` to `cfg.get('aug_prob', 0.30)`. Lower label smoothing + higher canonical SMILES ratio sharpens the peak of the softmax distribution for autoregressive beam search.
    - Enable gradient flow through the frozen auxiliary heads in `Spec2SmilesModel.encode` (`Cell 15`) when training: keep `AUX_HEAD_NAMES` parameters frozen (`requires_grad_(False)`), but compute `counts`, `maccs_probs`, `heavy`, and `rings` outside `torch.no_grad()` (without `.detach()` on `cond_vec`) so gradients from `self.condition` flow back into `pooled` and refine the encoder representations.

---

### 2. Dual-GPU (`cuda:0` + `cuda:1`) Parallel Bucket Execution (`Cell 41`)
- Currently, `generate_candidates` unwraps `model = model.module` and runs strictly on `cuda:0`, leaving `cuda:1` 100% idle during the entire 55,562-molecule inference phase!
- **Fix (`Cell 20` & `Cell 41`):**
  - Wrap the decoder/encoder call or split each partition batch across both GPUs using `concurrent.futures.ThreadPoolExecutor(max_workers=2)` with two model replicas: `model_0 = copy.deepcopy(model).to('cuda:0').eval()` and `model_1 = copy.deepcopy(model).to('cuda:1').eval()` (when `torch.cuda.device_count() >= 2`).
  - In `Cell 41`, process pairs of bucket files `(part_0, part_1)` concurrently—`part_0` on `model_0` (`cuda:0`) and `part_1` on `model_1` (`cuda:1`)! Because PyTorch CUDA kernels and C++ RDKit functions release the GIL, running `cuda:0` and `cuda:1` on paired buckets cuts wall-clock inference time in half (**2.0x speedup**).

---

### 3. Optimize `beam_search` Tensor Bandwidth & Sequence Budget (`Cells 19 & 20`)
- Look at `beam_search` in `Cell 19`:
  - `max_len = 124` runs 123 sequential decoder forward passes even when 99% of molecules finish before token 88, because `done` requires **all** `k=16` beams to fall below the 256th archive slot (`archive_scores[:, -1]`, which is `-inf` whenever fewer than 256 EOS tokens have been emitted!).
  - **Bug in `done` condition (`Cell 20`):** Because `archive_size = 256`, a molecule with 16 beams can emit at most ~30–60 `<eos>` tokens across 96 steps. That means `archive_scores[:, -1]` (the 256th slot!) is **permanently `-inf`**, so `done = live_upper <= archive_scores[:, -1]` is **NEVER True**, forcing `beam_search` to run all 123 steps on 100% of batches!
- **Fixes in `beam_search` (`Cells 19 & 20`):**
  1. Set `beam_size = 12`, `max_len = 96`, `mol_batch_size = 96`, and `archive_size = 64` (since we only need `top_k = 25` valid candidates, keeping the top 64 finished `<eos>` paths per molecule is more than enough).
  2. Fix the early-stopping check in `Cell 20`: compare `live_upper = scores.max(1).values / ((step + 1) ** 0.7)` against the **`top_k`-th (25th) best finished score** (`archive_scores[:, min(top_k - 1, archive_size - 1)]`) once `step >= 20`, not `archive_scores[:, -1]`!
  3. Enable PyTorch SDPA Flash/MemEfficient attention in `model.decode` (`Cell 16`) by passing ` causal` as a boolean mask with `norm_first=True` (already present) and slicing `max_active_len` dynamically.

---

### 4. Eliminate 90% of CPU RDKit Overhead in `generate_candidates` (`Cells 22 & 23`)
- Currently, for every molecule, `generate_candidates` loops over all finished paths in `paths` and immediately runs:
  `Chem.MolFromSmiles` $\to$ `evalkit.standardize_mol` (expensive `LargestFragmentChooser` + `Uncharger` + `SanitizeMol`) $\to$ `fast_key` $\to$ `compute_morgan_fp` $\to$ `compute_maccs_fp` $\to$ `mol_elem_counts` (`AddHs`).
- **Fixes in `Cells 22 & 23`:**
  1. **Deduplicate raw SMILES strings first:** Keep a `seen_raw_smi = set()` per molecule; if `smi = tokenizer.decode(ids)` is already in `seen_raw_smi` (different beam paths often produce the exact same SMILES or pad tail), skip it before calling RDKit!
  2. **Fast Mass Pre-Filter Before Standardization & Fingerprints:**
     - Parse `raw_mol = Chem.MolFromSmiles(smi)`.
     - Check element symbols on `raw_mol` (`any(a.GetSymbol() not in ELEMENTS_ORDER ...)`).
     - Immediately compute `raw_wt = rdMolDescriptors.CalcExactMolWt(raw_mol)`. Only call `evalkit.standardize_mol(raw_mol)` if `Chem.GetFormalCharge(raw_mol) != 0` or `'.' in smi` (98% of generated SMILES are already neutral single-fragment molecules, saving jutaan of `LargestFragmentChooser` + `Uncharger` calls!).
  3. **Two-Stage Candidate Reranking (Pre-filter to Top-36 by `log_prob + mass_score` before computing Morgan/MACCS/Elemental counts):**
     - First, deduplicate valid molecules by `key = evalkit.fast_key(mol)` using `pre_score = log_prob + 0.40 * mass_score`.
     - Keep only the **top 36 unique `fast_key` candidates** per molecule.
     - Compute `compute_morgan_fp`, `compute_maccs_fp`, `mol_elem_counts`, and the full 8-term hybrid `score` (`log_prob + 2.0*nb_ll + 1.8*cos_sim + 1.8*tan_sim + 0.6*maccs_cos + 0.6*maccs_tan + 0.40*mass_score - 0.04*elem_dist`) **only on those top 36 unique candidates**!
     - Finally, run `evalkit.canon_ik14` only until `top_k = 25` unique canonical structures are collected (`if rank >= top_k: break`).

---

### 5. Incremental Checkpointing & Progress Logging During Beam Search (`Cell 41`)
- In `Cell 41`, print progress every 8 partitions (`print(f'[{name}] {b_idx}/{len(partitions[name])} partitions | elapsed={(time.time()-START)/60:.1f}m | mrr25={reciprocal/max(queries,1):.4f}', flush=True)`) so we have live visibility into `MRR@25` and speed.
- Run `'test'` **first** in the loop `for name in ('test', 'val_model', 'val_rank'):` and immediately write `submission_spec2smiles.csv` right after `'test'` finishes (within 30 seconds of starting Cell 41!) so the competition submission and test candidates are safely on disk before the 55,162 validation molecules even begin!
- Add a session time guard inside the partition loop of `Cell 41`: if `time.time() - START > 11.2 * 3600`, break cleanly out of remaining validation partitions, close the ParquetWriter, and still write `spec2smiles_val_metrics.json` and `manifest.json` (`Cell 42`) so the Kaggle session **never** times out without saving outputs!

---

### 6. Output Requirements
- Update the notebook cells in-place while preserving the clean module-writing structure (`/kaggle/working/spec2fp.py`, `/kaggle/working/spec2smiles.py`, `/kaggle/working/evalkit.py`).
- Ensure all schema columns (`COLUMNS = ['molecule_id', 'smiles', 'ik14', 'fast_key', 'score', 'log_prob', 'nb_ll', 'cos_sim', 'tan_sim', 'maccs_cos', 'maccs_tan', 'elem_dist', 'ppm_err', 'rank', 'source']`) and `manifest.json` contracts (`heldout_ids_hash`, `submission_validation`) remain 100% compatible with Stage 09.
```