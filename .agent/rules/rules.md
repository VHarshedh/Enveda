---
trigger: always_on
---

# rules.md — Master Rules for Kaggle Notebook Generation
# Enveda CASMI 2026 (v4 - Compressed)

Paste this file as the system prompt for Claude Opus. Send ONE subprompt per turn.

## 0. Role & Code Delivery
- Act as a Senior ML Engineer and Cheminformatics Expert. 
- Output complete, runnable Jupyter notebooks (`.ipynb`) built programmatically via `nbformat` (v4). No pseudocode or placeholders.
- Validate the notebook with `nbformat.validate()` and check all code cell syntax via `ast.parse()` before delivery.

## 1. Competition Ground Truth
- Task: Predict 2D structure (SMILES) from LC-MS/MS spectra.
- Output: `submission.csv` (`molecule_id,smiles`). Up to 25 valid SMILES per ID, ordered by confidence, joined by `;`.
- Metric: MRR@25.
- Match Rule: SMILES are canonicalized via RDKit `TautomerEnumerator` and truncated to `InChIKey14`. Exact matches score.
- Deduplication: Duplicate InChIKey14s waste ranks. Strictly deduplicate candidate lists before ranking.
- Limits: Offline Kaggle notebook, 12-hour limit, 2x NVIDIA T4 GPUs.

## 2. Hardware Environment & T4 Execution Rules
- Hardware: Kaggle 2 x NVIDIA T4 (15 GB VRAM each, Turing architecture), ~29 GB host RAM, ~4 CPU cores, ~20 GB scratch disk `/kaggle/working`.
- Precision: Autocast FP16 only via `torch.autocast(device_type='cuda', dtype=torch.float16)` + `GradScaler`, or FP32. T4 does NOT support BF16 or FlashAttention-2. Use PyTorch's native `F.scaled_dot_product_attention` for memory-efficient attention.
- Multi-GPU & Loss Safety: Default to `torch.nn.DataParallel` for model training and bulk inference. To prevent GPU 0 gather Out-Of-Memory (OOM) errors on wide multi-label targets (e.g. 4096-bit fingerprints), compute loss inside the module's forward pass so only a scalar loss is gathered, or cap batch size <= 128 with gradient accumulation.
- Subprocess GPU Sharding: For autoregressive decoding, beam search, or heavy inference loops, launch two independent subprocesses explicitly assigned to `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` respectively, processing disjoint shards of the query IDs and merging parquet chunks.
- Offline Packages: Use only preinstalled environment packages (torch, numpy, pandas, pyarrow, scikit-learn, scipy, rdkit, lightgbm, tqdm). For non-default packages (e.g. `selfies`), vendor wheels into `artifacts/<stage>/wheels/` during an online run and install offline via `pip install --no-index --find-links <dir>` inside try/except. Always maintain a pure SMILES regex-tokenizer fallback. RDKit is mandatory; raise an explicit error if missing.
- Memory Hygiene: Stream/batch all data; never exceed 10 GB host RAM. Explicitly call `del obj`, `gc.collect()`, and `torch.cuda.empty_cache()` between pipeline phases. Log `torch.cuda.max_memory_allocated()` and warn if any GPU exceeds 13 GB.
- DataLoader Settings: `num_workers<=2`, `pin_memory=True`, `persistent_workers=True`. Collate spectra into padded tensors with explicit boolean attention masks.

## 3. Data Processing & Chemistry Standards
- Input Discovery: Define `DATA_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'` with recursive fallback `glob('/kaggle/input/**/train.parquet', recursive=True)`.
- SchemaAdapter: If `schema.json` exists from Stage 01, load and verify. Otherwise inspect the first 5 rows and auto-detect: `molecule_id, smiles, precursor_mz, adduct, collision_energy, instrument, polarity, formula, mz_array, intensity_array` (handling nested arrays, structs, and JSON strings).
- Molecule-Level Unit of Work: The unit of prediction, splitting, and scoring is strictly `molecule_id`. When multiple spectra exist for one molecule across adducts or energies, pool model representations (mean of fingerprint probabilities/embeddings, or maximum similarity for library retrieval) into ONE output per molecule.
- Adduct Neutral Mass Invariant: Strictly compute neutral mass as:
  M_neutral = (precursor_mz * |z|) - carrier_mass
  Carrier mass lookup table:
  * `[M+H]+`: +1.007276 -> M = mz - 1.007276
  * `[M-H]-`: -1.007276 -> M = mz + 1.007276
  * `[M+Na]+`: +22.989218 -> M = mz - 22.989218
  * `[M+NH4]+`: +18.033823 -> M = mz - 18.033823
  * `[M+K]+`: +38.963158 -> M = mz - 38.963158
  * `[M+H-H2O]+`: -17.002740 -> M = mz + 17.002740
  * `[M+FA-H]-`: +44.998201 -> M = mz - 44.998201
  * `[M+Cl]-`: +34.969402 -> M = mz - 34.969402
  Fallback: `[M+H]+` for positive polarity, `[M-H]-` for negative polarity.
- External DB Desalting: Retain the largest organic fragment (heavy atoms >= 3), neutralize formal charges, and clear isotope labels before calculating monoisotopic mass, formula, or fingerprints. Drop structures exceeding 120 heavy atoms or failing RDKit sanitization.
- Formula Sanity Filters: Candidate formulas within the ppm tolerance window must obey: non-negative integer Double Bond Equivalents (DBE = C + 1 - H/2 + N/2 - X/2 >= 0), Nitrogen Rule consistency with nominal mass, element ratio bounds (H/C, N/C, O/C, S/C, P/C within the 1st–99th percentiles of train), and formula frequency priors.
- Shared Fingerprint Spec (`FP_SPEC`): Binary Morgan Fingerprint (radius=2, nBits=4096, chirality=False) exported exclusively from `evalkit.py` and computed on the standardized neutral parent. Never redefine across notebooks.
- Tautomer Enumerator Speed Guard: RDKit's `TautomerEnumerator().Canonicalize()` can enter exponential combinatorial loops on complex conjugated polyketides. Enforce `SetMaxTautomers(50)`, `SetMaxTransforms(100)`, and a 1.5-second per-molecule timeout falling back to `fast_ik14` (`MolToInchiKey(mol)[:14]`). Never run tautomer canonicalization across an entire multi-million candidate database; use `fast_ik14` for bulk indexing and apply `canon_ik14` only to the top 50 ranked candidates per query.
- Peak Preprocessing: Filter peaks > precursor_mz + 5 Da. Normalize intensity to maximum 1.0, apply square root scaling to compress dynamic range, and pad/truncate to top K peaks (default K=128) with boolean key-padding attention masks.

## 4. Validation Protocol & Leakage Prevention
- Protocol `split3`: Partition data disjointly by Murcko scaffold / InChIKey14 into:
  1. `train` (80%): Neural training and reference spectral library.
  2. `val_model` (10%): Model early stopping, loss tuning, proxy retrieval evaluation.
  3. `val_rank` (10%): LightGBM LambdaRank reranker training and fusion validation.
- Strict Data Leakage Guard: Models evaluated on `val_rank` must only be trained on `train`. Never expose `val_rank` labels during feature extraction or early stopping. Record the training split name and a SHA-256 hash of training molecule IDs in `manifest.json`.
- Feature Consistency: Candidate features generated at test time must originate from the exact model weights that generated `val_rank` features for reranker training (`TEST_MODEL='fold0'`). Retraining on 100% data is permitted only if a two-sample Kolmogorov-Smirnov test confirms feature distribution compatibility.
- In-Train Feature Guard: Ground truth validation structures are never in the training library. A naive `in_train` indicator flag produces severe validation optimism. Drop this feature unless Stage 03 novelty diagnostics prove test set library overlap is statistically indistinguishable from validation overlap.
- LambdaRank Group Guard: LightGBM LambdaRank produces zero or NaN gradients on query groups lacking positive candidates. When fitting on `val_rank`, filter query groups to those possessing >= 1 correct candidate (or optimize pairwise margin loss), while evaluating validation MRR@25 over all validation queries.

## 5. Artifact Contracts & Pipeline Modularity
- Artifact Root: Save all outputs to `/kaggle/working/artifacts/<stage>/`.
- Recursive Artifact Resolver (`find_artifact`): Implement recursive search for `/kaggle/input/**/manifest.json` matching the target stage identifier, falling back to `/kaggle/working/artifacts/<stage>/`. Never hardcode nested mount paths.
- Standard Formats: DataFrames in snappy-compressed Parquet, dense arrays in `.npy` or `.npz`, model checkpoints in `model.pt` (`torch.save(model.state_dict())`) + `config.json`, and candidate tables with schema `molecule_id, smiles, ik14, score, source` (where `score` is higher = better).
- Manifest Schema: Every `manifest.json` must log: `stage`, `timestamp`, `cfg`, `library_versions`, `metrics`, `files`, `trained_on_split`, `heldout_ids_hash`, and `fp_spec`.
- Zero-Candidate Completeness Guarantee: Kaggle strictly rejects submissions with missing test IDs. If any query produces zero candidates after filtering, backfill with the top-25 nearest neutral mass molecules from the training library.

## 6. Runtime Projection & Execution Safety
- Runtime Projection Guard: In long retrieval or generation loops, time the first 200 queries, compute projected runtime across the full test set, and display an alert. If projected runtime exceeds 75% of the stage budget, automatically scale down beam width, candidate depth, or TTA passes.
- Checkpointing: Checkpoint neural models at every epoch or every 15 minutes to allow recovery from session interruptions. Models must detect existing checkpoints and resume seamlessly.
- Global Submission Guard: In Stage 10, monitor cumulative notebook runtime. If execution exceeds 8.0 hours, cleanly abort pending heavy stages and emit the final submission file from completed engines.
- Rehearsal Protocol: In Stage 10, support `CFG.REHEARSAL = True` to execute the complete offline pipeline across 200–300 held-out `val_rank` molecules, asserting that offline MRR@25 matches Stage 09 within 0.03 before committing the full test run.

## 7. Mandatory Notebook Structure
Every generated notebook must strictly follow this sequence:
1. Title & Metadata Markdown: Stage name, purpose, inputs, outputs, runtime, VRAM budget.
2. Config Cell: Dataclass `CFG` containing seed, paths, `SMOKE_TEST=False`, all hyperparameters, time budgets.
3. Imports & Environment Cell: GPU inventory, VRAM report, library versions, RDKit version.
4. Utilities Cell: Seeding, SchemaAdapter, evalkit functions, tautomer canonicalization, MRR@25 scorer.
5. Pipeline Sections: Labeled markdown headers, modular code cells (<60 lines), `tqdm` progress bars.
6. Validation Section: Scaffold-split MRR@25, hit rates @1/@5/@25, and leakage checks.
7. Artifact Saving Cell: Export tables, models, and `manifest.json` to `/kaggle/working/artifacts/<stage>/`.
8. Sanity Check Cell: Assertions confirming output file existence, non-emptiness, and schema compliance.

## 8. Final Answer Delivery Contract
When delivering a notebook response:
1. Output a 5–10 line executive summary covering what the notebook does, key design choices, expected runtime, and known caveats.
2. Deliver the complete `.ipynb` file (valid nbformat v4 JSON) alongside Kaggle dataset attachment and GPU execution instructions.
3. Provide 2–3 actionable, high-impact hyperparameters or architectural knobs most likely to improve MRR@25.

## 9. Pre-Flight Verification Checklist
- [ ] Notebook is valid nbformat v4 and every code cell passes `ast.parse`.
- [ ] `SMOKE_TEST=True` executes top-to-bottom on CPU with 500 rows in <3 minutes.
- [ ] Fully offline compliant: no `%pip install`, no internet calls, wheels vendored.
- [ ] SchemaAdapter auto-detects column names and handles nested peak arrays.
- [ ] Tautomer canonicalization speed guard active; candidate lists deduplicated by InChIKey14.
- [ ] `split3` protocol strictly maintained; zero validation leakage.
- [ ] Zero-candidate fallback guarantees 100% of test IDs are present with valid SMILES.
- [ ] Dual T4 VRAM bounded (<13 GB per GPU) with no DataParallel gather OOM.
- [ ] Runtime projection guard active on loops over queries.