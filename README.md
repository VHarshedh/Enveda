# Enveda CASMI 2026: Spectrum-to-SMILES Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/cheminformatics-RDKit-green.svg)](https://www.rdkit.org/)
[![PyTorch](https://img.shields.io/badge/deep%20learning-PyTorch-red.svg)](https://pytorch.org/)

End-to-end machine learning and cheminformatics pipeline for the **Enveda CASMI 2026 Molecule ID from Mass Spectra Competition**. The goal is to predict 2D molecular structures (SMILES) directly from LC-MS/MS spectra, evaluated on **MRR@25** under tautomer-canonical InChIKey14 deduplication.

---

## 📁 Repository Architecture

The project is structured into clear functional domains to separate reusable Python modules, notebook compilation pipelines, datasets, stage artifacts, and automated verification suites:

```text
Enveda/
├── src/                          # Core reusable library modules (casmi26 package)
│   ├── __init__.py               # Package exports & versioning
│   ├── evalkit.py                # Official competition scorer, MRR@25, InChIKey14 canonicalizer
│   ├── preprocess.py             # Precursor m/z, peak processing, adduct neutral-mass tables
│   ├── libretrieval.py           # Stage 03 spectral library retrieval & matching
│   ├── candidates.py             # Stage 04 million-compound DB indexer & query engine
│   ├── spec2fp.py                # Stage 05 Spectrum-to-Fingerprint Transformer model
│   ├── fprank.py                 # Stage 06 candidate ranking, coordinate grid tuning & fusion
│   └── spec2smiles.py            # Stage 07 autoregressive SMILES Transformer & beam search
│
├── notebooks/                    # Self-contained, runnable Kaggle stage notebooks (.ipynb)
│   ├── 01_eda_schema.ipynb       # Stage 01: EDA, schema discovery, distribution drift tests
│   ├── 02_eval_splits.ipynb      # Stage 02: Disjoint 3-way evaluation splits (train/val_model/val_rank)
│   ├── 03_library_retrieval.ipynb# Stage 03: Reference library spectral retrieval
│   ├── 04_candidate_db.ipynb     # Stage 04: Candidate database indexing & binary packing
│   ├── 05_spec2fp_train.ipynb    # Stage 05: Transformer training (spec -> 4096 Morgan + MACCS + elem)
│   ├── 05b_spec2fp_finetune.ipynb# Stage 05b: Warm-start fine-tuning & LLRD
│   ├── 06_fp_rank_infer.ipynb    # Stage 06: FP-based retrieval & candidate ranking
│   └── 07_spec2smiles_train.ipynb# Stage 07: Generative Transformer training & fast beam search
│
├── builders/                     # Programmatic notebook builders & AST validators
│   ├── __init__.py
│   ├── build_01_eda.py           # Compiles 01_eda_schema.ipynb
│   ├── build_02_eval_splits.py   # Compiles 02_eval_splits.ipynb
│   ├── build_03_library_retrieval.py
│   ├── build_04_candidate_db.py  # Compiles 04_candidate_db.ipynb
│   ├── build_05_spec2fp_train.py # Compiles 05_spec2fp_train.ipynb
│   ├── build_05b_spec2fp_finetune.py
│   ├── build_06_fp_rank_infer.py # Compiles 06_fp_rank_infer.ipynb
│   ├── build_07_spec2smiles_train.py # Compiles 07_spec2smiles_train.ipynb
│   ├── patch_05b_spec2fp.py      # Surgical AST patcher for Stage 05b
│   └── generate_eda_notebook.py
│
├── tests/                        # Automated unit, numerical, and contract test suites
│   ├── __init__.py
│   ├── conftest.py               # Path injection for src and builders
│   ├── test_fprank.py            # 33 numerical contract & ranking tests (pytest)
│   └── test_spec2smiles.py       # 9 Stage 07 tokenizer, causality & pipeline tests (unittest)
│
├── data/                         # Datasets & external databases
│   ├── raw/                      # Official competition data (train, test, sample_submission)
│   ├── samples/                  # Head and debug subsets (train_head100, test_clean_head200)
│   └── external/                 # External DB sources (LOTUS, NPAtlas, COCONUT, Metabolites)
│
├── artifacts/                    # Intermediate stage outputs & processed database indices
│   ├── eda/                      # Cleaned parquet data, schema, stats, and plots
│   ├── eval_splits/              # Held-out 3-way split parquet and evalkit contracts
│   ├── candidate_db/             # Mass-sorted parquet, packed bit-vectors, SQLite indices
│   ├── spec2fp/                  # Trained encoder weights, predictions, and config
│   ├── stage06/                  # Candidate FP feature tables and Stage 06 configs
│   └── stage07/                  # SMILES vocabularies, generator configs, and model weights
│
├── docs/                         # Technical reports, documentation, and reference figures
│   ├── figures/                  # Exploratory data analysis & split diagnostic plots
│   ├── Enveda.pdf                # Problem formulation & chemistry reference
│   ├── chat_history.pdf          # Implementation logs & discussion history
│   └── recommendations.md        # Modeling insights & algorithmic recommendations
│
├── archives/                     # Large compressed result packages & checkpoints (.zip)
├── pyproject.toml                # Standard packaging and test configuration
├── requirements.txt              # Production dependency specifications
└── .gitignore                    # Robust exclusion rules for datasets, binaries, and logs
```

---

## 🔬 Pipeline Overview

The competition pipeline is divided into disciplined, sequential stages adhering strictly to [rules.md](.agent/rules/rules.md):

| Stage | Name | Description | Key Artifacts |
| :--- | :--- | :--- | :--- |
| **01** | **EDA & Schema** | Inspect precursor m/z, adduct shifts, peak count distributions, and KS drift tests. | `train_clean.parquet`, `schema.json` |
| **02** | **Evaluation Splits** | Molecule-disjoint 3-way split (`train`, `val_model`, `val_rank`). Establishes competition `evalkit.py`. | `splits.parquet`, `evalkit.py` |
| **03** | **Library Retrieval** | Fast cosine and entropy similarity search against reference spectral libraries. | `cand_lib_*.parquet` |
| **04** | **Candidate Database** | Mass-indexed SQLite + memory-mapped 4096-bit Morgan fingerprints from COCONUT, LOTUS, and NPAtlas. | `db.parquet`, `fp_packed.npy`, `candidates.py` |
| **05** | **Spec2FP Encoder** | Transformer encoder predicting 4096 Morgan bits, 167 MACCS keys, and 14 element counts. | `spec2fp_model.pt`, `fp_pred_*.npy` |
| **06** | **FP Rank & Fusion** | Fast top-100 candidate retrieval, coordinate grid tuning on `val_model`, and zero-leakage ranking. | `cand_fp_*.parquet`, `fprank.py` |
| **07** | **Spec2SMILES Generative** | Autoregressive Transformer with dual T4 support, fast beam search, and Self-Spec2FP reranking. | `cand_gen_*.parquet`, `spec2smiles.py` |

---

## ⚡ Quickstart

### 1. Environment Setup

Install dependencies or install the package in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Running Automated Tests

Run the full contract test suite:
```bash
# Run Stage 06 candidate ranking & contract tests (33 tests)
pytest tests/test_fprank.py -v

# Run Stage 07 generative model & beam search tests (9 tests)
python -m unittest tests/test_spec2smiles.py
```

### 3. Rebuilding Kaggle Notebooks

All Jupyter notebooks are generated programmatically with per-cell AST validation and cell-length guards (<59 lines per code cell):
```bash
# Generate Stage 06 notebook
python builders/build_06_fp_rank_infer.py

# Generate Stage 07 notebook
python builders/build_07_spec2smiles_train.py
```
Compiled notebooks are written to `notebooks/<stage>.ipynb`.

---

## 🛡️ Competition Rules & Constraints

- **Evaluation Metric**: MRR@25 on `molecule_id`. Tautomer-canonicalized InChIKey14 deduplication prevents wasting submission slots.
- **Leakage Prevention**: Never read `val_rank` labels during model or hyperparameter tuning. Reranker models are strictly evaluated on identical upstream fold models.
- **Kaggle Environment**: Dual NVIDIA T4 GPUs (15 GB each), internet disabled, <= 9 hours total runtime. Self-contained notebooks embed required source modules via `%%writefile /kaggle/working/<module>.py`.
