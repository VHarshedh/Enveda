# Modeling & Pipeline Recommendations

### 1. Spectral Representation & Peak Capping
* **Recommended Peak Cap (K)**: **`300`** peaks per spectrum.
  * Captures >95% of spectral peaks (Train 95th percentile = `628`, 99th percentile = `1347`).
  * Keeping K <= 300 bounds GPU memory while preserving key diagnostic fragments.
* **m/z Bin Range**: `[0, 2430383]` Da.
* **Bin Resolution**:
  * Unit/Low resolution: `0.1 Da` (~24303830 bins).
  * High resolution / Transformer models: `0.01 Da` or direct continuous embedding (m/z, sqrt(intensity)).
* **Intensity Preprocessing**: Intensities are normalized to base peak = 1.0. Square-root transform is highly recommended to upweight informative low-abundance fragments.

### 2. Formula & Chemical Search Space Constraints
* **Maximum Molecular Weight**: **`1013` Da** (99th percentile train MW = `912.5` Da).
* **Adduct Mass Conversion**: `69.48%` of train spectra match SMILES exact mass within 10 ppm using the standard adduct table.
* **Element Universe**: Restrict formula generation to `['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'H']`.

### 3. Validation & Splitting Strategy
* **Strict Molecule Disjointness**: Multiple spectra exist per molecule. Cross-validation MUST group by `molecule_id` or `ik14`.
* **Scaffold-Disjoint Split**: Recommended `GroupKFold` on Murcko Scaffold (`127238` unique scaffolds). This accurately mimics novel scaffold generalization required on the test set.

### 4. Baseline & Retrieval Ceiling
* **Mass Overlap**: `100.00%` of test molecules share a neutral mass within +/-5.0 ppm of a training molecule.
* Pure library mass retrieval has an upper bound MRR@25 of `100.00%`. Structure generation / retrieval over a massive external candidate library is essential.

### 5. Candidate Database & Retrieval Knobs (Stage 04)
* **Precursor Mass Window (`CFG.FP_WINDOW_PPM = 10.0`)**:
  * High-resolution MS/MS (Orbitrap / Q-TOF) instruments provide mass accuracy well within 5–10 ppm.
  * Retaining 10.0 ppm for the primary build bounds candidate pools while achieving complete test mass coverage. Expanding to 20 ppm swells candidate pools by ~2–3× without boosting true candidate recall.
* **Validation Provenance Isolation (`CFG.EXCLUSION_SCOPE = "train"`)**:
  * Accurately simulates competition test behavior. The hidden test set contains genuine natural products that exist in external repositories (COCONUT/LOTUS) without matching training spectra.
  * Clearing only the training provenance bit (`source_mask & ~1`) allows external natural product hits to be retrieved during validation, avoiding artificial penalization while preventing train-split label leakage.
* **Tautomer Canonicalization Guard (`ORACLE_CANON_TOP_N = 256`)**:
  * Uniform `fast_key` bulk deduplication keeps candidate database construction fast (<3 hours) and RAM-bounded (<7.5 GB). Exact tautomer canonicalization via `evalkit.canon_ik14` is deferred to `finalize_candidates()` on the top-256 ranked candidates per query.