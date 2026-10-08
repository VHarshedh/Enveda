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