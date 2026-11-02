# External Validation & Testing Report: Processed_output2

**Dataset Target:** `data/processed/Processed_output2`  
**Evaluation Date:** 2026-11-01  
**Modality:** 2D Axial T1-Weighted MRI Slices (224 × 224 grayscale)  
**Total Images:** 14,760  

---

## 1. Executive Summary

This report documents the rigorous external evaluation performed on the image cohort located in `data/processed/Processed_output2`. This dataset represents an external multi-class cross-sectional neuroimaging slice bank spanning three clinical stages of cognitive impairment and dementia:
1. **Cognitively Normal (CN):** 4,528 slices (30.68%)
2. **Mild Cognitive Impairment (MCI):** 7,165 slices (48.54%)
3. **Alzheimer's Disease (AD):** 3,067 slices (20.78%)

The dataset was evaluated across two key validation streams:
- **Biophysical & Morphometric Phenotyping:** Brain parenchymal area fraction, compactness, tissue intensity, and edge energy.
- **Deep Representation & Clinical Diagnostic Benchmarks:** Deep feature extraction via pretrained ResNet-18 encoders, followed by stratified 5-fold cross-validated linear probing, non-linear ensemble classification (Random Forest), and binary discriminative benchmarks (CN vs. AD, CN vs. MCI).

---

## 2. Cohort Composition & Data Summary

| Cohort / Diagnosis | Class Subfolder | Number of Slices | Proportion (%) | Dimensions | Channels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cognitively Normal (CN)** | `CN_Images1/` | 4,528 | 30.68% | 224 × 224 | 1 (Grayscale) |
| **Mild Cognitive Impairment (MCI)** | `MCI_Images1/` | 7,165 | 48.54% | 224 × 224 | 1 (Grayscale) |
| **Alzheimer's Disease (AD)** | `AD_Images1/` | 3,067 | 20.78% | 224 × 224 | 1 (Grayscale) |
| **Total** | — | **14,760** | **100.00%** | **224 × 224** | **1** |

---

## 3. Morphometric Phenotype Analysis

To characterize whether 2D slices exhibit global macro-morphometric shifts across stages, we quantified:
- **Brain Parenchyma Area Fraction ($\text{fg\_ratio}$):** Foreground pixels ($\text{intensity} > 25$) normalized by total field of view ($224 \times 224$).
- **Normalized Tissue Intensity:** Mean pixel intensity of segmented parenchyma.
- **Sulcal Edge / Boundary Energy:** Average absolute difference gradient along $x$ and $y$ axes (quantifying cortical fold boundary sharpness and ventricular enlargement).

### Morphometric Summary Table

| Metric | Cognitively Normal (CN) | Mild Cognitive Impairment (MCI) | Alzheimer's Disease (AD) |
| :--- | :--- | :--- | :--- |
| **Parenchyma Area Fraction** | $0.2028 \pm 0.0576$ | $0.1996 \pm 0.0632$ | $0.2029 \pm 0.0619$ |
| **Tissue Intensity (Mean ± SD)** | $148.73 \pm 11.23$ | $147.50 \pm 12.04$ | $147.15 \pm 11.89$ |
| **Edge Energy (Sulcal Sharpness)** | $6.78 \pm 1.67$ | $6.70 \pm 1.79$ | $6.75 \pm 1.78$ |
| **Tissue Compactness** | $0.0603 \pm 0.015$ | $0.0898 \pm 0.021$ | $0.0617 \pm 0.018$ |

### Two-Sample Statistical Significance ($t$-tests)
- **CN vs. AD Parenchyma Fraction:** $t = -0.023$, $p = 0.981$ (Not statistically distinct on isolated 2D slice level without slice z-depth conditioning).
- **CN vs. MCI Parenchyma Fraction:** $t = 0.845$, $p = 0.399$
- **MCI vs. AD Parenchyma Fraction:** $t = -0.838$, $p = 0.402$

> **Clinical Finding:** Because individual 2D axial slices originate from varied anatomical heights (ranging from ventral cerebellar slices to mid-ventricular hippocampal slices to high-convexity vertex slices), raw 2D parenchymal area alone contains high slice-selection variance. Full diagnostic discriminability requires localized structural representation.

---

## 4. Diagnostic Classification Benchmarks

A stratified sample of 1,500 slices (500 slices per class) was embedded using deep neuroimaging representations (ResNet-18 feature manifold, $\mathbb{R}^{512}$). Models were evaluated with 5-fold stratified cross-validation.

### Performance Summary Table

| Task | Classifier Model | Accuracy (%) | Macro F1-Score | AUROC (OVR) |
| :--- | :--- | :--- | :--- | :--- |
| **3-Class (CN vs. MCI vs. AD)** | Linear Probe (L2 LogReg) | **41.93% ± 1.81%** | **0.4181 ± 0.019** | **0.5960 ± 0.013** |
| **3-Class (CN vs. MCI vs. AD)** | Random Forest (100 Trees) | **44.33% ± 2.45%** | **0.4428 ± 0.025** | **0.6305 ± 0.017** |
| **Binary: CN vs. AD** | Deep Feature Classifier | **60.60%** | **0.6004** | **0.6510** |
| **Binary: CN vs. MCI** | Deep Feature Classifier | **56.00%** | **0.5591** | **0.5849** |

*(Note: In 3-class classification, random chance baseline is 33.33%. The deep representation significantly outperforms random chance across all folds).*

---

## 5. Detailed Classification Metrics (Linear Probe)

### Confusion Matrix

| True \ Predicted | Predicted CN | Predicted MCI | Predicted AD | Class Total |
| :--- | :---: | :---: | :---: | :---: |
| **True CN** | **207** | 158 | 135 | 500 |
| **True MCI** | 155 | **196** | 149 | 500 |
| **True AD** | 126 | 148 | **226** | 500 |

### Per-Class Diagnostic Performance

| Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cognitively Normal (CN)** | 0.4242 | 0.4140 | 0.4190 | 500 |
| **Mild Cognitive Impairment (MCI)** | 0.3904 | 0.3920 | 0.3912 | 500 |
| **Alzheimer's Disease (AD)** | 0.4431 | 0.4520 | 0.4475 | 500 |
| **Macro Average** | **0.4193** | **0.4193** | **0.4193** | **1,500** |
| **Weighted Average** | **0.4193** | **0.4193** | **0.4193** | **1,500** |

---

## 6. Key Scientific Takeaways & Integration with Longitudinal Twin

1. **Slice-level vs. Volumetric Graph Modeling:**
   - Isolated 2D slices provide instantaneous cross-sectional classification capability (up to **0.651 AUROC** for CN vs. AD without longitudinal context or volume metadata).
   - In contrast, the primary OASIS-2 pipeline leverages 3D multi-session volumetric SynthSeg segmentations (56 Desikan-Killiany parcels) coupled with structural connectome diffusion (NDM) and ST-GNN-ODE, which achieves **0.9908 Pearson $r$** and **78.38% CDR progression accuracy**.
2. **Transition Boundary (MCI):**
   - MCI remains the most challenging transitional boundary (precision $39.0\%$), exhibiting overlapping morphometric characteristics between late-stage normal aging and early-stage AD neurodegeneration.
3. **Artifacts & Data Outputs Generated:**
   - Detailed JSON metrics: `outputs/external_test/external_testing_results.json`
   - Tabular summary: `outputs/external_test/external_testing_summary.csv`
   - Morphometrics summary: `outputs/external_test/morphometric_summary.csv`
   - Evaluation runner script: `scripts/evaluate_external_dataset.py`

