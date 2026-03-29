# ECG Arrhythmia Classifier — FPGA Edge Device

> **Binary-first ECG preprocessing pipeline with fixed-point IIR filtering on FPGA and 1D ResNet-18 classification.**  
> Low-power edge deployment for real-time Normal vs Abnormal beat detection using the MIT-BIH Arrhythmia Database.

---

## Project Overview

This project implements a complete ECG arrhythmia classification system designed for low-power edge deployment on an FPGA. The system is split into three tightly coupled stages — MATLAB preprocessing, Verilog hardware filter, and deep learning classification — all sharing the same binary Q8.8 signal representation so that FPGA output can be verified integer-for-integer against the MATLAB reference.

```
Raw ECG (ADC)
     │
     ▼
┌─────────────────────────────┐
│  MATLAB  (Ground Truth)     │
│  • Q8.8 quantisation        │
│  • SOS Butterworth filter   │
│  • Beat extraction (252 sa) │
│  • Z-score normalisation    │
│  • Patient-wise splitting   │
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────────┐
│  Colab  │  │  Verilog / FPGA      │
│ ResNet  │  │  • SOS filter        │
│ 1D-18   │  │  • 4 biquad cascade  │
│ binary  │  │  • Q16.16 coeffs     │
│ classif.│  │  • Verified vs MATLAB│
└─────────┘  └──────────────────────┘
     │            │
     └─────┬──────┘
           ▼
    Normal / Abnormal
    classification output
```

---

## Dataset

| Property | Value |
|---|---|
| Source | MIT-BIH Arrhythmia Database (PhysioNet) |
| Records | 48 |
| Total beats | 112,536 |
| Beat length | 252 samples (0.7 s at 360 Hz) |
| Sampling rate | 360 Hz |
| Labels | Normal (0) = 75,003 beats (66.6%) · Abnormal (1) = 37,533 beats (33.4%) |
| Split strategy | Patient-wise — no patient appears in two sets |
| Train | 34 records · 74,843 beats |
| Val | 7 records · 18,101 beats |
| Test | 8 records · 19,592 beats |

---

## Repository Structure

```
ecg-fpga-classifier/
│
├── README.md                        ← this file
│
├── matlab/                          ← MATLAB preprocessing pipeline
│   ├── filter_design.m              ← Butterworth SOS design + Q16.16 coefficients
│   ├── ecg_pipeline_corrected.m     ← Binary-first full pipeline (replaces 3 original scripts)
│   ├── verify_binary_filter_v2.m    ← FPGA path verification (SNR = 42.5 dB, max error 0.9 LSB)
│   └── output/
│       ├── data/
│       │   ├── processed/
│       │   │   ├── filter_coeffs.mat          ← SOS + Q16.16 coefficients
│       │   │   ├── butterworth_fpga_coeffs.txt← Human-readable coefficient table
│       │   │   └── ecg_dataset.mat            ← Full processed dataset
│       │   └── splits/
│       │       ├── train.csv                  ← Float Z-scored beats (ResNet input)
│       │       ├── val.csv
│       │       ├── test.csv
│       │       ├── test_with_records.csv      ← Includes record IDs for per-patient analysis
│       │       ├── train_binary.csv           ← Q8.8 ASCII binary (Verilog $readmemb)
│       │       ├── val_binary.csv
│       │       ├── test_binary.csv
│       │       ├── train_int16.bin            ← Raw int16 little-endian binary
│       │       ├── val_int16.bin
│       │       └── test_int16.bin
│       └── testbench/
│           ├── ecg_raw_binary.csv             ← FPGA input (10k samples, Record 100)
│           └── ecg_filtered_ref.csv           ← FPGA expected output (MATLAB reference)
│
├── verilog/                         ← FPGA SOS filter implementation
│   ├── biquad_section.v             ← Single second-order IIR section (reusable)
│   ├── sos_filter.v                 ← Top-level: gain stage + 4 cascaded biquads
│   ├── sos_filter_tb.v              ← Testbench: loads MATLAB CSVs, integer comparison
│   └── README.md                    ← Vivado setup instructions
│
├── colab/                           ← Google Colab ResNet training
│   └── ECG_ResNet_Fixed_v2.ipynb   ← 1D ResNet-18, fixed overfitting (v2)
│
└── results/                         ← Simulation and training outputs (added after runs)
    ├── vivado/
    │   └── simulation_log.txt       ← Per-sample DUT vs MATLAB comparison
    └── colab/
        ├── training_curves.png
        ├── confusion_roc.png
        ├── per_record_accuracy.png
        └── sample_beats.png
```

---

## Filter Design

4th-order Butterworth bandpass filter — 0.5 Hz to 45 Hz at 360 Hz sampling rate.  
Implemented as 4 cascaded second-order sections (biquads) to avoid fixed-point overflow in direct form.

| Property | Value |
|---|---|
| Filter type | Butterworth bandpass |
| Order | 4th (8th-order polynomial → 4 biquads) |
| Passband | 0.5 – 45 Hz |
| Sampling rate | 360 Hz |
| Implementation | Cascaded SOS (Direct Form II) |
| Signal format | Q8.8 signed 16-bit |
| Coefficient format | Q16.16 signed 32-bit |
| Gain | 645 (= 0.009838 × 65536) |

### Q16.16 Coefficients (from `filter_design.m`)

```
Section 1:  B = [ 65536  -131072   65536]   A = [65536   -57134   14251]
Section 2:  B = [ 65536   131080   65544]   A = [65536   -73411   38064]
Section 3:  B = [ 65536   131064   65528]   A = [65536  -130003   64472]
Section 4:  B = [ 65536  -131072   65536]   A = [65536  -130637   65106]
```

### Verification Results

| Metric | Value |
|---|---|
| Max error (Path A vs Path B) | 0.9 LSB = 0.0037 mV |
| SNR | 42.5 dB |
| Verilog vs MATLAB mismatches | 0 (target — pending Vivado run) |
| FPGA resource estimate | ~21 DSP48 slices, <1% LUTs, <1% FFs |

---

## ResNet-18 1D Classifier

| Property | Value |
|---|---|
| Architecture | 1D ResNet-18 (Conv1D residual blocks) |
| Input | (batch, 1, 252) — Z-scored ECG beat |
| Output | (batch, 2) — logits for [Normal, Abnormal] |
| Parameters | 8,727,874 |
| Loss | CrossEntropyLoss (weighted + label smoothing 0.1) |
| Optimiser | Adam lr=1e-3, weight decay=1e-3 |
| Scheduler | CosineAnnealingLR |
| Early stopping | Patience = 10 epochs on val F1 |
| Dropout | 0.5 (v2 fix — was 0.3) |
| Class weights | Normal=1.0, Abnormal=1.97 (ratio-based) |
| Target accuracy | 95%+ on patient-wise test set |

### Training Status

| Run | Val F1 | Test Acc | Test F1 | AUC | Status |
|---|---|---|---|---|---|
| v1 (original) | 0.8628 (ep 2) | 76.63% | 0.6363 | 0.8238 | Overfit — best at epoch 2 |
| v2 (fixed) | — | — | — | — | Pending |

---

## Progress Tracker

| Stage | Status | Notes |
|---|---|---|
| MATLAB filter design | ✅ Complete | 4 SOS sections, Q16.16 coefficients |
| MATLAB pipeline | ✅ Complete | 112,536 beats, patient-wise splits |
| MATLAB verification | ✅ Complete | SNR 42.5 dB, max error 0.9 LSB |
| Verilog SOS filter | ⏳ Simulation pending | Code written, Vivado run needed |
| Verilog testbench | ⏳ Simulation pending | Awaiting PASS confirmation |
| ResNet v2 training | ⏳ Pending | ECG_ResNet_Fixed_v2.ipynb ready |
| R-peak detector | 🔲 Not started | After Verilog filter verified |
| Beat windowing | 🔲 Not started | After R-peak detector |
| Z-score normaliser | 🔲 Not started | After beat windowing |
| Full system integration | 🔲 Not started | Final stage |

---

## How to Run

### MATLAB (run in this order)

```matlab
% Step 1 — Design filter and save coefficients
run('matlab/filter_design.m')

% Step 2 — Process all 48 MIT-BIH records (takes ~10 min)
run('matlab/ecg_pipeline_corrected.m')

% Step 3 — Verify FPGA path matches MATLAB reference
run('matlab/verify_binary_filter_v2.m')
% Expected: PASS — max error <= 2 LSB, SNR > 40 dB
```

**Prerequisites:**
- WFDB Toolbox for MATLAB — [physionet.org/content/wfdb-matlab](https://physionet.org/content/wfdb-matlab)
- MIT-BIH Arrhythmia Database — [physionet.org/content/mitdb](https://physionet.org/content/mitdb)

### Verilog (Xilinx Vivado)

1. Create RTL project, add `biquad_section.v` and `sos_filter.v` as design sources
2. Add `sos_filter_tb.v` as simulation source
3. Copy `ecg_raw_binary.csv` and `ecg_filtered_ref.csv` into the Vivado project folder
4. Set `sos_filter_tb` as simulation top, run Behavioral Simulation
5. Expected console output: `PASS -- output matches MATLAB exactly`

### Google Colab (ResNet training)

1. Open `colab/ECG_ResNet_Fixed_v2.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run Cell 1 (imports), then Cell 2 (upload train.csv, val.csv, test.csv, test_with_records.csv)
4. Run all remaining cells in order
5. Expected: test accuracy ≥ 95%, training curves saved, weights downloaded

---

## Key Design Decisions

**Why SOS instead of direct-form filter?**  
The direct-form 8th-order polynomial has a maximum `a` coefficient of 23.56. In Q16.16 this is 1,543,722 — multiplied by a signal sample it overflows a 32-bit accumulator. SOS biquads have maximum `|a|` of 1.993, giving 130,637 in Q16.16 — safe for 32-bit arithmetic in each DSP48 slice.

**Why `sosfilt` not `filtfilt` in MATLAB?**  
`filtfilt` is zero-phase — it processes the signal backwards in time to cancel group delay. The FPGA processes samples causally (one at a time, forward only). Using `sosfilt` in MATLAB exactly matches the causal behaviour of the hardware.

**Why Q8.8 for the signal and Q16.16 for coefficients?**  
ECG amplitude range (±2 mV) fits comfortably in 8 integer bits. Coefficient precision needs to be higher because rounding errors in feedback coefficients accumulate in the recursive filter — Q16.16 keeps coefficient error below 0.002%.

**Why patient-wise splitting?**  
Beat-wise splits leak patient identity into both train and test sets — the model memorises patient-specific morphology and reports inflated accuracy. Patient-wise splitting tests genuine generalisation across unseen patients, which is the clinically meaningful evaluation.

---

## References

- Moody G, Mark R. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 2001.
- Pan J, Tompkins W. *A real-time QRS detection algorithm.* IEEE Transactions on Biomedical Engineering, 1985.
- He K et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Association for the Advancement of Medical Instrumentation. *ANSI/AAMI EC57: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.* 2012.

---

## Author

Project developed as part of a Major Project on low-power FPGA-based ECG classification.  
Supervisor guidance: DSP-first approach — verify hardware filter before AI model.
