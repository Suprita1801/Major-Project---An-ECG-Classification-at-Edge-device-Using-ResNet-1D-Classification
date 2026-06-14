# 🫀 FPGA-Based ECG Arrhythmia Classification — Two-Branch DSCResNet

A complete two-stage pipeline for real-time ECG arrhythmia classification:
- **Stage 1 (Software):** 4th-order Butterworth bandpass filter designed in MATLAB + a dual-branch depthwise separable convolutional neural network trained in Python (Google Colab)
- **Stage 2 (Hardware):** The same Butterworth filter implemented in Verilog as four cascaded Q16.16 fixed-point biquad sections on a Xilinx Zynq-7020 FPGA

> 📄 This project accompanies the conference paper:
> **"FPGA-Based ECG Arrhythmia Classification Using Dual-Branch DSCResNet with Inter-Patient Evaluation"**

---

## Why This Project Is Different

Most published ECG classifiers report 96–99% accuracy — but achieve it by splitting beats *randomly*, so the same patient's data appears in both training and test sets. This is data leakage, and the AAMI EC57 standard explicitly prohibits it.

This project enforces **strict inter-patient evaluation**: the 8 test patients are completely unseen during training. The reported **83.11% accuracy and macro F1 of 0.521** are honest, clinically meaningful numbers.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Overall Accuracy | **83.11%** |
| Macro F1-Score | **0.521** |
| Ventricular Ectopic Recall | **79.3%** (life-critical PVCs) |
| Supraventricular Recall | **43.2%** (PACs via RR timing) |
| Model Size | **0.43 MB** (smallest in comparison) |
| Parameters | 107,621 |
| Evaluation Protocol | AAMI EC57 inter-patient ✅ |

### FPGA Filter (Xilinx Zynq-7020)

| Resource | Usage |
|---|---|
| LUTs | 2,326 (4.37%) |
| Flip-Flops | 516 (0.48%) |
| DSP48 Blocks | 0 |
| Logic Power | 29 mW |
| Max Frequency | 116.6 MHz |
| Pipeline Latency | 4 clock cycles |

---

## Pipeline Overview

```
MIT-BIH Database (360 Hz)
        │
        ▼
┌─────────────────────────────┐   STAGE 1 — SOFTWARE (MATLAB + Colab)
│  Butterworth BPF            │   filter_design.m
│  4th-order, 0.5–45 Hz      │   zero-phase via filtfilt()
│  SOS coefficients exported  │
└────────────┬────────────────┘
             │  Same coefficients shared ──────────────────────┐
             ▼                                                  ▼
┌─────────────────────────────┐             ┌──────────────────────────────┐
│  Beat Extraction            │             │  Verilog BPF (sos_filter.v)  │  STAGE 2 — HARDWARE
│  252-sample window/beat     │             │  4 cascaded Q16.16 biquads   │  (Zynq-7020)
│  Pan-Tompkins R-peak detect │             │  116.6 MHz, 4-cycle latency  │
└────────────┬────────────────┘             └──────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Two-Branch DSCResNet (Python/Colab)    │
│                                         │
│  CNN Branch          RR-MLP Branch      │
│  (waveform shape)    (8 RR features)    │
│       128-dim   +       32-dim          │
│              ↓                          │
│       160-dim Fusion Classifier         │
│              ↓                          │
│   N  |  S  |  V  |  F  |  U            │
└─────────────────────────────────────────┘
```

---

## Architecture

### Two-Branch DSCResNet

The model processes each beat through two parallel branches:

**CNN Branch — Waveform Morphology**
- Stem convolution block → 3× DSC ResBlocks (32→64→128 channels)
- Depthwise separable convolutions reduce MAC count ~8× vs standard convolutions
- Global Average Pooling → 128-dimensional morphology vector
- Captures wide, aberrant QRS morphology for Ventricular ectopic (PVC) detection

**RR-MLP Branch — Rhythm Timing**
- 8 normalized RR interval features → 2× FC layers with ReLU
- 32-dimensional timing vector encoding rhythm context
- Essential for Supraventricular ectopic (PAC) detection — waveform morphology alone gives 0% S-class recall; adding the RR branch brings it to 43.2%

**Fusion Classifier**
- Concatenates both vectors → 160-dimensional representation
- 2× FC layers → 5 AAMI class logits → argmax

### Why SOS for the FPGA Filter?

Direct-form IIR coefficients reach values like `a = 26.3`. At Q16.16, that becomes `26.3 × 65536 = 1,722,368`. Multiplied by a 32-bit signal sample → immediate int32 overflow → garbage output.

SOS splits the 8th-order polynomial into 4 cascaded 2nd-order sections where the max denominator coefficient is only `1.993`, giving `1.993 × 65536 = 130,637` — safely within the 32-bit signed range. Each biquad implements Direct Form I:

```
y[n] = B0·x[n] + B1·x[n-1] + B2·x[n-2] − A1·y[n-1] − A2·y[n-2]
```

### Exported Q16.16 SOS Coefficients (float × 65536)

| Section | B0 | B1 | B2 | A1 | A2 |
|---|---|---|---|---|---|
| 1 | 645 | 1,289 | 645 | -57,134 | 14,251 |
| 2 | 65,536 | 131,072 | 65,536 | -73,411 | 38,064 |
| 3 | 65,536 | -131,072 | 65,536 | -130,003 | 64,472 |
| 4 | 65,536 | -131,072 | 65,536 | -130,637 | 65,106 |

---

## Dataset

**MIT-BIH Arrhythmia Database** — 48 two-lead ECG recordings, 360 Hz, 47 subjects.

Beats are grouped into 5 AAMI classes:

| Class | Label | Description |
|---|---|---|
| Normal | N | Dominant class (>85% of beats) |
| Supraventricular ectopic | S | PAC — rhythm disorder |
| Ventricular ectopic | V | PVC — life-critical |
| Fusion | F | Rare class |
| Unknown | U | Not in test split |

**Inter-patient split (AAMI EC57 compliant):**
- Train: 34 patients (100-series records)
- Validation: 7 patients
- Test: 8 patients (200-series records — the hardest in the database)
- Zero beat-level overlap between any two partitions

> MIT-BIH data is available free at [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

---

## Repository Structure

```
ecg_fpga_classifier/
│
├── matlab/
│   ├── filter_design.m              # Butterworth BPF design + SOS export  ← run first
│   ├── preprocessing.m              # ECG preprocessing pipeline
│   └── data/
│       ├── processed/
│       │   ├── filter_coeffs.mat            # b, a, SOS (float + fixed-point)
│       │   ├── filter_coeffs_scaled.mat     # Q16.16 direct-form (reference)
│       │   └── butterworth_fpga_coeffs.txt  # Human-readable coefficient file
│       ├── raw/                     # Place MIT-BIH .dat/.hea files here
│       └── splits/                  # Train/val/test index files
│
├── colab/
│   └── ecg_classifier.ipynb         # Two-branch DSCResNet training notebook
│
├── verilog/
│   ├── sos_filter.v                 # Top-level 4-stage cascaded biquad filter
│   ├── biquad_section.v             # Single Direct Form I biquad
│   └── tb_sos_filter.v              # Testbench (reads input_stimulus.txt)
│
├── weights/
│   └── best.pt                      # Best model checkpoint (epoch 14)
│
├── results/
│   └── confusion_matrix.png
│
└── input_stimulus.txt               # 100-sample 10 Hz sine test vector for Verilog TB
```

---

## Getting Started

### Prerequisites

| Tool | Purpose | Notes |
|---|---|---|
| MATLAB + Signal Processing Toolbox | Filter design & preprocessing | R2021a or later |
| Python 3.8+ / Google Colab | Model training | Free Colab tier sufficient |
| PyTorch 1.12+ | Neural network | `pip install torch` |
| Xilinx Vivado | FPGA synthesis | 2022.1 or later, free WebPACK edition works |

### Step 1 — Design the Filter (MATLAB)

```matlab
% Run this FIRST before anything else
run('matlab/filter_design.m')
```

This will design the 4th-order Butterworth bandpass filter, export SOS and Q16.16 fixed-point coefficients, generate the `input_stimulus.txt` Verilog test vector, and plot the magnitude and phase frequency response.

### Step 2 — Preprocess ECG Data (MATLAB)

```matlab
% Requires MIT-BIH .dat/.hea files placed in matlab/data/raw/
run('matlab/preprocessing.m')
```

Applies the Butterworth filter via `filtfilt` (zero-phase), detects R-peaks using a simplified Pan-Tompkins algorithm, extracts 252-sample beat windows (108 pre + 144 post R-peak), computes 8 RR interval features, and saves inter-patient train/val/test splits.

### Step 3 — Train the Classifier (Google Colab)

1. Open `colab/ecg_classifier.ipynb` in Google Colab
2. Upload the processed split files from `matlab/data/splits/`
3. Run all cells — training takes ~30 epochs
4. Download `best.pt` from the output

### Step 4 — Synthesize the FPGA Filter (Vivado)

1. Create a new RTL project targeting `xc7z020clg400-1`
2. Add `verilog/sos_filter.v` and `verilog/biquad_section.v` as design sources
3. Add `verilog/tb_sos_filter.v` as simulation source
4. Run behavioral simulation using `input_stimulus.txt` to verify filter output
5. Run Synthesis → Implementation → check timing report for WNS ≥ 0 ns

Expected results: 2,326 LUTs, 516 FFs, 0 DSP48s, Fmax ≥ 116.6 MHz.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 5 × 10⁻⁵ |
| Weight Decay | 1 × 10⁻² |
| Epochs | 30 (best checkpoint at epoch 14) |
| Class Weighting | Square-root inverse |
| Label Smoothing | 0.10 |
| LR Schedule | ReduceLROnPlateau (factor 0.5, patience 5) |

---

## Citation

If you use this code, filter coefficients, or architecture in your research, please cite:

```bibtex
@inproceedings{ecg_fpga_dscresnet,
  title     = {FPGA-Based ECG Arrhythmia Classification Using Dual-Branch DSCResNet with Inter-Patient Evaluation},
  author    = {Juanita Martina T and Suprita T and Anuraj V and Vinoth Raj R and Dhandapani Vaithiyanathan},
  booktitle = {[Conference Name and Year]},
  year      = {2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE). Feel free to use, build upon, and adapt with attribution.

---

## Authors

| Name | Institution |
|---|---|
| Suprita T | SRM Institute of Science and Technology, Tiruchirappalli |
| Juanita Martina T | SRM Institute of Science and Technology, Tiruchirappalli |
| Anuraj V | National Institute of Technology Delhi |
| Vinoth Raj R | SRM Institute of Science and Technology, Tiruchirappalli |
| Dhandapani Vaithiyanathan | National Institute of Technology Delhi |

---

## Acknowledgements

MIT-BIH Arrhythmia Database courtesy of [PhysioNet](https://physionet.org). FPGA synthesis on Xilinx Zynq-7020 using Vivado Design Suite.
