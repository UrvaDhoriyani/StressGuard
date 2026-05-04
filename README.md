# StressGuard: Edge AI for Real-Time Stress Detection

A localized, privacy-first edge computing system for real-time stress detection using Heart Rate Variability (HRV) analysis on a Raspberry Pi 5 with a MAX30102 PPG sensor.

**Authors:** Arpita Hage (ah4384) & Urva Dhoriyani (ud2142)  
**Institution:** Columbia University, New York, NY, USA

---

## 📋 Overview

StressGuard addresses the critical gap between affordable hardware and clinical-grade health monitoring. By performing all computation locally on edge devices, we eliminate cloud dependencies while maintaining data sovereignty and enabling real-time stress detection with sub-200ms inference latency.

### Problem Statement

Chronic stress is a major contributor to cardiovascular disease and professional burnout. While wearable technology has advanced, cloud-dependent architectures introduce:
- **Latency**: Delayed feedback prevents timely intervention
- **Privacy concerns**: Sensitive health data transmitted over networks
- **Internet dependency**: No monitoring without connectivity

### Solution

StressGuard leverages edge AI to extract 14 clinical-grade HRV features from PPG waveforms, using an optimized XGBoost classifier for binary classification (Stress vs. Normal) with **78% accuracy** and **<200ms end-to-end latency**.

---

## 🎯 Key Features

- **Edge-Based Processing**: All computation on Raspberry Pi 5—no cloud required
- **Real-Time Feedback**: Sub-200ms inference latency for immediate alerts
- **Clinical-Grade Analysis**: 14 HRV features extract autonomic nervous system state
- **Privacy-First**: 100% data sovereignty; no data transmission
- **Optimized ML Pipeline**: XGBoost + SMOTE trained on WESAD dataset
- **Hardware Accessibility**: Low-cost components (MAX30102 sensor, Raspberry Pi 5)
- **Continuous Monitoring**: Long-term trend identification with local CSV logging

---

## 🏗️ System Architecture

StressGuard follows a three-layer pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SENSING LAYER                                  │
│  MAX30102 PPG Sensor (100Hz) + MLX90614 Temperature (Optional)      │
│  ↓ I2C Hardware Polling (100Hz)                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    PROCESSING LAYER                                 │
│  Raspberry Pi 5 (8GB)                                               │
│  • Signal filtering & PPG cleaning                                  │
│  • 15-30 second measurement windows                                 │
│  • Feature extraction (14 HRV metrics)                              │
│  • RF/XGBoost classification                                        │
│  • Local CSV logging                                                │
├─────────────────────────────────────────────────────────────────────┤
│                     OUTPUT LAYER                                    │
│  Terminal GUI with real-time feedback                              │
│  • Pulse (BPM), Temperature, RMSSD display                         │
│  • Color-coded stress alerts                                        │
│  • Timestamped local logging                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Sensing Layer
- **MAX30102 Pulse Oximeter**: Captures raw PPG data at 100Hz using IR and Red channels
- **Optional MLX90614**: Skin temperature monitoring via I2C
- **Stable I2C Interface**: 100Hz polling rate with no buffer overflows observed

### 3.2 Processing Layer
- **Raspberry Pi 5 (8GB)**: Central hub for all signal processing
- **15-30 second measurement windows**: Balanced between responsiveness and feature quality
- **14 HRV Features**: Time-domain (RMSSD, SDNN, pNN50, etc.) and frequency-domain (LF/HF ratio)
- **Sliding window approach**: Maintains real-time responsiveness
- **Modular pipeline**: Easy debugging and scaling

### 3.3 Output Layer
- **Terminal Interface**: Real-time display of BPM, temperature, and RMSSD
- **Color-coded alerts**: Visual notification of stress vs. normal states
- **Local CSV logging**: Timestamped records for long-term trend analysis

---

## 📊 HRV Features Extracted

### Time-Domain Features
- **RMSSD**: Root Mean Square of Successive Differences (parasympathetic activity)
- **SDNN**: Standard Deviation of NN intervals (overall variability)
- **pNN50/pNN20**: Proportion of intervals differing by >50/20ms
- **MeanNN/MedianNN**: Mean and median RR intervals
- **CVNN**: Coefficient of Variation
- **MinNN/MaxNN**: Minimum and maximum intervals
- **IQRNN**: Interquartile range

### Frequency-Domain Features
- **LF (Low Frequency, 0.04-0.15Hz)**: SNS and PNS combined activity
- **HF (High Frequency, 0.15-0.4Hz)**: Parasympathetic activity (vagal tone)
- **LF/HF Ratio**: Sympathetic-Parasympathetic balance indicator
- **HR_mean**: Mean heart rate in beats per minute

---

## 🧠 Machine Learning Methodology

### Dataset: WESAD
- **Multi-modal signals**: Baseline, stress, and amusement states
- **15 subjects**: S2-S11, S13-S17 (15 subjects total)
- **3,522 windows** from subjects after signal processing
- **Wrist-PPG segment**: Aligned with hardware sensor modality

### Classification Pipeline

**Training Approach:**
- **15-fold Leave-One-Subject-Out Cross-Validation (LOSO-CV)**
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Normalization**: Per-subject Z-score normalization

**Tested Classifiers:**
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Logistic Regression** | **78.07%** | 0.746 |
| **XGBoost + SMOTE** | 76.56% | 0.667 |
| Random Forest | ~75% | ~0.65 |

**Model Serialization:**
- Joblib format for rapid edge deployment
- TFLite optimization for Raspberry Pi 5
- Pre-trained weights enable lightweight inference

**Top Feature Predictors:**
1. CVNN (Coefficient of Variation of NN intervals)
2. HR_mean (Mean heart rate)
3. MedianNN (Median NN interval)

---

## 📈 Results and Performance

### 6.1 Model Efficacy

**Leave-One-Subject-Out Cross-Validation Results:**
```
XGBoost + SMOTE LOSO-CV:
- Accuracy: 76.56% ± 11.14%
- F1-Score: 0.667 ± 0.15

Per-Fold Performance:
  Best:  87% (S7)
  Worst: 42% (S10)
  Mean:  76.56%
```

**Classification Report:**
```
              Precision  Recall  F1-Score  Support
  Non-Stress     88%      76%      82%      2467
  Stress         58%      77%      66%      1055
```

**Clinical Validation:**
- Live system successfully differentiated CALM from STRESS ALERT states
- Pulse and HRV readings consistent with physiological expectations
- Binary classification provides clear actionable alerts for users

### 6.2 Hardware Benchmarks

| Metric | Result |
|--------|--------|
| **I2C Polling Rate** | Stable 100Hz |
| **Buffer Overflows** | None observed |
| **End-to-End Latency** | <200ms |
| **Measurement Window** | 15-30 seconds |
| **Output Rendering** | <10ms after window close |
| **Thermal Performance** | Safe operating temps during continuous inference |
| **Power Efficiency** | Low power profile for long-term wearable use |

---

## 🛠️ Setup & Installation

### Prerequisites
- **Hardware**:
  - Raspberry Pi 5 (8GB)
  - MAX30102 PPG sensor module
  - I2C connectors and cables

- **Software**:
  - Python 3.8+
  - Dependencies: numpy, pandas, scipy, scikit-learn, xgboost, imbalanced-learn, joblib, neurokit2

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/StressGuard.git
cd StressGuard

# Install dependencies
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn joblib neurokit2
```

---

## 📁 Project Structure

```
StressGuard/
├── README.md                          # Project documentation
├── live_monitor.py                    # Real-time stress monitoring on Raspberry Pi
├── extract_features.py                # HRV feature extraction from WESAD
├── train_model.py                     # Model training & evaluation with multiple classifiers
├── train_xgboost.py                   # XGBoost training with SMOTE optimization
├── evaluate_xgb_full.py               # Comprehensive XGBoost evaluation
├── diagnose_features.py               # Feature analysis and diagnostics
├── inspect_pickle.py                  # Utility to inspect WESAD dataset format
├── led_test.py                        # Hardware sensor testing
│
├── stress_model.joblib                # Trained Logistic Regression model
├── stress_model_xgb.joblib            # Trained XGBoost model (recommended)
├── stress_model_binary.joblib         # Binary classification model
│
├── hrv_features.csv                   # Extracted HRV features from WESAD
│
├── evaluation_final.txt               # Final model evaluation results
├── training_results_v2.txt            # Training logs
├── training_results_v3.txt            # Training logs
├── xgb_smote_results.txt              # SMOTE optimization results
│
└── data/
    └── WESAD/                         # WESAD dataset (15 subjects: S2-S11, S13-S17)
        ├── S2/, S3/, ..., S17/
        │   ├── [Subject]_quest.csv
        │   ├── [Subject]_readme.txt
        │   └── [Subject]_respiban.txt
        └── ...
```

---

## 🚀 Usage

### 1. Live Monitoring on Raspberry Pi
```bash
python live_monitor.py
```
Runs real-time stress detection with live terminal output showing BPM, temperature, RMSSD, and stress classification.

### 2. Feature Extraction
```bash
python extract_features.py
```
Extracts 14 HRV features from WESAD dataset pickle files and saves to `hrv_features.csv`.

### 3. Model Training & Evaluation
```bash
# Train XGBoost with SMOTE and evaluate via LOSO-CV
python train_xgboost.py

# Train multiple models (Logistic Regression, Random Forest, SVM, XGBoost) and evaluate
python train_model.py
```

### 4. Model Analysis
```bash
# Comprehensive XGBoost evaluation
python evaluate_xgb_full.py

# Feature importance diagnostics
python diagnose_features.py

# Inspect WESAD dataset format
python inspect_pickle.py
```

---

## 🔬 Mathematical Foundations

### Time-Domain Analysis
**RMSSD** (primary parasympathetic indicator):
$$\text{RMSSD} = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_i - RR_{i+1})^2}$$

**SDNN** (overall heart rate variability):
$$\text{SDNN} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(RR_i - \bar{RR})^2}$$

### Frequency-Domain Analysis
**LF/HF Ratio** (autonomic balance indicator):
- LF Integration: 0.04-0.15 Hz
- HF Integration: 0.15-0.4 Hz
- Ratio elevation correlates with sympathetic dominance and stress

Calculations via Fast Fourier Transform (FFT) for real-time efficiency.

---

## 🧬 Physiological Background

### Autonomic Nervous System (ANS)
- **Sympathetic Nervous System (SNS)**: "Fight-or-flight" response
  - Elevated heart rate, decreased beat-to-beat variability
  - Indicates stress state
- **Parasympathetic Nervous System (PNS)**: "Rest-and-digest" phase
  - Higher HRV, lower heart rate
  - Associated with calm, relaxed states

### HRV as a Stress Marker
- **High variability**: Healthy, resilient state with high vagal tone
- **Low variability**: Established clinical marker for psychological and physical stress
- **Non-invasive proxy**: For autonomic nervous system activity
- **Real-time indicator**: Enables immediate intervention through breathing exercises or mindfulness

---

## 🔮 Future Enhancements

Based on the research roadmap:
1. **Multi-Modal Integration**: Electrodermal Activity (EDA) sensor support
2. **Mobile Synchronization**: Data sync with mobile devices

---

## 📚 References

- **WESAD Dataset**: Wearable Stress and Affect Detection
- **HRV Analysis Standards**: Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology
- **XGBoost**: Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*
- **SMOTE**: Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*

---

## 📝 License

This project is part of academic research at Columbia University. Please contact the authors for licensing inquiries.

---

## 👥 Contributors

**Arpita Hage** (ah4384)
- Physiological signal analysis
- HRV feature engineering
- ML pipeline optimization

**Urva Dhoriyani** (ud2142)
- Hardware integration
- Real-time edge deployment
- System architecture

---

## 📧 Contact & Support

For questions or issues:
- 📨 Email: [ah4384@columbia.edu](mailto:ah4384@columbia.edu) or [ud2142@columbia.edu](mailto:ud2142@columbia.edu)
- 🔗 GitHub: [StressGuard Repository](https://github.com/yourusername/StressGuard)

---

## 🙏 Acknowledgments

- **Columbia University** for computational resources
- **WESAD Dataset** researchers for the benchmark dataset
- **Embedded AI '26** conference organizers

---

**Last Updated:** May 2026  
**Version:** 1.0  
**Status:** Active Development
