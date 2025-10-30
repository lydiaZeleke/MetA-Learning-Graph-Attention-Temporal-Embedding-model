# MAL-GATE: Meta-Learning Graph Attention Temporal Embedding for Anomaly Detection

MAL-GATE is a Python-based framework designed to enhance **trustworthy situational awareness** in Detect-and-Avoid (DAA) systems for Autonomous Aerial Vehicles (AAV).  
Sensor uncertainties can degrade alerting and guidance reliability, eroding operator trust. MAL-GATE addresses this challenge by integrating **graph-based spatial encoding**, **hybrid temporal modeling**, and **meta-learning adaptation** to detect, localize, and interpret anomalies across diverse encounter datasets.

## Project Overview
- **Objective**: Provide a data-driven mechanism for detecting and diagnosing anomalies in real-time, while linking them to DAA performance metrics such as alert jitter, severity of loss of well-clear (SLoWC), and guidance fluctuation.  
- **Impact**: Supports system transparency, operator trust calibration, and safe integration of autonomous aerial vehicles in shared airspace.  

## Features
- Meta-learning for fast adaptation to short, varied encounter datasets.  
- Graph attention networks for modeling feature interdependencies.  
- Temporal embeddings for both short- and long-term sequential dependencies.  
- Integrated explainability for feature-level anomaly diagnosis.  
- Automated hyperparameter optimization (grid and randomized search).  
- GPU-accelerated training and inference.  

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mal-gate.git
   cd mal-gate

   Usage
1. **Running the Model**

#### Conventional Training (without meta-learning):

```bash 
python train.py --dataset CUSTOM --normalize False
```

## Meta-Learning Training (with meta-learning enabled):

```bash
python train.py --dataset CUSTOM --normalize False --meta_training True
```

## Inference (using a trained model for anomaly detection):
```bash
python predict.py --dataset CUSTOM --normalize False --load_path output/Custom/24082025_211433
```
2. ** Hyperparameter Optimization**

## Randomized Search
Run randomized search for hyperparameter tuning:
```bash
python train_randomized_search.py
```

## Grid Search
Run grid search for hyperparameter tuning:
```bash
python train_grid_search.py
```

Optimized hyperparameters are saved as a JSON file in the HP_Logs/ directory.
