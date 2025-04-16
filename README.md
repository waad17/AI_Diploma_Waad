# AI_Diploma_Waad
Large-Scale GPU-Accelerated Machine  Learning Project
# Predictive Maintenance on MetroPT-3 Using GPU-Accelerated Machine Learning

This project applies machine learning models on real-time sensor data from a metro train's air compressor system (MetroPT-3) to predict system failures. Using RAPIDS and GPU-accelerated libraries, the goal is to detect anomalies early and support predictive maintenance strategies that minimize unplanned downtime.

## 🚀 Project Objectives
- Analyze multivariate time-series data from 15 sensors
- Use cuDF and cuML for GPU-accelerated data processing
- Implement and compare three ML models:
  - Random Forest (cuML)
  - Logistic Regression (cuML)
  - XGBoost (`gpu_hist`)
- Track GPU resource utilization: training time, memory, GPU load
- Provide interpretable model outputs (confusion matrices, precision, recall)

## 📊 Dataset Description
- **Source:** UCI MetroPT-3 Dataset
- **Type:** Multivariate time-series
- **Records:** 1.5M+
- **Sampling Rate:** 1 Hz
- **Target Variable:** `label` (0 = Normal, 1 = Fault)

## 🧪 Key Results
| Model               | Accuracy | Training Time | GPU Util (%) | Memory Used (MiB) |
|--------------------|----------|----------------|----------------|-------------------|
| Random Forest       | 0.9967   | 3.24 s         | 28.5%          | 2100              |
| Logistic Regression | 0.9895   | 2.01 s         | 18.2%          | 1350              |
| XGBoost             | 0.9995   | 4.37 s         | 35.6%          | 2800              |

## 📈 Visual Analysis
- Confusion Matrices for all models
- Boxplots by label for feature impact
- Correlation heatmap of sensor variables
- Bar chart of model vs. GPU performance

## 📂 Repository Structure
```
.
├── waad_project.ipynb           # Main notebook with analysis & models
├── model.joblib                 # Saved trained model
├── scaler.joblib                # Scaler for feature normalization
├── X_test.csv                   # Test features
├── y_test.csv                   # Test labels
├── prediction.py                # Script to load model and make predictions
├── README.md                    # Project overview and results
```

## 💡 Recommendations
- Use **XGBoost** for highest fault detection accuracy
- Use **Random Forest** for fast, real-time applications
- Use **Logistic Regression** for constrained environments (e.g., edge devices)

## 👩‍💻 Author
**Waad Alqahtani**  
Tuwaiq Academy – AI Diploma (April 2025)

---


