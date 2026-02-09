# 🛠️ Engine Predictive Maintenance - MLOps Pipeline

> **An end-to-end Machine Learning Operations (MLOps) system for predicting engine failures using sensor data**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-green?logo=github-actions)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Hub-yellow?logo=huggingface)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [File Documentation](#-file-documentation)
- [Dataset Description](#-dataset-description)
- [Installation](#-installation)
- [Usage](#-usage)
- [MLOps Pipeline](#-mlops-pipeline)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **predictive maintenance system** for engines using machine learning. The system analyzes sensor data (RPM, pressures, temperatures) to classify whether an engine is **healthy** or **requires maintenance**, enabling proactive maintenance scheduling and reducing downtime.

### Key Objectives

- **Binary Classification**: Predict engine condition (0 = Normal, 1 = Requires Maintenance)
- **MLOps Best Practices**: Automated pipeline with CI/CD, experiment tracking, and model versioning
- **Deployment Ready**: Containerized application with Hugging Face Spaces hosting
- **Reproducibility**: Complete data versioning and experiment tracking with MLflow

---

## 🔗 Live Demo

| Resource | Link |
|----------|------|
| 🚀 **Live Application** | [Engine Maintenance App](https://huggingface.co/spaces/UsamaMasood12ak/engine-maintenance-space) |
| 📦 **Trained Model** | [Hugging Face Model](https://huggingface.co/UsamaMasood12ak/engine-maintenance-model) |
| 📊 **Dataset Repository** | [Hugging Face Dataset](https://huggingface.co/datasets/UsamaMasood12ak/engine-maintenance-dataset) |
| 🔄 **GitHub Actions** | [Workflow Runs](https://github.com/UsamaMasood12/engine-predictive-maintenance/actions) |

---

## ✨ Features

- **Real-time Predictions**: Interactive web interface for instant engine health assessment
- **Sensor Visualization**: Radar charts and gauge displays for sensor readings
- **Experiment Tracking**: MLflow integration for hyperparameter tuning and metric logging
- **Automated Pipeline**: GitHub Actions workflow for continuous deployment
- **Model Versioning**: Hugging Face Model Hub for model storage and versioning
- **Data Versioning**: Hugging Face Dataset Hub for dataset management
- **Containerization**: Docker support for consistent deployments

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOPS PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │    DATA     │    │    DATA     │    │   MODEL     │    │  DEPLOYMENT │ │
│  │ REGISTRATION│───▶│ PREPARATION │───▶│  TRAINING   │───▶│  & HOSTING  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  HF Dataset │    │ Train/Test  │    │   MLflow    │    │  HF Space   │ │
│  │     Hub     │    │   Splits    │    │  Tracking   │    │  (Docker)   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GITHUB ACTIONS CI/CD AUTOMATION                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
engine-predictive-maintenance/
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 pipeline.yml          # CI/CD pipeline with 4 jobs
├── 📁 data/
│   ├── 📄 engine_data.csv           # Raw sensor dataset (1.3 MB, 7 columns)
│   └── 📁 processed/
│       ├── 📄 train.csv             # Training split (80%)
│       └── 📄 test.csv              # Testing split (20%)
├── 📁 models/
│   └── 📄 best_model.joblib         # Trained Random Forest model
├── 📁 src/                           # Main source code (10 files)
│   ├── 📄 config.py                 # Central configuration management
│   ├── 📄 data_register.py          # Upload raw data to HF Dataset Hub
│   ├── 📄 data_prep.py              # Data cleaning and train/test splitting
│   ├── 📄 eda.py                    # Exploratory Data Analysis
│   ├── 📄 train.py                  # Model training with MLflow tracking
│   ├── 📄 inference.py              # Prediction utilities
│   ├── 📄 app.py                    # Streamlit web application
│   ├── 📄 deploy_to_hf.py           # Deploy to Hugging Face Space
│   ├── 📄 hf_data_utils.py          # Hugging Face Dataset Hub utilities
│   └── 📄 hf_model_utils.py         # Hugging Face Model Hub utilities
├── 📄 Dockerfile                     # Container definition for deployment
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                     # Git ignore patterns
└── 📄 README.md                      # This documentation
```

---

## 📚 File Documentation

### Source Code Files (`src/`)

#### 1. `config.py` - Central Configuration Management
**Lines: 90 | Purpose: Application-wide configuration and path management**

```python
# Key configurations:
- PROJECT_ROOT          # Automatic project root detection
- DATA_DIR              # Path to data directory
- RAW_DATA_FILE         # Path to engine_data.csv
- PROCESSED_DIR         # Path to processed data folder
- TRAIN_FILE / TEST_FILE # Paths to train/test CSV files

# Feature configuration:
- FEATURE_COLUMNS       # ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 
                        #  'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature']
- TARGET_COLUMN         # 'Engine_Condition'
- RAW_COLUMN_RENAME_MAP # Maps raw column names to standardized names

# Hugging Face configuration:
- HF_TOKEN              # Authentication token (from env variable)
- HF_DATASET_REPO       # Dataset repository ID
- HF_MODEL_REPO         # Model repository ID
- HF_SPACE_REPO         # Space repository ID

# MLflow configuration:
- MLFLOW_TRACKING_URI   # Local MLflow tracking URI
- MLFLOW_EXPERIMENT_NAME # Experiment name for tracking

# Model artifacts:
- MODELS_DIR            # Path to models directory
- BEST_MODEL_LOCAL_PATH # Path to saved model file
```

---

#### 2. `data_register.py` - Dataset Registration
**Lines: 60 | Purpose: Upload raw engine dataset to Hugging Face Dataset Hub**

**What it does:**
1. Validates that raw data file exists locally
2. Checks for valid Hugging Face token
3. Creates dataset repository on HF Hub if it doesn't exist
4. Uploads `engine_data.csv` to the dataset repository

**Key function:**
```python
def main():
    # Validates configuration and uploads raw data to HF
    register_raw_engine_data_to_hf()
```

**Usage:**
```bash
cd src
python data_register.py
```

---

#### 3. `data_prep.py` - Data Preparation & Cleaning
**Lines: 155 | Purpose: Clean data, create train/test splits, upload to HF**

**What it does:**
1. **Load data**: From HF Dataset Hub or local fallback
2. **Clean data**: 
   - Rename columns to standardized names
   - Drop duplicates
   - Handle missing values (fill with median)
   - Ensure target is binary integer
3. **Split data**: 80/20 train/test split with stratification
4. **Save & upload**: Save locally and upload to HF Dataset Hub

**Key functions:**
```python
def _load_raw_data_from_hf_or_local() -> pd.DataFrame
    # Loads data from HF Hub or falls back to local CSV

def _clean_data(df: pd.DataFrame) -> pd.DataFrame
    # Performs data cleaning and preprocessing

def _train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
    # Creates stratified train/test splits
```

**Usage:**
```bash
cd src
python data_prep.py
```

---

#### 4. `eda.py` - Exploratory Data Analysis
**Lines: 86 | Purpose: Generate EDA visualizations and statistics**

**What it does:**
1. Loads and cleans data using shared functions
2. Prints data overview (shape, types, missing values, statistics)
3. Generates visualizations:
   - Target distribution bar chart
   - Feature histograms
   - Correlation heatmap
   - Pairplot for feature relationships

**Output location:** `notebooks/figures/`

**Key function:**
```python
def run_eda():
    # Generates all EDA outputs and saves figures
```

**Usage:**
```bash
cd src
python eda.py
```

---

#### 5. `train.py` - Model Training with Experiment Tracking
**Lines: 203 | Purpose: Train Random Forest model with hyperparameter tuning and MLflow logging**

**What it does:**
1. **Load data**: From HF Dataset Hub or local files
2. **Build model**: sklearn Pipeline with StandardScaler + RandomForestClassifier
3. **Hyperparameter tuning**: RandomizedSearchCV with 5-fold CV
   - `n_estimators`: [100, 200, 300, 400]
   - `max_depth`: [None, 5, 10, 20]
   - `min_samples_split`: [2, 5, 10]
   - `min_samples_leaf`: [1, 2, 4]
   - `max_features`: ['sqrt', 'log2', None]
   - `bootstrap`: [True, False]
4. **MLflow tracking**: Log all experiments, parameters, and metrics
5. **Model evaluation**: Accuracy, Precision, Recall, F1-score
6. **Save & upload**: Save model locally and upload to HF Model Hub

**Key functions:**
```python
def _load_train_test_from_hf_or_local() -> Tuple[pd.DataFrame, pd.DataFrame]
def _build_model_and_search_space() -> Tuple[Pipeline, Dict]
def _evaluate_model(model, X_test, y_test) -> Dict[str, float]
def main() -> None  # Main training pipeline
```

**Usage:**
```bash
cd src
python train.py
```

---

#### 6. `inference.py` - Prediction Utilities
**Lines: 98 | Purpose: Load models and make predictions**

**What it does:**
1. Load trained model from local storage or Hugging Face Hub
2. Build input DataFrame from sensor readings
3. Generate predictions with probability scores

**Key functions:**
```python
def load_local_model() -> object
    # Loads model from models/best_model.joblib

def load_hf_model() -> object
    # Downloads and loads model from Hugging Face

def build_input_dataframe(inputs: Dict[str, float]) -> pd.DataFrame
    # Converts sensor readings to model input format

def predict_engine_condition(
    inputs: Dict[str, float],
    model: Optional[object] = None,
    source: str = "local"
) -> Dict[str, float]
    # Returns {'prediction': 0/1, 'probability_faulty': float}
```

**Example usage:**
```python
from inference import predict_engine_condition

result = predict_engine_condition({
    "Engine_RPM": 1500,
    "Lub_Oil_Pressure": 4.5,
    "Fuel_Pressure": 15.0,
    "Coolant_Pressure": 3.0,
    "Lub_Oil_Temperature": 85.0,
    "Coolant_Temperature": 88.0
})
# Returns: {'prediction': 0, 'probability_faulty': 0.12}
```

---

#### 7. `app.py` - Streamlit Web Application
**Lines: 434 | Purpose: Interactive web interface for engine condition prediction**

**What it does:**
1. **Modern UI**: Custom CSS styling with gradients and animations
2. **Sensor inputs**: Number inputs for 6 sensor readings
3. **Visualization**: 
   - Radar chart for sensor overview
   - Gauge chart for fault probability
4. **Prediction display**: Color-coded results with recommendations
5. **Responsive design**: Works locally and on Hugging Face Spaces

**Key features:**
- Automatic model source detection (local vs HF)
- Real-time sensor visualization
- Probability-based risk assessment
- Recommendation engine for maintenance actions

**Key functions:**
```python
def create_gauge_chart(value, title, color) -> go.Figure
def create_sensor_comparison_chart(sensor_data) -> go.Figure
def main() -> None  # Streamlit app entry point
```

**Usage:**
```bash
streamlit run src/app.py
```

---

#### 8. `hf_data_utils.py` - Hugging Face Dataset Utilities
**Lines: 144 | Purpose: Upload/download files to/from HF Dataset Hub**

**Key functions:**
```python
def create_or_get_dataset_repo(repo_id, token, private) -> None
    # Creates dataset repo if it doesn't exist

def upload_dataset_file(local_path, repo_id, repo_path, token) -> None
    # Uploads a file to the dataset repository

def download_dataset_file(filename, repo_id, token, local_dir) -> Path
    # Downloads a file from the dataset repository

def register_raw_engine_data_to_hf(token, repo_id) -> None
    # Convenience function to upload engine_data.csv
```

---

#### 9. `hf_model_utils.py` - Hugging Face Model Utilities
**Lines: 92 | Purpose: Upload/download models to/from HF Model Hub**

**Key functions:**
```python
def create_or_get_model_repo(repo_id, token, private) -> None
    # Creates model repo if it doesn't exist

def upload_model(local_model_path, repo_id, repo_path, token) -> None
    # Uploads trained model to HF Model Hub

def download_model(repo_id, filename, token, local_dir) -> object
    # Downloads and loads model from HF Model Hub
```

---

#### 10. `deploy_to_hf.py` - Hugging Face Space Deployment
**Lines: 109 | Purpose: Deploy Streamlit app to Hugging Face Space**

**What it does:**
1. Creates HF Space with Docker SDK if it doesn't exist
2. Uploads Space-specific README with metadata
3. Uploads all project files (excluding data, models, mlruns, .git)

**Key function:**
```python
def main():
    # Creates Space and uploads deployment files
```

**Usage:**
```bash
cd src
python deploy_to_hf.py
```

---

### Configuration Files

#### `Dockerfile`
**Lines: 22 | Purpose: Container definition for deployment**

```dockerfile
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
```

---

#### `requirements.txt`
**Lines: 10 | Purpose: Python package dependencies**

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning (Random Forest) |
| `mlflow` | Experiment tracking |
| `huggingface_hub` | HF Hub integration |
| `streamlit` | Web application framework |
| `joblib` | Model serialization |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualization |
| `plotly` | Interactive charts |

---

#### `.github/workflows/pipeline.yml`
**Lines: 114 | Purpose: CI/CD pipeline with GitHub Actions**

**Pipeline Jobs (sequential):**

| Job | Description | Depends On |
|-----|-------------|------------|
| `register-dataset` | Upload raw data to HF Dataset Hub | - |
| `data-prep` | Clean data and create train/test splits | register-dataset |
| `model-training` | Train model with MLflow tracking | data-prep |
| `deploy-hosting` | Deploy app to HF Space | model-training |

**Required GitHub Secrets:**
- `HF_TOKEN` - Hugging Face access token
- `HF_DATASET_REPO` - Dataset repository ID
- `HF_MODEL_REPO` - Model repository ID
- `HF_SPACE_REPO` - Space repository ID

---

## 📊 Dataset Description

### Engine Sensor Data (`data/engine_data.csv`)

The dataset contains sensor readings from engines with 7 columns:

| Column | Renamed To | Type | Range | Description |
|--------|-----------|------|-------|-------------|
| Engine rpm | Engine_RPM | Float | 0-4000 | Engine speed in revolutions per minute |
| Lub oil pressure | Lub_Oil_Pressure | Float | 0-10 | Lubricating oil pressure (bar) |
| Fuel pressure | Fuel_Pressure | Float | 0-30 | Fuel injection pressure (bar) |
| Coolant pressure | Coolant_Pressure | Float | 0-10 | Coolant system pressure (bar) |
| lub oil temp | Lub_Oil_Temperature | Float | 0-150 | Lubricating oil temperature (°C) |
| Coolant temp | Coolant_Temperature | Float | 0-150 | Coolant temperature (°C) |
| Engine Condition | Engine_Condition | Int | 0/1 | Target: 0=Normal, 1=Requires Maintenance |

### Data Statistics
- **Total records**: ~19,000 rows
- **Training set**: 80% (stratified)
- **Test set**: 20% (stratified)
- **Class distribution**: Balanced binary classification

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/UsamaMasood12/Engine-Failure-Classification.git
cd Engine-Failure-Classification

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set environment variables (optional, for HF integration)
set HF_TOKEN=your_huggingface_token
```

---

## 💻 Usage

### Run the Complete Pipeline

```bash
# Step 1: Register data to Hugging Face
cd src
python data_register.py

# Step 2: Prepare data (clean and split)
python data_prep.py

# Step 3: Run EDA (optional)
python eda.py

# Step 4: Train model
python train.py

# Step 5: Run web application
streamlit run app.py
```

### Run Only the Web Application

If you already have a trained model:

```bash
# Using local model
streamlit run src/app.py

# Using Hugging Face model (set HF_TOKEN first)
streamlit run src/app.py
```

### Make Predictions Programmatically

```python
from src.inference import predict_engine_condition

# Define sensor readings
sensor_data = {
    "Engine_RPM": 1500.0,
    "Lub_Oil_Pressure": 4.5,
    "Fuel_Pressure": 15.0,
    "Coolant_Pressure": 3.0,
    "Lub_Oil_Temperature": 85.0,
    "Coolant_Temperature": 88.0
}

# Get prediction
result = predict_engine_condition(sensor_data, source="local")
print(f"Prediction: {'Maintenance Required' if result['prediction'] == 1 else 'Normal'}")
print(f"Fault Probability: {result['probability_faulty']:.2%}")
```

---

## 🔄 MLOps Pipeline

### Pipeline Flow

```
┌──────────────────┐
│ 1. DATA REGISTER │
│  data_register.py │
│  → HF Dataset Hub │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. DATA PREP     │
│  data_prep.py    │
│  → Clean & Split │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. MODEL TRAIN   │
│  train.py        │
│  → MLflow Track  │
│  → HF Model Hub  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. DEPLOY        │
│  deploy_to_hf.py │
│  → HF Space      │
└──────────────────┘
```

### Experiment Tracking with MLflow

All training runs are logged to MLflow with:
- **Parameters**: All hyperparameter combinations tested
- **Metrics**: accuracy, precision, recall, f1
- **Artifacts**: Trained model file
- **Models**: Registered sklearn model

View experiments locally:
```bash
mlflow ui
# Open http://localhost:5000
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (for HF) | Hugging Face access token |
| `HF_DATASET_REPO` | Optional | Dataset repo ID (default in config.py) |
| `HF_MODEL_REPO` | Optional | Model repo ID (default in config.py) |
| `HF_SPACE_REPO` | Optional | Space repo ID (default in config.py) |
| `TEST_SIZE` | Optional | Test split ratio (default: 0.2) |

### Modify Configuration

Edit `src/config.py` to customize:
- Repository IDs
- Feature columns
- Model hyperparameters
- File paths

---

## 📖 API Reference

### Inference API

```python
predict_engine_condition(
    inputs: Dict[str, float],    # Sensor readings
    model: Optional[object],      # Pre-loaded model (optional)
    source: str = "local"         # "local" or "hf"
) -> Dict[str, float]             # {"prediction": 0/1, "probability_faulty": float}
```

### Model Input Format

```python
{
    "Engine_RPM": float,          # 0-4000
    "Lub_Oil_Pressure": float,    # 0-10
    "Fuel_Pressure": float,       # 0-30
    "Coolant_Pressure": float,    # 0-10
    "Lub_Oil_Temperature": float, # 0-150
    "Coolant_Temperature": float  # 0-150
}
```

---

## 🐳 Deployment

### Docker Deployment

```bash
# Build image
docker build -t engine-maintenance .

# Run container
docker run -p 7860:7860 -e HF_TOKEN=your_token engine-maintenance
```

### Hugging Face Spaces

The app auto-deploys via GitHub Actions. Manual deployment:

```bash
cd src
python deploy_to_hf.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Usama Masood**
- GitHub: [@UsamaMasood12](https://github.com/UsamaMasood12)
- Hugging Face: [UsamaMasood12ak](https://huggingface.co/UsamaMasood12ak)

---

## 🙏 Acknowledgments

- Scikit-learn for the machine learning framework
- Streamlit for the web application framework
- Hugging Face for model and dataset hosting
- MLflow for experiment tracking

---

<p align="center">
  <b>Built with ❤️ using Python, Scikit-learn, Streamlit & MLflow</b>
</p>
