# 🎓 Student Exam Performance Prediction

An end-to-end Machine Learning project for predicting a student's **Math Score** based on demographic, educational, and academic features.

The project covers the complete machine learning workflow, including exploratory data analysis, preprocessing, model training, hyperparameter tuning, model evaluation, prediction pipelines, custom logging and exception handling, and deployment through a Flask web application.

---

## 📌 Project Overview

The main objective of this project is to predict a student's **Math Score** using information such as:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

The problem is formulated as a **supervised regression task**.

---

## 🎯 Problem Statement

Student academic performance can be influenced by several educational and demographic factors.

This project investigates these relationships through Exploratory Data Analysis and builds Machine Learning models capable of predicting a student's Math Score.

The target variable is:

```text
math_score
```

The input features are:

```text
gender
race_ethnicity
parental_level_of_education
lunch
test_preparation_course
reading_score
writing_score
```

---

## 📊 Dataset

The dataset contains approximately **1,000 student records** with the following columns:

| Feature | Description |
|---|---|
| `gender` | Student gender |
| `race_ethnicity` | Student race/ethnicity group |
| `parental_level_of_education` | Highest education level of parents |
| `lunch` | Lunch type |
| `test_preparation_course` | Test preparation course status |
| `math_score` | Math exam score |
| `reading_score` | Reading exam score |
| `writing_score` | Writing exam score |

The `math_score` column is used as the prediction target.

---

## 🔍 Exploratory Data Analysis

The EDA notebook investigates:

- Dataset shape and structure
- Missing values
- Duplicate records
- Data types
- Unique categories
- Statistical summaries
- Score distributions
- Gender-based performance
- Effect of lunch type
- Effect of parental education
- Test preparation performance
- Outlier analysis
- Relationships between Math, Reading, and Writing scores

Additional EDA features were created:

```text
total_score
average
```

These features were used only for exploratory analysis and are not part of the final prediction pipeline.

---

## 📈 Key EDA Insights

The analysis indicates that:

- Reading and Writing scores are strongly related to Math performance.
- Students with standard lunch generally show higher average academic scores in this dataset.
- Students who completed the test preparation course tend to achieve higher average scores.
- Parental education level shows associations with academic performance.
- Female students show higher average Reading and Writing scores in this dataset.
- Male students show a higher average Math score.
- Math, Reading, and Writing scores are positively correlated.

> These relationships should be interpreted as associations within this dataset and not necessarily as causal effects.

---

## 🤖 Machine Learning Models

Several regression algorithms were evaluated during experimentation:

```text
Linear Regression
Ridge Regression
Lasso Regression
K-Nearest Neighbors
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
CatBoost Regressor
AdaBoost Regressor
```

The modular training pipeline currently evaluates multiple models and performs hyperparameter tuning using `GridSearchCV`.

---

## 📏 Evaluation Metrics

The models are evaluated using the following metrics:

### R² Score

Measures how much of the variance in the target variable is explained by the model.

### Mean Absolute Error — MAE

Measures the average absolute prediction error.

### Root Mean Squared Error — RMSE

Measures prediction error while penalizing larger errors more strongly.

---

## 🏆 Model Performance

During notebook experimentation, the best-performing models achieved approximately:

| Model | Test R² |
|---|---:|
| Ridge Regression | ~0.881 |
| Linear Regression | ~0.880 |
| Random Forest | ~0.853 |
| CatBoost | ~0.852 |
| AdaBoost | ~0.846 |
| XGBoost | ~0.828 |
| Lasso | ~0.825 |
| KNN | ~0.784 |
| Decision Tree | ~0.759 |

The production training pipeline evaluates configured models and saves the best acceptable model automatically.

---

## 🏗️ Project Architecture

```text
ML-Project/
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── Notebook/
│   ├── Data/
│   │   └── stud.csv
│   │
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
│
├── src/
│   └── ml_project/
│       │
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── train_pipeline.py
│       │   └── predict_pipeline.py
│       │
│       ├── __init__.py
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── logs/
│
├── app.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .gitignore
└── README.md
```

---

# ⚙️ Machine Learning Pipeline

The project follows a modular training architecture.

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ├── train.csv
     ├── test.csv
     └── data.csv
     │
     ▼
Data Transformation
     │
     ├── Numerical Pipeline
     ├── Categorical Pipeline
     └── preprocessor.pkl
     │
     ▼
Model Training
     │
     ├── GridSearchCV
     ├── Model Evaluation
     └── Best Model Selection
     │
     ▼
model.pkl
```

---

## 1️⃣ Data Ingestion

`data_ingestion.py` is responsible for:

- Reading the original dataset
- Creating the artifacts directory
- Saving a copy of the raw dataset
- Splitting the dataset into training and testing sets
- Saving `train.csv` and `test.csv`

The data split is:

```text
80% Training
20% Testing
```

with:

```python
random_state=42
```

for reproducibility.

---

## 2️⃣ Data Transformation

`data_transformation.py` builds the preprocessing pipeline.

### Numerical Features

```text
reading_score
writing_score
```

Numerical preprocessing:

```text
Median Imputation
        ↓
Standard Scaling
```

### Categorical Features

```text
gender
race_ethnicity
parental_level_of_education
lunch
test_preparation_course
```

Categorical preprocessing:

```text
Most Frequent Imputation
        ↓
One-Hot Encoding
        ↓
Scaling
```

The fitted preprocessing object is saved as:

```text
artifacts/preprocessor.pkl
```

This allows the exact same preprocessing steps to be applied during prediction.

---

## 3️⃣ Model Training

`model_trainer.py` trains and evaluates multiple regression algorithms.

Hyperparameter tuning is performed using:

```python
GridSearchCV
```

with cross-validation.

The model with the highest acceptable test R² score is selected and saved as:

```text
artifacts/model.pkl
```

---

## 4️⃣ Training Pipeline

`train_pipeline.py` coordinates the complete training workflow:

```text
DataIngestion
      ↓
DataTransformation
      ↓
ModelTrainer
      ↓
Final Model
```

Run the complete training pipeline with:

```bash
PYTHONPATH=src uv run python -m ml_project.pipeline.train_pipeline
```

---

# 🔮 Prediction Pipeline

The prediction pipeline is implemented in:

```text
src/ml_project/pipeline/predict_pipeline.py
```

It contains two main classes:

### `CustomData`

Receives input data from the user and converts it into a Pandas DataFrame.

### `PredictPipeline`

Loads:

```text
artifacts/preprocessor.pkl
artifacts/model.pkl
```

and performs:

```text
User Input
     ↓
Pandas DataFrame
     ↓
Preprocessor
     ↓
Transformed Features
     ↓
Trained Model
     ↓
Predicted Math Score
```

---

# 🌐 Flask Web Application

A Flask web application is included to allow users to interact with the trained model through a browser.

The user provides:

```text
Gender
Race / Ethnicity
Parental Education
Lunch Type
Test Preparation Course
Reading Score
Writing Score
```

The application then returns the predicted Math Score.

Application flow:

```text
HTML Form
    ↓
Flask
    ↓
CustomData
    ↓
PredictPipeline
    ↓
Preprocessor
    ↓
ML Model
    ↓
Predicted Math Score
    ↓
Browser
```

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

Move into the project directory:

```bash
cd ML-Project
```

---

# 📦 Environment Setup with uv

This project uses `uv` for Python environment and dependency management.

If `uv` is already installed:

```bash
uv sync
```

To create a virtual environment manually:

```bash
uv venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## Install Dependencies

Using `uv`:

```bash
uv sync
```

Alternatively, using `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run the complete training pipeline:

```bash
PYTHONPATH=src uv run python -m ml_project.pipeline.train_pipeline
```

After successful training, the following files are generated:

```text
artifacts/
├── data.csv
├── train.csv
├── test.csv
├── preprocessor.pkl
└── model.pkl
```

---

# 🌍 Run the Flask Application

Run:

```bash
PYTHONPATH=src uv run python app.py
```

Then open the application in your browser:

```text
http://127.0.0.1:5000
```

If port `5000` is already in use, the Flask application can be configured to run on another port such as:

```python
app.run(host="0.0.0.0", port=5001, debug=True)
```

Then open:

```text
http://127.0.0.1:5001
```

---

# 🛠️ Technologies Used

### Programming

```text
Python
```

### Data Analysis

```text
Pandas
NumPy
```

### Data Visualization

```text
Matplotlib
Seaborn
```

### Machine Learning

```text
Scikit-learn
XGBoost
CatBoost
```

### Model Optimization

```text
GridSearchCV
Cross Validation
```

### Backend

```text
Flask
```

### Environment & Dependency Management

```text
uv
```

### Version Control

```text
Git
GitHub
```

---

# 🧩 Additional Engineering Features

The project also includes:

## Custom Exception Handling

`exception.py` provides detailed error messages containing:

```text
File name
Line number
Error description
```

This makes debugging easier across different components of the Machine Learning pipeline.

---

## Logging

`logger.py` automatically creates log files for tracking each stage of the project.

Example logged events include:

```text
Data ingestion started
Dataset loaded
Train/test split completed
Data transformation started
Preprocessor saved
Model training started
Best model selected
Training completed
```

---

## Utility Functions

`utils.py` contains reusable functionality for:

```text
Saving Python objects
Loading trained objects
Model evaluation
Hyperparameter tuning
```

Important utility functions include:

```python
save_object()
load_object()
evaluate_models()
```

---

# 📂 Generated Artifacts

The application generates reusable Machine Learning artifacts.

## `preprocessor.pkl`

Stores the fitted preprocessing pipeline.

It contains the transformations required to convert raw input data into the format expected by the model.

The prediction pipeline loads this object so that incoming user data goes through exactly the same transformations used during model training.

---

## `model.pkl`

Stores the final trained regression model.

The prediction pipeline loads this model and uses it to generate Math Score predictions without retraining the model.

---

# 🔄 End-to-End Workflow

```text
                           TRAINING PIPELINE

Raw Student Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Ingestion
        │
        ▼
Train / Test Split
        │
        ▼
Data Transformation
        │
        ├────────────► preprocessor.pkl
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        └────────────► model.pkl


                          PREDICTION PIPELINE

User Input
    │
    ▼
HTML Form
    │
    ▼
Flask Application
    │
    ▼
CustomData
    │
    ▼
Pandas DataFrame
    │
    ▼
preprocessor.pkl
    │
    ▼
Feature Transformation
    │
    ▼
model.pkl
    │
    ▼
Math Score Prediction
    │
    ▼
Prediction Displayed in Browser
```

---

# 🧪 Example Prediction Flow

Suppose a user enters:

```text
Gender: Female
Race/Ethnicity: Group C
Parental Education: Bachelor's Degree
Lunch: Standard
Test Preparation: Completed
Reading Score: 82
Writing Score: 78
```

The following process occurs:

```text
User Form
   ↓
Flask receives POST request
   ↓
CustomData object
   ↓
Pandas DataFrame
   ↓
preprocessor.pkl
   ↓
Feature transformation
   ↓
model.pkl
   ↓
Prediction
   ↓
Predicted Math Score displayed in browser
```

---

# 🧠 Modular Design

The project separates responsibilities into independent modules.

### Components

```text
data_ingestion.py
data_transformation.py
model_trainer.py
```

Each component is responsible for one stage of the training workflow.

### Pipelines

```text
train_pipeline.py
predict_pipeline.py
```

The pipeline modules coordinate the individual components.

### Supporting Modules

```text
exception.py
logger.py
utils.py
```

These modules provide reusable functionality for exception handling, logging, serialization, and evaluation.

This modular architecture makes the project easier to maintain, debug, test, and extend.

---

# 🔮 Future Improvements

Possible improvements include:

- Add automated unit tests
- Add model experiment tracking
- Add MLflow integration
- Add Docker support
- Add CI/CD with GitHub Actions
- Deploy the Flask application to a cloud platform
- Add REST API endpoints
- Improve the web interface
- Add prediction input validation
- Add model versioning
- Add feature importance analysis
- Add automated retraining
- Add monitoring for model performance and data drift
- Add cloud deployment using AWS, Azure, or another platform

---

# 📚 What This Project Demonstrates

This project demonstrates practical knowledge of:

```text
Exploratory Data Analysis
Data Cleaning
Feature Engineering
Feature Transformation
Machine Learning Regression
Model Comparison
Cross Validation
Hyperparameter Tuning
Modular Python Programming
ML Pipeline Design
Logging
Exception Handling
Model Serialization
Flask Integration
Web-based Model Inference
Git/GitHub
uv Dependency Management
```

---

# 👨‍💻 Author

**Hamed Goldoust**

Data Science & Machine Learning

GitHub: [Clonerhamed](https://github.com/Clonerhamed)

---

## ⭐ Support

If you find this project useful, feel free to give the repository a ⭐ on GitHub.
