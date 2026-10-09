# 🎓 Student Exam Performance Prediction

An end-to-end Machine Learning project that predicts a student's **Math
Score** from demographic, educational, and academic features. It
includes exploratory data analysis, preprocessing, model training and
tuning, a reusable prediction pipeline, a Flask web app, Docker,
automated tests, and CI/CD deployment to Microsoft Azure.

-   **Live application:**
    https://hamed-ml-project-gyerdccbf6embngk.italynorth-01.azurewebsites.net/
-   **Prediction form:**
    https://hamed-ml-project-gyerdccbf6embngk.italynorth-01.azurewebsites.net/predictdata
-   **Repository:** https://github.com/Clonerhamed/ML-Project

------------------------------------------------------------------------

## Project Overview

The model predicts `math_score` using:

-   `gender`
-   `race_ethnicity`
-   `parental_level_of_education`
-   `lunch`
-   `test_preparation_course`
-   `reading_score`
-   `writing_score`

This is a supervised regression task. The source dataset is
`Notebook/Data/stud.csv` and contains approximately 1,000 student
records.

## Exploratory Data Analysis

The EDA notebooks explore dataset structure, missing values, duplicates,
categories, statistical summaries, score distributions, outliers, and
relationships between Math, Reading, and Writing scores. They also
compare scores by gender, lunch type, parental education, and
test-preparation status.

The notebooks create `total_score` and `average` for exploratory
analysis only; these features are not part of the production prediction
pipeline.

The analysis found associations between Reading/Writing and Math scores,
as well as differences across some demographic and educational groups.
These are associations in this dataset, not proof of causation.

## Machine Learning

The notebook experiments include Linear Regression, Ridge, Lasso, KNN,
Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, and
AdaBoost regressors. The training pipeline compares configured models
and uses `GridSearchCV` for hyperparameter tuning.

Evaluation metrics:

-   **R²:** variance in the target explained by the model.
-   **MAE:** mean absolute prediction error.
-   **RMSE:** prediction error that penalizes larger errors more
    strongly.

Approximate test R² scores from earlier notebook experiments:

  Model                 Test R²
  ------------------- ---------
  Ridge Regression      \~0.881
  Linear Regression     \~0.880
  Random Forest         \~0.853
  CatBoost              \~0.852
  AdaBoost              \~0.846
  XGBoost               \~0.828
  Lasso                 \~0.825
  KNN                   \~0.784
  Decision Tree         \~0.759

These values describe the notebook experiments and are not a guarantee
of the score from every later training run.

## Project Structure

``` text
ML-Project/
├── .github/workflows/ci-cd.yml
├── Notebook/Data/stud.csv
├── Notebook/1. EDA STUDENT PERFORMANCE.ipynb
├── Notebook/2. MODEL TRAINING.ipynb
├── src/ml_project/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── tests/test_app.py
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

The `artifacts/` directory is generated during training or Docker image
construction. It contains the ingested data files, `preprocessor.pkl`,
and `model.pkl`. Generated artifacts do not need to be committed to Git.

> **Important:** Keep the template folder named `templates` in
> lowercase. Linux containers are case-sensitive, so `Templates` and
> `templates` are different paths.

## Machine Learning Pipeline

``` text
Notebook/Data/stud.csv
        ↓
Data Ingestion
        ├── data.csv
        ├── train.csv
        └── test.csv
        ↓
Data Transformation
        ├── Numerical preprocessing
        ├── Categorical preprocessing
        └── preprocessor.pkl
        ↓
Model Training + GridSearchCV
        ↓
Model evaluation and selection
        ↓
model.pkl
```

### Data ingestion

`src/ml_project/components/data_ingestion.py` reads the dataset, creates
the artifacts directory, and creates training/test splits. The split is
80% training and 20% testing, using `random_state=42` for
reproducibility.

### Data transformation

`src/ml_project/components/data_transformation.py` builds the fitted
preprocessor.

-   **Numerical features:** `reading_score`, `writing_score`; median
    imputation followed by standard scaling.
-   **Categorical features:** gender, race/ethnicity, parental
    education, lunch, and test-preparation status; most-frequent
    imputation followed by one-hot encoding.

The fitted preprocessor is saved to `artifacts/preprocessor.pkl`.

### Model training

`src/ml_project/components/model_trainer.py` evaluates configured
regressors, tunes hyperparameters with `GridSearchCV`, and saves the
selected model to `artifacts/model.pkl`.

`src/ml_project/pipeline/train_pipeline.py` coordinates ingestion,
transformation, and training.

## Prediction Pipeline and Flask App

`src/ml_project/pipeline/predict_pipeline.py` contains:

-   **`CustomData`:** converts the submitted form values into a Pandas
    DataFrame.
-   **`PredictPipeline`:** loads `preprocessor.pkl` and `model.pkl`,
    transforms input data, and predicts a Math Score.

`app.py` serves the web interface:

-   `/` renders `templates/index.html`.
-   `/predictdata` renders the form in `templates/home.html` and handles
    prediction submissions.

The prediction flow is:

``` text
HTML form → Flask → CustomData → DataFrame
          → preprocessor.pkl → model.pkl
          → predicted Math Score → browser
```

The project also includes custom exception handling in `exception.py`,
file-based logging in `logger.py`, and reusable serialization/evaluation
helpers in `utils.py`.

------------------------------------------------------------------------

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/Clonerhamed/ML-Project.git
cd ML-Project
```

### 2. Install dependencies

With `uv`:

``` bash
uv sync
```

Alternatively, create a virtual environment and install
`requirements.txt`:

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model

With `uv`:

``` bash
PYTHONPATH=src uv run python -m ml_project.pipeline.train_pipeline
```

Or with the activated environment:

``` bash
PYTHONPATH=src python -m ml_project.pipeline.train_pipeline
```

### 4. Run Flask

``` bash
PYTHONPATH=src uv run python app.py
```

Open http://127.0.0.1:5001. The local Flask development server uses port
`5001`; the production container runs Gunicorn on port `8000`.

## Docker

The multi-stage `Dockerfile` installs dependencies and trains the model
in its builder stage. It checks that `artifacts/model.pkl` and
`artifacts/preprocessor.pkl` were created, then copies the runtime
environment, Flask app, templates, source code, and artifacts into a
runtime image. Gunicorn listens on port `8000`.

Build the image:

``` bash
docker build --platform linux/amd64 -t student-performance:local .
```

Run it:

``` bash
docker run --rm -p 8000:8000 student-performance:local
```

Open http://localhost:8000 or http://localhost:8000/predictdata.

The model is trained during image construction, not for every prediction
request. `.dockerignore` excludes local-only content such as the virtual
environment, Git metadata, logs, notebooks, tests, and locally generated
artifacts from the build context.

## Automated Tests

`tests/test_app.py` uses Flask's test client to check that `GET /` and
`GET /predictdata` return HTTP 200.

Run locally:

``` bash
PYTHONPATH=.:src python -m pytest -q
```

The GitHub Actions test job also runs a Python syntax check using
`compileall`.

## CI/CD with GitHub Actions

Workflow: `.github/workflows/ci-cd.yml`.

It runs on pushes to `main` and pull requests targeting `main`.

### Test job

1.  Checks out the repository.
2.  Sets up Python 3.11.
3.  Installs `requirements.txt` and `pytest`.
4.  Checks Python syntax.
5.  Runs the tests.

### Build and deploy job

For a push to `main`, after tests pass, GitHub Actions:

1.  Logs in to Azure Container Registry (ACR).
2.  Builds a Docker image for `linux/amd64`.
3.  Tags the image with the Git commit SHA.
4.  Pushes the image to ACR.
5.  Deploys the image to Azure App Service using `azure/webapps-deploy`.

``` text
Push to main → tests → Docker build → push to ACR → deploy to Azure App Service
```

Pull requests run tests; deployment is restricted to pushes to `main`.

## Azure Deployment

The Flask application is deployed as a container to Azure App Service,
with its Docker image stored in Azure Container Registry.

-   **App Service name:** `hamed-ml-project`
-   **Live URL:**
    https://hamed-ml-project-gyerdccbf6embngk.italynorth-01.azurewebsites.net/
-   **Container port:** `8000`
-   **ACR repository:** `student-performance`

### GitHub repository variables

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `AZURE_WEBAPP_NAME`                 App Service resource name only:
                                      `hamed-ml-project`, not the full
                                      domain

  `ACR_LOGIN_SERVER`                  ACR login server, for example
                                      `yourregistry.azurecr.io`

  `ACR_REPOSITORY`                    Image repository name, for example
                                      `student-performance`
  -----------------------------------------------------------------------

### GitHub repository secrets

  -----------------------------------------------------------------------
  Secret                              Purpose
  ----------------------------------- -----------------------------------
  `ACR_USERNAME`                      Username used to authenticate to
                                      ACR

  `ACR_PASSWORD`                      Password used to authenticate to
                                      ACR

  `AZURE_WEBAPP_PUBLISH_PROFILE`      Publish profile XML for the target
                                      App Service
  -----------------------------------------------------------------------

Keep these credentials private and never commit them to the repository.
For production environments, consider moving from long-lived secrets to
an identity-based approach such as GitHub Actions OIDC, where supported.

The App Service must be configured to pull the image from ACR and listen
on port `8000`. If the site does not start, inspect the App Service
container settings and logs, along with the GitHub Actions deployment
logs.

## Technologies Used

-   Python, Pandas, NumPy
-   Matplotlib, Seaborn
-   Scikit-learn, XGBoost, CatBoost
-   GridSearchCV and cross-validation
-   Flask, Jinja templates, Gunicorn
-   `uv`, `pip`
-   Docker
-   Git, GitHub, GitHub Actions, pytest
-   Azure Container Registry (ACR), Azure App Service

## Future Improvements

-   Add tests for submitted form values and prediction results.
-   Improve input validation and user-facing error messages.
-   Add experiment tracking and model versioning, for example with
    MLflow.
-   Monitor application health, model performance, and data drift.
-   Add API documentation and a repeatable model-retraining process.
-   Prefer identity-based CI/CD authentication over long-lived
    credentials where possible.

## Author

**Hamed Goldoust** --- Data Science & Machine Learning

-   GitHub: [Clonerhamed](https://github.com/Clonerhamed)
-   LinkedIn: [Hamed
    Goldoust](https://www.linkedin.com/in/hamed-goldoust/)

If you find this project useful, feel free to give the repository a ⭐.
