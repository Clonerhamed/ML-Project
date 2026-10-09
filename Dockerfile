# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Install dependencies and train ML
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    VIRTUAL_ENV=/opt/venv

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY Notebook/Data/stud.csv ./Notebook/Data/stud.csv

# Train the model inside the Docker build
RUN python -m ml_project.pipeline.train_pipeline \
    && test -s artifacts/model.pkl \
    && test -s artifacts/preprocessor.pkl


# ==========================================
# Stage 2: Runtime image
# ==========================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    VIRTUAL_ENV=/opt/venv

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Reuse exactly the environment used to train the model
COPY --from=builder /opt/venv /opt/venv

COPY app.py ./app.py
COPY src/ ./src/
COPY templates/ ./templates/

RUN mkdir -p /app/artifacts

COPY --from=builder /app/artifacts/model.pkl /app/artifacts/model.pkl
COPY --from=builder /app/artifacts/preprocessor.pkl /app/artifacts/preprocessor.pkl

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "600", "app:app"]