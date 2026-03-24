# Sentiment Analysis Project

This project builds a sentiment analysis pipeline for Amazon food reviews.
It preprocesses review text, trains a TF-IDF + Logistic Regression model, and serves predictions with a Flask API.

## Dataset

Source: Kaggle, Amazon Fine Food Reviews

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### 1. Preprocess

```bash
python src/preprocessing/clean_text.py
```

### 2. Train

```bash
python src/models/train.py
```

### 3. Predict

```bash
python src/models/predict.py "This product is really good"
```

### 4. Run API

```bash
python api/app.py
```

## API Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is really good"}'
```

## Project Structure

```text
sentiment-ai-project/
├── api/
│   └── app.py
├── data/
├── models/
├── scripts/
├── src/
│   ├── models/
│   │   ├── predict.py
│   │   └── train.py
│   └── preprocessing/
│       └── clean_text.py
├── utils/
│   └── text_cleaning.py
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```
