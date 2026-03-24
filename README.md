 
# Sentiment AI — Short Overview

What this project does
- Downloads Amazon food reviews, cleans the text, and trains a simple sentiment model (positive / neutral / negative).
- Provides a small web API to send a review and receive a predicted sentiment and confidence.

Quick, non-technical steps
1) Get the data
- We use a public Kaggle dataset. If you have Kaggle set up, download the dataset into the `data/` folder and unzip it.

2) Preprocess the data
- Run the provided preprocessing step to clean text and create a processed CSV.

3) Train the model
- Run the training script to produce model files in the `models/` folder.

4) Run the API
- Start the small web service and POST a piece of text to get a sentiment back.

How to test quickly
- There are helper scripts and a Makefile so you can run the above steps with minimal typing. See the project root for `make preprocess`, `make docker-build`, and `make docker-run`.

Notes for non-developers
- If you prefer not to run anything locally, tell me and I can (a) package the model and a short demo, or (b) show screenshots of sample outputs.
- If you need help with any step (installing Docker, running the API, or downloading the data), tell me which OS and I will give exact one-line instructions.

Files to look at
- `api/app.py` — web API
- `src/preprocessing/clean_text.py` — data cleaning
- `src/models/train.py` — training and evaluation

Want this as a short slide or a one-paragraph email to a colleague? I can write that next.
