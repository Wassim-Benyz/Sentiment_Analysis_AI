PYTHON := python3

.PHONY: preprocess
preprocess:
	$(PYTHON) src/preprocessing/clean_text.py

.PHONY: docker-build docker-run
docker-build:
	docker build -t sentiment-api .

docker-run:
	docker run --rm -p 5000:5000 sentiment-api
