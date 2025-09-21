.PHONY: setup emb serve rank test clean docker-build docker-run

setup:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt')"

emb:
	python -m src.embeddings --build

serve:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

rank:
	python -m src.ranker --train

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t matchmaking-he .

docker-run:
	docker run -p 8000:8000 matchmaking-he

dev:
	python -m src.data_generator --generate-sample
	make emb
	make serve
