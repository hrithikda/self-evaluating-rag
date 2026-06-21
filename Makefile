.PHONY: install ingest api app eval lint clean

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Data ─────────────────────────────────────────────────────────────────────
ingest:
	python data/ingest.py

ingest-fresh:
	python data/ingest.py --clear

# ── Servers ───────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app.py

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python -m evaluation.benchmark --subset all

eval-quick:
	python -m evaluation.benchmark --limit 5 --output evaluation/results_quick.json

eval-local:
	python -m evaluation.benchmark --subset local

eval-web:
	python -m evaluation.benchmark --subset web

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check . --fix

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf chroma_db __pycache__ .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
