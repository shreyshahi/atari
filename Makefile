.PHONY: train evaluate video dashboard test lint fmt

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py

video:
	uv run python scripts/record_video.py

dashboard:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
