.PHONY: up down test lint migrate benchmark

up:
	docker compose up -d

down:
	docker compose down

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

migrate:
	docker compose exec postgres psql -U ragops -d ragops \
		-f /docker-entrypoint-initdb.d/001_init.sql

benchmark:
	python scripts/run_benchmark.py
