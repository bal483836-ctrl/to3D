.PHONY: help install dev test docker-up docker-down clean

help:
	@echo "make install     创建 venv 并安装依赖(含开发依赖)"
	@echo "make dev         本地启动(热重载, http://127.0.0.1:8000)"
	@echo "make test        运行测试"
	@echo "make docker-up   构建并启动容器(http://127.0.0.1:8000)"
	@echo "make docker-down 停止容器"

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r backend/requirements-dev.txt

dev:
	. .venv/bin/activate && cd backend && uvicorn app.main:app --reload --port 8000

test:
	. .venv/bin/activate && cd backend && python -m pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .venv backend/.pytest_cache **/__pycache__
