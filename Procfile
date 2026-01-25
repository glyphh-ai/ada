web: python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
release: alembic -c alembic.ini upgrade head
