FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY data/sample_docs /app/data/sample_docs

CMD ["python", "-m", "scripts.run_api_docker"]
