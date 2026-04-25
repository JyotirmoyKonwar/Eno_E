FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
COPY src ./src
COPY outputs ./outputs
COPY README.md ./
COPY .env.example ./

ENV PYTHONPATH=/app/src
EXPOSE 7860

CMD ["python", "app.py"]
