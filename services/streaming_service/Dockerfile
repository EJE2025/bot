FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services ./services

ENV PYTHONPATH=/app

EXPOSE 7000

CMD ["uvicorn", "services.streaming_service.app:app", "--host", "0.0.0.0", "--port", "7000"]
