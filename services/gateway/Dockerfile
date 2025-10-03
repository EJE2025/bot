FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services ./services

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "services.gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
