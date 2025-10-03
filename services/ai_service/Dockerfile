FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services ./services

ENV PYTHONPATH=/app

EXPOSE 5003

CMD ["python", "-m", "services.ai_service.app"]
