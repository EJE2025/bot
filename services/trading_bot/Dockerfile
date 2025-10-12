FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY trading_bot ./trading_bot
COPY data ./data
COPY models ./models
COPY README.md ./README.md

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "trading_bot.bot"]
