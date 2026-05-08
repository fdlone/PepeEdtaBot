FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN mkdir -p /app/data

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

CMD ["python", "main.py"]
