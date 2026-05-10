# Pin to a specific minor Python release for reproducibility.
# Bump deliberately when upgrading Python.
FROM python:3.14.0-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 1000 bot \
    && mkdir -p /app/data \
    && chown -R bot:bot /app

COPY --chown=bot:bot requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY --chown=bot:bot . .

USER bot

# Liveness probe: only verifies the Python interpreter still works inside the
# container. The bot uses long-poll, so there is no HTTP endpoint to hit.
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "main.py"]
