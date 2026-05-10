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

# The entrypoint chowns /app/data at runtime (necessary for bind-mounted
# host directories owned by a different UID/GID) and then drops privileges
# from root to the `bot` user via runuser. `runuser` ships with the
# util-linux package, which is already present in python:*-slim.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# NOTE: no `USER bot` here — the entrypoint starts as root so it can fix
# ownership of /app/data, then runuser drops privileges before exec'ing
# the bot. To skip the entrypoint and run as a fixed user (for example
# when /app/data is provisioned externally with the right owner), pass
# `--user 1000:1000` to docker run / docker compose.

# Liveness probe: only verifies the Python interpreter still works inside
# the container. The bot uses long-poll, so there is no HTTP endpoint to
# hit.
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py"]
