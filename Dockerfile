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

# Копируем ровно то, что нужно рантайму, а не `COPY . .`: белый список нельзя
# забыть обновить при появлении нового файла с данными в корне репозитория, а
# чёрный список в .dockerignore — можно (и однажды уже забыли, см. PR #101).
# app/ включает app/migrations/*.sql — их читает мигратор на старте.
COPY --chown=bot:bot main.py ./
COPY --chown=bot:bot app ./app

# Идентичность сборки — первая строка /stats. Нужна не для красоты:
# `docs/OPERATIONS.md` предписывает различать «счётчика нет, потому что нечего
# показать» и «счётчика нет в выкаченной сборке», а различать было нечем, и один
# раз это уже стоило недели (рестарт 14.08.2026 увёз сборку старше M3R-140,
# заметили 21.08 по отсутствию строки в /stats).
#
# Штамп времени, а не ревизия: `.git` в контекст сборки не уходит и уходить не
# должен (белый список — тот же, что держит переписку чата вне слоёв), поэтому
# ревизию пришлось бы передавать снаружи и помнить про неё при каждом деплое.
# Время отвечает на тот самый вопрос, который провалился: «сборка новее коммита
# X или нет» — сверкой с `git log -1 --format=%cI`.
#
# Строка стоит **сразу после** COPY выше, и это не косметика: кэш слоя
# обновляет штамп тогда и только тогда, когда изменилось содержимое `main.py`
# или `app/`. Правка `compose.yaml` или доков оставит прежний штамп — верно по
# смыслу, рантайм в образе действительно тот же.
RUN date -Is > /app/BUILD_AT

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
# the container and the DB file is readable. The bot uses long-poll, so
# there is no HTTP endpoint to hit. The connection is opened read-only via
# a URI (audit N6): the healthcheck runs as root (no USER directive — the
# entrypoint drops privileges only for the bot process), and a plain
# sqlite3.connect would create a root-owned DB file if it were missing.
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import sqlite3, os; sqlite3.connect('file:' + os.getenv('DB_PATH', 'data/markov.db') + '?mode=ro', uri=True).execute('SELECT 1')"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py"]
