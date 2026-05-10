#!/bin/sh
# Drop-privileges wrapper.
#
# When the container starts as root (default for Docker bind-mounts where
# the host directory may be owned by another UID/GID, or auto-created by
# Docker as root:root), this script makes /app/data writable by the
# `bot` user and re-execs the original command via `runuser -u bot`.
#
# When the container is started with --user already set to bot (or any
# non-root UID), this script simply execs the command unchanged.
#
# This keeps the bot running as a non-root user without forcing the
# operator to manually `chown` the host bind-mount.
set -e

if [ "$(id -u)" = "0" ]; then
    # The chown may fail on read-only mounts or when /app/data does not
    # exist for some unusual reason. Best-effort — do not abort startup.
    chown -R bot:bot /app/data 2>/dev/null || true
    exec runuser -u bot -- "$@"
fi

exec "$@"
