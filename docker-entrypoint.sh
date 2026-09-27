#!/bin/sh
# Runs the container's command as the user `app` (uid 1000 in the image).
# DATA_DIR is data/ bind-mounted from the host, and the app writes uploads, transcripts and the manifest there. On Linux,
# a folder owned by another user isn't writable by uid 1000, so `app` first takes the folder owner's uid and gid: uploads
# work, and what the app writes has the same owner as the rest of data/. `docker exec -u app` runs commands the same way.
set -e

if [ "$(id -u)" != 0 ]; then
  exec "$@"  # started with --user: run as that user, as is
fi

data=${DATA_DIR:-data}
if [ -d "$data" ] && ! setpriv --reuid=app --regid=app --init-groups test -w "$data"; then
  uid=$(stat -c %u "$data") gid=$(stat -c %g "$data")
  [ "$gid" = "$(id -g app)" ] || groupmod --non-unique --gid "$gid" app
  [ "$uid" = "$(id -u app)" ] || usermod --non-unique --uid "$uid" app  # also re-owns app's files in /home/app
fi
# the model cache volume, if another user wrote to it (a root `docker exec`, or a folder owner that changed)
find /home/app/.cache ! -user app -exec chown -h app:app {} + 2>/dev/null || echo "docker-entrypoint.sh: couldn't re-own the model cache" >&2
exec setpriv --reuid=app --regid=app --init-groups -- "$@"
