#!/bin/sh
set -e

exit_backend() {
  echo "Exiting application..."
  exit 0
}

trap exit_backend INT TERM

# Hang out until SIGTERM received
while true; do
    sleep 1
done