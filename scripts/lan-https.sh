#!/usr/bin/env bash
# Generates a self-signed HTTPS cert for this Mac's current LAN IP, then starts the server over HTTPS
# on 0.0.0.0:8443 so a device on the same Wi-Fi (e.g. a Quest headset) can reach it.
#
# WebXR requires a secure origin. A self-signed cert satisfies the browser's "https" scheme check;
# the Quest browser will show a one-time "connection not private" warning to click through, with no
# flags or special config needed on the Quest side.
set -euo pipefail
cd "$(dirname "$0")/.."

IP="${1:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}"
if [ -z "$IP" ]; then
  echo "Could not detect a LAN IP. Pass it explicitly: scripts/lan-https.sh 192.168.1.x" >&2
  exit 1
fi

mkdir -p certs
CERT=certs/lan.crt
KEY=certs/lan.key

if [ ! -f "$CERT" ] || ! openssl x509 -in "$CERT" -noout -checkend 0 -ext subjectAltName 2>/dev/null | grep -q "IP Address:$IP"; then
  echo "Generating a self-signed certificate for $IP ..."
  openssl req -x509 -newkey rsa:2048 -sha256 -days 30 -nodes \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=$IP" \
    -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost"
else
  echo "Reusing existing certificate for $IP."
fi

echo
echo "=============================================================="
echo " On the Quest headset (same Wi-Fi), open:"
echo "   https://$IP:8443"
echo " Accept the one-time certificate warning (Advanced -> Proceed)."
echo "=============================================================="
echo

uv run uvicorn server.app:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT"
