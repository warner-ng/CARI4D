#!/usr/bin/env bash
set -euo pipefail

TARGET_DATA_ROOT="/mnt/projects_ext4/docker-data"
DAEMON_JSON="/etc/docker/daemon.json"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/install-docker-external.sh"
  exit 1
fi

if [[ ! -d /mnt/projects_ext4 ]]; then
  echo "ERROR: /mnt/projects_ext4 not found. Please mount /dev/sdb1 first."
  exit 1
fi

echo "[1/8] Installing Docker..."
apt update
apt install -y docker.io

echo "[2/8] Enabling and starting Docker service..."
systemctl enable --now docker

echo "[3/8] Preparing external Docker data root: ${TARGET_DATA_ROOT}"
mkdir -p "${TARGET_DATA_ROOT}"
chmod 711 /mnt/projects_ext4 || true
chown root:root "${TARGET_DATA_ROOT}"
chmod 710 "${TARGET_DATA_ROOT}"

echo "[4/8] Writing Docker daemon config..."
mkdir -p /etc/docker
cat > "${DAEMON_JSON}" <<EOF
{
  "data-root": "${TARGET_DATA_ROOT}"
}
EOF

echo "[5/8] Restarting Docker..."
systemctl restart docker

echo "[6/8] Adding invoking user to docker group..."
if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "${SUDO_USER}"
  USER_HOME=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
  ln -sfn "${TARGET_DATA_ROOT}" "${USER_HOME}/docker-data"
  chown -h "${SUDO_USER}":"${SUDO_USER}" "${USER_HOME}/docker-data" || true
  echo "Symlink created: ${USER_HOME}/docker-data -> ${TARGET_DATA_ROOT}"
else
  echo "WARNING: SUDO_USER is empty; skipped usermod and home symlink."
fi

echo "[7/8] Verifying docker root dir..."
docker info 2>/dev/null | grep -E "Docker Root Dir|Server Version" || true

echo "[8/8] Done."
echo "IMPORTANT: log out and log back in (or reboot) so docker group permission takes effect."
