#!/usr/bin/env bash
set -uo pipefail

report_error() {
  local code="${1:-1}"
  shift || true
  local msg="${*:-unknown error}"
  echo "[ERROR][${code}] ${msg}" >&2
}

run_or_report() {
  "$@"
  local code=$?
  if [[ $code -ne 0 ]]; then
    report_error "$code" "cmd failed: $*"
  fi
  return 0
}

TARGET_DATA_ROOT="/mnt/projects_ext4/docker-data"
DAEMON_JSON="/etc/docker/daemon.json"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/install-docker-external.sh"
  report_error 1 "Root privilege required"
fi

if [[ ! -d /mnt/projects_ext4 ]]; then
  echo "ERROR: /mnt/projects_ext4 not found. Please mount /dev/sdb1 first."
  report_error 1 "/mnt/projects_ext4 not found"
fi

echo "[1/8] Installing Docker..."
run_or_report apt update
run_or_report apt install -y docker.io

echo "[2/8] Enabling and starting Docker service..."
run_or_report systemctl enable --now docker

echo "[3/8] Preparing external Docker data root: ${TARGET_DATA_ROOT}"
run_or_report mkdir -p "${TARGET_DATA_ROOT}"
run_or_report chmod 711 /mnt/projects_ext4
run_or_report chown root:root "${TARGET_DATA_ROOT}"
run_or_report chmod 710 "${TARGET_DATA_ROOT}"

echo "[4/8] Writing Docker daemon config..."
run_or_report mkdir -p /etc/docker
cat > "${DAEMON_JSON}" <<EOF
{
  "data-root": "${TARGET_DATA_ROOT}"
}
EOF

echo "[5/8] Restarting Docker..."
run_or_report systemctl restart docker

echo "[6/8] Adding invoking user to docker group..."
if [[ -n "${SUDO_USER:-}" ]]; then
  run_or_report usermod -aG docker "${SUDO_USER}"
  USER_HOME=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
  run_or_report ln -sfn "${TARGET_DATA_ROOT}" "${USER_HOME}/docker-data"
  run_or_report chown -h "${SUDO_USER}":"${SUDO_USER}" "${USER_HOME}/docker-data"
  echo "Symlink created: ${USER_HOME}/docker-data -> ${TARGET_DATA_ROOT}"
else
  echo "WARNING: SUDO_USER is empty; skipped usermod and home symlink."
fi

echo "[7/8] Verifying docker root dir..."
run_or_report sh -c 'docker info 2>/dev/null | grep -E "Docker Root Dir|Server Version"'

echo "[8/8] Done."
echo "IMPORTANT: log out and log back in (or reboot) so docker group permission takes effect."
