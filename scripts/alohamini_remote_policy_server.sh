#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 REMOTE_SSH_PORT" >&2
    exit 2
fi

ssh_port=$1
remote_host=${ALOHAMINI_REMOTE_HOST:-connect.bjb1.seetacloud.com}
remote_user=${ALOHAMINI_REMOTE_USER:-root}
remote_repo=${ALOHAMINI_REMOTE_REPO:-/root/autodl-tmp/pi_train/repos/lerobot_alohamini}
remote_python=${ALOHAMINI_REMOTE_PYTHON:-/root/autodl-tmp/pi_train/envs/lerobot/bin/python}
remote_hf_home=${ALOHAMINI_REMOTE_HF_HOME:-/root/autodl-tmp/pi_train/hf_cache}
remote_log=${ALOHAMINI_REMOTE_LOG:-/root/autodl-tmp/pi_train/logs/policy_server_pi_direct.log}
server_port=${ALOHAMINI_REMOTE_GRPC_PORT:-8080}
fps=${ALOHAMINI_FPS:-25}

remote_script=$(printf '%q ' \
    env \
    "PYTHONPATH=${remote_repo}/src" \
    "HF_HOME=${remote_hf_home}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${remote_python}" \
    -u \
    -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    "--port=${server_port}" \
    "--fps=${fps}" \
    --inference_latency=0 \
    --obs_queue_timeout=10)

# The server is intentionally loopback-only.  The Raspberry Pi reaches it via
# alohamini_pi_direct_tunnel.sh.
ssh -p "${ssh_port}" "${remote_user}@${remote_host}" \
    "mkdir -p '$(dirname "${remote_log}")'; \
     if test -s '${remote_log}.pid' && kill -0 \"\$(cat '${remote_log}.pid')\" 2>/dev/null; then \
         echo 'Remote policy server already running'; \
     else \
         cd '${remote_repo}'; \
         nohup ${remote_script} > '${remote_log}' 2>&1 < /dev/null & \
         echo \$! > '${remote_log}.pid'; \
         echo \"Started remote policy server pid=\$! log=${remote_log}\"; \
     fi"
