#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 REMOTE_SSH_PORT [LOCAL_GRPC_PORT] [REMOTE_GRPC_PORT]" >&2
    exit 2
fi

remote_ssh_port=$1
local_grpc_port=${2:-18080}
remote_grpc_port=${3:-8080}
remote_host=${ALOHAMINI_REMOTE_HOST:-connect.bjb1.seetacloud.com}
remote_user=${ALOHAMINI_REMOTE_USER:-root}

# Run this on the Raspberry Pi.  The policy gRPC socket remains private on the
# GPU instance; only the Pi-local loopback port is exposed.
exec ssh \
    -N \
    -T \
    -p "${remote_ssh_port}" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${local_grpc_port}:127.0.0.1:${remote_grpc_port}" \
    "${remote_user}@${remote_host}"
