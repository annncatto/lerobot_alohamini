#!/usr/bin/env bash

set -euo pipefail

repo=${ALOHAMINI_REPO:-/home/pi5/lerobot_alohamini}
python=${ALOHAMINI_PYTHON:-/home/pi5/miniconda3/envs/lerobot_alohamini/bin/python}
server_address=${ALOHAMINI_POLICY_SERVER:-127.0.0.1:18080}

if [[ ! -x "${python}" ]]; then
    echo "Python environment not found: ${python}" >&2
    exit 2
fi

# Run this on the Raspberry Pi after alohamini_pi_direct_tunnel.sh.  This process
# opens the motors and cameras itself: do not run alohamini_host.py at the same
# time.  Only latest observations and action chunks cross the WAN; there is no
# AlohaMini Host/ZMQ or PC relay.  Callers can override any default by repeating
# the corresponding argparse option later in "$@".
cd "${repo}"
export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" examples/alohamini/evaluate_bi_remote.py \
    --remote.manage_server=false \
    --server.address="${server_address}" \
    --robot.transport=direct \
    --fps=25 \
    --observation.send_mode=latest \
    --async.aggregate_fn=latest_only \
    --async.image_compression_quality=85 \
    --eval.record_dataset=false \
    --dataset.push_to_hub=false \
    "$@"
