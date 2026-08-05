#!/usr/bin/env bash
# ============================================================
# dsv4f-cluster.sh — DeepSeek-V4-Flash-0731 on 2x DGX Spark
# Usage:
#   ./dsv4f-cluster.sh head                        # the acer (API host)
#   ./dsv4f-cluster.sh worker <HEAD_FABRIC_IP>     # the spark
# Serve: tp-2, 8 concurrent, 1M context, all interfaces (tailscale-reachable)
# Requires: weights at $MODEL on BOTH nodes, fabric interface enp1s0f1np1
# ============================================================
set -euo pipefail

ROLE="${1:-head}"
HEAD_IP="${2:-}"
MODEL=/home/rhmaomao/hosting/deepseek_v4_flash_0731/model
MN_IF_NAME=enp1s0f1np1
VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
CACHE=~/.cache/huggingface
PORT=8000

# ---------- 0. pre-flight ----------
[ -d "$MODEL" ] || { echo "!! model missing here: $MODEL"; exit 1; }
[ "$(ls -A "$MODEL" | wc -l)" -ge 48 ] || { echo "!! model incomplete"; exit 1; }

# ---------- 1. the vLLM image (both nodes) ----------
docker pull "$VLLM_IMAGE"

# ---------- 2. Ray cluster over the fabric ----------
export VLLM_HOST_IP=$(ip -4 addr show "$MN_IF_NAME" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export UCX_NET_DEVICES=$MN_IF_NAME NCCL_SOCKET_IFNAME=$MN_IF_NAME
export OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME GLOO_SOCKET_IFNAME=$MN_IF_NAME
export TP_SOCKET_IFNAME=$MN_IF_NAME RAY_memory_monitor_refresh_ms=0

if [ "$ROLE" = head ]; then
  echo "== Ray head on $VLLM_HOST_IP =="
  bash run_cluster.sh "$VLLM_IMAGE" --head "$CACHE" \
    -e VLLM_HOST_IP="$VLLM_HOST_IP" -e UCX_NET_DEVICES="$MN_IF_NAME" \
    -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
    -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
    -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
    -e RAY_memory_monitor_refresh_ms=0
else
  [ -n "$HEAD_IP" ] || { echo "!! worker needs <HEAD_FABRIC_IP>"; exit 1; }
  echo "== joining $HEAD_IP as worker =="
  bash run_cluster.sh "$VLLM_IMAGE" "$HEAD_IP" --worker "$CACHE" \
    -e VLLM_HOST_IP="$VLLM_HOST_IP" -e UCX_NET_DEVICES="$MN_IF_NAME" \
    -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
    -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
    -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
    -e RAY_memory_monitor_refresh_ms=0 -e MASTER_ADDR="$HEAD_IP"
  echo "== worker up; nothing else runs here =="
  exit 0
fi

# ---------- 3. validate + serve (head only) ----------
echo "== waiting for Ray to see both GPUs =="
sleep 15
docker exec -it "$VLLM_IMAGE" ray status || true

echo "== serving: tp-2, 8 concurrent, 1M ctx =="
docker exec -it "$VLLM_IMAGE" vllm serve "$MODEL" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 1000000 --max-num-seqs 8 \
  --distributed-executor-backend ray
