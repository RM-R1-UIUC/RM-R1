#!/usr/bin/env bash
set -e                       # stop on error

# ---------- host-side paths ----------
export HF_HOME=/shared/nas2/hf_cache       # shared HF cache (must be writable)

# ---------- tweakables ----------
IMAGE="nvcr.io/nvidia/nemo:24.09.framework"    # or any recent tag
MODEL_REPO="nvidia/Llama2-13B-SteerLM-RM"
CONTAINER_NAME="steer-rm-13b"
PORT=1424

# ---------- run the container ----------
docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus device=0,1,2,3 \
  --user $(id -u):$(id -g) \                  # run as *you*, not root
  -v ${HF_HOME}:/root/.cache/huggingface \
  -p ${PORT}:${PORT} \
  --shm-size 32g \
  ${IMAGE} \
  bash -lc "
    python /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
      rm_model_file=${MODEL_REPO} \
      trainer.num_nodes=1 \
      trainer.devices=4 \
      ++model.tensor_model_parallel_size=4 \
      ++model.pipeline_model_parallel_size=1 \
      inference.micro_batch_size=4 \
      inference.port=${PORT}
  "

echo "Container '${CONTAINER_NAME}' is starting.  Tail logs with:
  docker logs -f ${CONTAINER_NAME}"