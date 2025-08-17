#!/bin/bash

# Default values
job_name="eval_qwen_vsi"
hf_token=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --hf)
        hf_token="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        echo "Usage: $0 --hf <hf_token>"
        exit 1
        ;;
    esac
done

# Check if hf_token was provided
if [ -z "$hf_token" ]; then
    echo "Error: --hf argument is required"
    echo "Usage: $0 --hf <hf_token>"
    exit 1
fi

base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/log/eval_qwen_vsi_$(date +%Y%m%d_%H%M%S)"

for i in {1..2}; do
  submit_job --gpu 4 --cpu 64 --nodes 1 \
  --partition=grizzly,polar,polar3,polar4 \
  --account=nvr_av_foundations \
  --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
  --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3,/home/ymingli/.local:/home/ymingli/.local,/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models:/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models \
  --duration 4 \
  --dependency=singleton \
  --name ${job_name}_${i} \
  --logdir ${base_logdir}/run_${i} \
  --notimestamp \
  --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/github/vsibench_eval_qwen/nv_cluster_scripts/evaluate_qwen.sh --model qwen25_7b --num_processes 1 --benchmark vsibench --hf $hf_token"
done