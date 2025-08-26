#!/bin/bash
# CRITICAL: Override any distributed environment for single-process execution
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
# Clear any SLURM distributed variables that might be set by the job scheduler
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NTASKS
unset SLURM_NPROCS
export VLLM_PLATFORM=cuda

# Add user site-packages to Python path
export PYTHONPATH="/home/ymingli/.local/lib/python3.10/site-packages:$PYTHONPATH"
# Source conda and activate environment
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate vsibench
#export HUGGING_FACE_HUB_TOKEN=""
export HF_HUB_CACHE="/lustre/fsw/portfolios/nvr/users/ymingli/cache/huggingface/hub"

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

available_models="llava_one_vision_qwen2_0p5b_ov_32f,llava_one_vision_qwen2_7b_ov_32f,llava_next_video_7b_qwen2_32f,llama3_vila1p5_8b_32f,llama3_longvila_8b_128frames_32f,longva_7b_32f,internvl2_2b_8f,internvl2_8b_8f"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    --hf)
        hf="$2"
        shift 2
        ;;
    --thought)
        thought="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VSI_THOUGHT_PROCESS=$thought
export VSI_DATASET="full"
export HUGGING_FACE_HUB_TOKEN="$hf"

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=4
num_frames=32
launcher=python


if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "qwen25_7b")
        model_family="qwen25vl"
        model_args="pretrained=Qwen/Qwen2.5-VL-7B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,device_map=auto,modality=video"
        num_processes=1
        ;;
    "qwen25_7b_sft")
        model_family="qwen25vl_tuned"
        model_args="pretrained=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/vlm_odom/7B+sft+grpo/checkpoint-10684,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,device_map=auto,modality=video"
        num_processes=1
        ;;
    "qwen25_7b_tuned")
        model_family="qwen25vl_tuned"
        model_args="pretrained=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/vlm_odom/7B+grpo/checkpoint-10684,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,device_map=auto,modality=video"
        num_processes=1
        ;;
    *)
        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
