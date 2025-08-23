#!/bin/bash

# Add user site-packages to Python path
export PYTHONPATH="/home/ymingli/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Override cluster distributed settings for vLLM single-process execution
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NTASKS
unset SLURM_NPROCS

# Debug GPU detection
echo "=== GPU Debug Information ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================="

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

export OPENAI_API_KEY="" # API KEY FOR OPENAI CHATGPT
export GOOGLE_API_KEY="" # API KEY FOR GOGOLE GEMINI
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VSI_THOUGHT_PROCESS=1
export VSI_DATASET="full"

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=8
num_frames=32
launcher=python
tensor_parallel_size=1
pipeline_parallel_size=1

available_models="llava_one_vision_qwen2_0p5b_ov_32f,llava_one_vision_qwen2_7b_ov_32f,llava_next_video_7b_qwen2_32f,llama3_vila1p5_8b_32f,llama3_longvila_8b_128frames_32f,longva_7b_32f,internvl2_2b_8f,internvl2_8b_8f"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --tensor_parallel_size)
        tensor_parallel_size="$2"
        shift 2
        ;;
    --pipeline_parallel_size)
        pipeline_parallel_size="$2"
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
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

export HUGGING_FACE_HUB_TOKEN="$hf"

if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "qwen25_72b")
        model_family="qwen25vl"
        model_args="pretrained=Qwen/Qwen2.5-VL-72B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,tensor_parallel_size=$tensor_parallel_size,pipeline_parallel_size=$pipeline_parallel_size,modality=video"
#        num_processes=1
#        model_args="pretrained=Qwen/Qwen2.5-VL-72B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,modality=video"
        num_processes=$num_processes
        ;;
    "qwen25_7b")
        model_family="qwen25vl"
        model_args="pretrained=Qwen/Qwen2.5-VL-7B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,tensor_parallel_size=$tensor_parallel_size,pipeline_parallel_size=$pipeline_parallel_size,modality=video"
#        num_processes=1
#        model_args="pretrained=Qwen/Qwen2.5-VL-7B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,device_map=0,modality=video"
        num_processes=$num_processes  # Use 8 processes for data parallelism across 8 GPUs
        # launcher=accelerate  # Use accelerate for multi-GPU distribution
        ;;
    "qwen25vl_tuned")
        model_family="qwen25vl_tuned"
        model_args="pretrained=Qwen/Qwen2.5-VL-7B-Instruct,download_dir=/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models/qwen,video_decode_backend=decord,conv_template=qwen_2_5,max_frames_num=64,tensor_parallel_size=$tensor_parallel_size,pipeline_parallel_size=$pipeline_parallel_size,modality=video"
        num_processes=$num_processes
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
    
    echo "=== Evaluation Command Debug ==="
    echo "Model family: $model_family"
    echo "Model args: $model_args"
    echo "Num processes: $num_processes"
    echo "Launcher: $launcher"
    echo "Full command: $evaluate_script"
    echo "================================"

    echo "=== Debug Parallelism Parameters ==="
    echo "tensor_parallel_size: $tensor_parallel_size"
    echo "pipeline_parallel_size: $pipeline_parallel_size"
    echo "model_args: $model_args"
    echo "================================="

    echo "Full command: $evaluate_script"

    eval $evaluate_script
done
