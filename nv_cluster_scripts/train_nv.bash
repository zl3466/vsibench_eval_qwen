job_name="eval_qwen_vsi"
base_logdir="/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/log/eval_qwen_vsi_$(date +%Y%m%d_%H%M%S)"

for i in  {1..2}; do
  submit_job --gpu 4 --cpu 64 --nodes 1 \
  --partition=grizzly,polar,polar3,polar4 \
  --account=nvr_av_foundations \
  --image=/lustre/fsw/portfolios/nvr/users/ymingli/dockers/2304py3.sqsh \
  --mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/nvr/users/ymingli/miniconda3:/home/ymingli/miniconda3,/home/ymingli/.local:/home/ymingli/.local,/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models:/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/models \
  --duration 4 \
  --dependency=singleton \
  --name $eval_qwen_vsi \
  --logdir ${base_logdir}/run_${i} \
  --notimestamp \
  --command "bash /lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/github/vsibench_eval_qwen/nv_cluster_scripts/evaluate_qwen.sh --model qwen25_7b --num_processes 1 --benchmark vsibench --hf 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"
done
