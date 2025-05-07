module load cudnn cuda

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32

BASE_FOLDER=~/scratch/CVFinal

accelerate launch --num_processes=1 modified_train_svd_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
  --base_folder=${BASE_FOLDER}/dataset \
  --output_dir=${BASE_FOLDER} \
  --num_train_epochs=1 \
  --num_frames=12 \
  --width=320 \
  --height=192 \
  --per_gpu_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --max_train_steps=500 \
  --validation_steps=250 \
  --num_validation_images=3 \
  --checkpoints_total_limit=5 \
  --seed=42 \
