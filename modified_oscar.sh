module load cudnn cuda

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_FOLDER=~/scratch/CVFinal

#Have to use a smaller base model to fit in memory
accelerate launch --num_processes=1 modified_train_svd_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid \
  --base_folder=${BASE_FOLDER}/dataset \
  --rank=2 \
  --output_dir=${BASE_FOLDER} \
  --num_train_epochs=1 \
  --num_frames=4 \
  --width=128 \
  --height=64 \
  --per_gpu_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpointing_steps=25 \
  --max_train_steps=50 \
  --validation_steps=25 \
  --num_validation_images=1 \
  --checkpoints_total_limit=2 \
  --seed=42 \
