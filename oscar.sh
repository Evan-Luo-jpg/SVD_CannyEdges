module load cudnn cuda

BASE_FOLDER=~/scratch/CVFinal

accelerate launch train_svd_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt \
  --base_folder=${BASE_FOLDER}/dataset \
  --output_dir=${BASE_FOLDER} \
  --num_train_epochs=50 \
  --width=512 \
  --height=320 \
  --per_gpu_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpoints_total_limit=5 \
  --seed=42 \
