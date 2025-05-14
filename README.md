# SVD_Canny_Edges

This repository fine-tunes [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) using a subset of the [BDD100K](https://bdd-data.berkeley.edu/) dataset with **Canny edge maps** as control signals. It implements LoRA fine-tuning on the UNet backbone with added support for custom 4-channel inputs (RGB + edge).

---

## Dataset: BDD100K

We use 172 videos from the BDD100K driving dataset. Each video is extracted into individual frames and stored in its own folder. For the modified training script, we also extracted Canny edges from each frame and saved them in a subfolder inside the dataset folder.

### Video Data Processing
Note that BDD100K is a driving video/image dataset, but this is not a necessity for training. Any video can be used to initiate your training. Please refer to the `DummyDataset` data reading logic. In short, you only need to modify `self.base_folder` in the oscar.sh scripts. Then arrange your videos in the following file structure:
```bash
self.base_folder
    ├── video_name1
    │   ├── video_frame1
    │   ├── video_frame2
    │   ...
    ├── video_name2
    │   ├── video_frame1
        ├── ...
```
We have scripts that do this for you, depending on the model you want to run. You will need to change the `DATA_FOLDER` and `BASE` to match your data folder and base, respectively.

## Running the model
To run the model, clone the GitHub repo. Here are the pretrained weights for if you just want to run inference on the BDD100K trained model [weights](https://drive.google.com/drive/folders/18FeLhJ_C3SEs9GQ14LHCjFveGnZQgbIH?usp=drive_link). You want to download the weights into the base folder, or where you cloned the repo.
The oscar.sh file is in this format:
```bash
module load cudnn cuda

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32

BASE_FOLDER=~/scratch/CVFinal

accelerate launch --num_processes=2 train_svd_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid \
  --base_folder=${BASE_FOLDER}/dataset \
  --output_dir=${BASE_FOLDER} \
  --num_train_epochs=1 \
  --num_frames=12 \
  --width=448 \
  --height=256 \
  --per_gpu_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --max_train_steps=500 \
  --validation_steps=250 \
  --num_validation_images=1 \
  --checkpoints_total_limit=5 \
  --seed=42 \
```
### Note:
You may have to modify your base folder to match wherever you cloned the repo. The script is also setup for multiple GPUs, to only run on one, get rid of
```bash
--num_processes=2
```
Also, you will need a minimum of 24 GB of VRAM to train and run inference on the models, even with the weights loaded, so this will likely not work locally. More args be found in the training script as well.

## Acknowledgement

Our model is related to [Diffusers](https://github.com/huggingface/diffusers) and [Stability AI](https://github.com/Stability-AI/generative-models). Thanks for their great work!
We also thank [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend/tree/main) for their training script, which we were able to work off of.
Also thanks to our TA Shania Guo, Professor James Tompkin as well as the Hugging Face ecosystem.


