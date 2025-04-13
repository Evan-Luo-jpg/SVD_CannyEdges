# NovelViewGeneration-LITE

## Useful Links
### https://objaverse.allenai.org/
### https://github.com/cvlab-columbia/zero123/tree/main
### https://github.com/facebookresearch/segment-anything
### https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/main/deep-dives/011-stable-zero123/README.md
#### UNet, 8k images, 20 view points per image, 3090

##In setting up the repo, you want to follow this link, https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering

The files you want to change are in the main file of the scripts and rendering folder. You want to change the render_dir to be this folder and not ~/.objaverse. This is just to make it easy to access. We want to modify the get_example object function to sample our own from objaverse not just the json sample.

If you are on windows, you need to make an .venv folder with all the libraries and run it in wsl with linux. This is because for some reason these scripts only work with macOS and with linux. Otherwise, just make a new environment with miniforge (you can just copy the csci1430 env) and then download the required libraries.
