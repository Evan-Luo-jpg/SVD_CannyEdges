# SVD_Canny_Edges

This repository fine-tunes [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) using a subset of the [BDD100K](https://bdd-data.berkeley.edu/) dataset with **Canny edge maps** as control signals. It implements LoRA fine-tuning on the UNet backbone with added support for custom 4-channel inputs (RGB + edge).

---

## 📁 Dataset: BDD100K

We use 172 videos from the BDD100K driving dataset. Each video is extracted into individual frames and stored in its own folder. For the modified training script, we also extracted Canny edges from each frame and saved them in a subfolder inside the dataset folder.

### Video Data Processing
Note that BDD100K is a driving video/image dataset, but this is not a necessity for training. Any video can be used to initiate your training. Please refer to the `DummyDataset` data reading logic. In short, you only need to modify `self.base_folder`. Then arrange your videos in the following file structure:
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

## Running the model
To run the model, 

## Acknowledgement

Our model is related to [Diffusers](https://github.com/huggingface/diffusers) and [Stability AI](https://github.com/Stability-AI/generative-models). Thanks for their great work!
We also thank [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend/tree/main) for their training script, which we were able to work off of.
Also thanks to our TA Shania Guo, Professor James Tompkin as well as the Hugging Face ecosystem.


