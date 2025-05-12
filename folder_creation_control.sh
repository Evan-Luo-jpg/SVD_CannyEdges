BASE=~/scratch/CVFinal
DATA_FOLDER=~/scratch/Download_Scratch/bdd100k_videos_train_00/bdd100k/videos/train/
OUTPUT_DIR=${BASE}/dataset
PYTHON_DIR=${BASE}

#Do one for each video in the base folder
for video in ${DATA_FOLDER}/*.mov
do
  # Run the python script on that folder
  python ${PYTHON_DIR}/extract_frames_control.py \
    --video_path ${video} \
    --output_dir ${OUTPUT_DIR}
done
