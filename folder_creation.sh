BASE_FOLDER=~/scratch/CVFinal/dataset
OUTPUT_DIR=~/scratch/CVFinal/dataset
PYTHON_DIR=~/scratch/CVFinal

#Do one for each video in the base folder
for video in ${BASE_FOLDER}/*.mov
do
  # Run the python script on that folder
  python ${PYTHON_DIR}/extract_frames.py \
    --video_path ${video} \
    --output_dir ${OUTPUT_DIR}
done