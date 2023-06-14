#!/bin/bash
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Download COCO 2017 dataset http://cocodataset.org
# Example usage: bash data/scripts/get_coco.sh
# parent
# ├── yolov5
# └── datasets
#     └── coco  ← downloads here
current_script_dir=$(dirname $0)
project_dir=$current_script_dir/..

cd $project_dir

if [ -e data/datasets/SJCAM_lens-140_320x240 ]; then
  echo "The ROT_AI dataset has already been downloaded."
  exit 0
fi

# Arguments (optional) Usage: bash data/scripts/get_coco.sh --train --val --test --segments
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    --test) test=true ;;
    --segments) segments=true ;;
    esac
  done
else
  train=true
  val=true
  test=false
  segments=false
fi

# Download/unzip labels
#d='../data/datasets/' # unzip directory
d='data/datasets/' # unzip directory
d_label='data/training_cfg/labels/Data_center/'
url=http://192.168.9.93:8000/ROT_AI/
files="ROT_4_classes.zip IR_Extended_dataset_infer_4class.zip ROT_gray_192x192_IRLed.zip GC_IRCAM_TEST_2023-05-29.zip SA_ROT_IR_gray_192x192_20230526.zip SA_ROT_gray_IRLed_20230526.zip"
for file in $files
do
  echo 'Downloading' $url$f ' ...'
  curl -L $url$file -o data/$file -# && unzip -q data/$file -d $d && rm data/$file &
done
wait