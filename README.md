## <div align="center">AI PIPELINE🚀</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/FITI-HCITA/yolov5.git  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

download custom dataset or use example data

inference testing data with tflite pretrained model which can download in model zoo
[Human model (low power)](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Human_Detection/Yolo/HUMAN_DET_6_001_001.tflite)

```bash
python3 tflite_runtime.py -s data/dataset/test/human_001.jpg -w HUMAN_DET_6_001_001.tflite
```
transfer learning with pytorch pretrained model which can download in model zoo [Human model (low power)](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Human_Detection/Yolo/HUMAN_DET_6_001_001.pt)

```bash
python3 train.py --device 0 --data data/training_cfg/data_config.yaml --weights HUMAN_DET_6_001_001.pt --imgsz 96 --imgch 1 --cfg models/yolov5n_WM005_DM033.yaml
```
inference testing data with tflite pretrained model which can download in model zoo
[Human model (normal)](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Human_Detection/Yolo/HUMAN_DET_7_002_002.tflite)

```bash
python3 tflite_runtime.py -s data/dataset/test/human_002.jpg -w HUMAN_DET_7_002_002.tflite
```
transfer learning with pytorch pretrained model which can download in model zoo [Human model (normal)](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Human_Detection/Yolo/HUMAN_DET_7_002_002.pt)

```bash
python3 train.py --device 0 --data data/training_cfg/data_config.yaml --weights HUMAN_DET_7_002_002.pt --imgsz 320 --imgch 3 --cfg models/2_head_yolov5n_WM022.yaml
```

<details open>
<summary>Example for train from scatch</summary>


run training only

```bash
python3 ai_pipeline.py --data <data yaml path> --cfg <model yaml path> --epochs 10 --batch-size 64 --imgch 1 --imgsz 320 --patience 0 --device 0 --run train
```

run export only

```bash
python3 ai_pipeline.py --data <data yaml path> --weights <torch model path> --batch-size 1 --imgch 1 --imgsz 192 --device 0 --include tflite --int8 --run export

```

run inference only

```bash
python3 ai_pipeline.py --data <data yaml path> --conf-thres-test 0 --device 0 --tflite_model_path <tflite_model_path> --save_dir <xml save folder path> --run inference
```

run tflite inference for va8801 results only

```bash
python3 tflite_runtime.py -s <image data> -w <tflite model> 
```
</details>
