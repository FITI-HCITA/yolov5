## <div align="center">AI PIPELINE🚀</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone -b ROT_IRAI_VA8801 https://github.com/FITI-HCITA/yolov5.git  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

## <div align="center">How to Generate Yolov5 model for VA8801?</div>
1. Prepare Dataset: Download custom dataset or use example data

2.  Inference: Inference testing data with a TFLite pretrained model, which can be downloaded from the model zoo for the
[Sweep Robot Detection model](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Sweep_Robot_Detection/Yolo/VA8801_ROTBOT_1.002.002-int8.tflite)

```bash
python tflite_runtime.py -s data/datasets/test/0_0_14_1_0_3.30_PetStool_83.bmp -w /path/to/VA8801_ROTBOT_1.002.002-int8.tflite --img_ch 1
```
3. Train model: Transfer learning with a PyTorch pretrained model, which can be downloaded from the model zoo for the [Sweep Robot Detection model](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Sweep_Robot_Detection/Yolo/VA8801_ROTBOT_1.002.002.pt)
- Please check your model path **--weights "pretrained pytorch model path"**
- Please check your PC device **--device "cuda device, i.e. 0 or 0,1,2,3 or cpu"**

```bash
python train.py --device 0 --data data/training_cfg/data_config.yaml --weights /path/to/VA8801_ROTBOT_1.002.002.pt --imgsz 320 --imgch 1 --cfg models/2head_yolov5n_WM028.yaml
```

4.  Export int8 tflite model
- The conversion is supported by **Python version 3.9.0** and **TensorFlow version 3.9.16**.
- Please check your model path **--weights "your pytorch model path"**
- Please check the image size for export to the TFLite model **--imgsz_tflite "image size"**.
- Please check your PC device **--device "cuda device, i.e. 0 or 0,1,2,3 or cpu"**

```bash
python ai_pipeline.py --data data/training_cfg/data_config.yaml --weights /path/to/VA8801_ROTBOT_1.002.002.pt --batch-size 1 --imgch 1 --imgsz 320 --imgsz_tflite 320 --device 0 --include tflite --int8 --run export

```

<details open>
<summary>Example for train from scatch</summary>


run training only

```bash
python ai_pipeline.py --data <data yaml path> --cfg <model yaml path> --epochs 10 --batch-size 64 --imgch 1 --imgsz 320 --patience 0 --device 0 --run train
```

run export only

```bash
python ai_pipeline.py --data <data yaml path> --weights <torch model path> --batch-size 1 --imgch 1 --imgsz 320 --device 0 --include tflite --int8 --run export

```

run inference only

```bash
python ai_pipeline.py --data <data yaml path> --conf-thres-test 0 --device 0 --tflite_model_path <tflite_model_path> --save_dir <xml save folder path> --run inference
```

run tflite inference for va8801 results only

```bash
python tflite_runtime.py -s <image data> -w <tflite model> 
```
</details>
