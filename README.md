## <div align="center">AI PIPELINE🚀</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

1. Create python environment.
- It is recommended to use **Anaconda** to set up the Python environment. Here is the [Miniconda Install Tutorial](https://medium.com/@hmchang/%E7%B5%A6%E5%88%9D%E5%AD%B8%E8%80%85%E7%9A%84-python-%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-578bf0de9cf8).
- The TFlite conversion is supported by **Python version 3.9.0** and **TensorFlow version 2.13.0**.
```bash
conda create --name yolov5 python=3.9.0
conda activate yolov5
pip install tensorflow==2.13.0
```

2. Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone -b hand_gesture_VA8801 https://github.com/FITI-HCITA/yolov5.git  # clone
cd yolov5
pip install -r requirements.txt  # install
```

3. Clone VA8801_Model_Zoo (Download VA8801 pretrained models)
```bash
git clone https://github.com/FITI-HCITA/VA8801_Model_Zoo.git
```

</details>

## <div align="center">How to Generate Yolov5 model for VA8801?</div>
1. Prepare Dataset: Use example data at ``data/dataset`` or Use your custom dataset

2.  Inference: Inference testing data with a TFLite pretrained model, which can be downloaded from the model zoo for the
[Hand model](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Hand_Gestures/Hand_Gestures_3_001_001.tflite)

- Please check your local model path **-w "pretrained pytorch model path"**

    Example of your local model folder
    
    path: ``VA8801_Model_Zoo/ObjectDetection/Hand_Gestures/Yolo``

```bash
python tflite_runtime.py -s data/dataset/test/hand_001.jpg -w path/Hand_Gestures_3_001_001.tflite --img_ch 3
```
3.  Train model: Transfer learning with a PyTorch pretrained model, which can be downloaded from the model zoo for the [Hand model](https://github.com/FITI-HCITA/VA8801_Model_Zoo/blob/main/ObjectDetection/Hand_Gestures/Hand_Gestures_3_001_001.pt)
- Please check your local model path **--weights "pretrained pytorch model path"**
  
    Example of your local model folder
    
    path: ``VA8801_Model_Zoo/ObjectDetection/Hand_Gestures/Yolo``
- Please check your PC device **--device "cuda device, i.e. 0 or 0,1,2,3 or cpu"**

```bash
python train.py --device 0 --data data/training_cfg/data_config.yaml --weights path/Hand_Gestures_3_001_001.pt --imgsz 320 --imgch 3 --cfg models/2head_yolov5n_WM022.yaml
```

4.  Export int8 tflite model
- Please check your local model path **--weights "your pytorch model path"**
    - After training, your trained model will be saved at ``results/yyyy_mm_dd/trialx/weights/best.pt``
- Please check the image size for export to the TFLite model **--imgsz_tflite "image size"**.
- Please check your PC device **--device "cuda device, i.e. 0 or 0,1,2,3 or cpu"**

```bash
python ai_pipeline.py --data data/training_cfg/data_config.yaml --weights path/Hand_Gestures_3_001_001.pt --batch-size 1 --imgch 3 --imgsz 320 --imgsz_tflite 320 --device 0 --include tflite --int8 --run export

```
 
<details open>
<summary>Example for train from scatch</summary>


run training only

```bash
python ai_pipeline.py --data <data yaml path> --cfg <model yaml path> --epochs 10 --batch-size 64 --imgch 1 --imgsz 320 --patience 0 --device 0 --run train
```

run export only

```bash
python ai_pipeline.py --data <data yaml path> --weights <torch model path> --batch-size 1 --imgch 1 --imgsz 192 --device 0 --include tflite --int8 --run export

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
