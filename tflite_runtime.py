"""
example for inference tool:
python tflite_runtime.py -s dataset/image/face01.jpg -w models/face.tflite
"""

import numpy as np
import tensorflow as tf
import cv2
import os
import argparse
try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,

from postprocessor.yolov5 import yolov5_postprocess


np.set_printoptions(threshold=np.inf, suppress=True, linewidth = np.inf, formatter={'float': '{:.3f}'.format})


class TFLiteModel:
    def __init__(self, weight_file: str) -> None:
        self.interpreter = Interpreter(model_path=weight_file)  # load TFLite model
        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs

    def getInputSize(self):
        [n, inputH, inputW, c] = self.input_details[0]["shape"]
        return (inputH, inputW)

    def infer(self, im):
        input, output = self.input_details[0], self.output_details[0]

        scale, zero_point = input["quantization"]
        if input["dtype"] == np.int8:
            im = (im / scale + zero_point).astype(np.int8)  # de-scale
        elif input["dtype"] == np.uint8:
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale

        self.interpreter.set_tensor(input["index"], im)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(output["index"])
        if input["dtype"] == np.int8 or input["dtype"] == np.uint8:
            scale, zero_point = output["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale  # re-scale

        return y

def tflite_run(tflite_model_path, input_image):
    assert input_image is not None
    INFERENCE_3CHANNEL = True

    if tflite_model_path.endswith(".tflite"):
        model = TFLiteModel(tflite_model_path)
        (inputH, inputW) = model.getInputSize()

    input_image = cv2.resize(image, (inputW, inputH)) # resize RGB
    im = input_image.astype(np.float32) / 255 # HWC, RGB scale 0-255 to 0-1

    if INFERENCE_3CHANNEL:
        im = im[None,...] # batch, channel, height, width            
    else:
        im = im[None, None, ...]

    im = np.ascontiguousarray(im)
    output_data = model.infer(im)


    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',type=str, default= './data/images/zidane.jpg', help='source for image file')
    parser.add_argument('--data_cfg',type=str, default= './data/training_cfg/data_config.yaml', help='testing data')
    parser.add_argument('-w', '--weight',type=str, default= './best-int8.tflite', help='path of tflite weight file')
    parser.add_argument('--img_ch', default=3, help="1 or 3 channel")
    parser.add_argument('--conf_thres', type=float, default = 0.4, help="confidence threshold")
    parser.add_argument('--iou_thres', type=float, default = 0.45, help="iou threshold")

    args = parser.parse_args()
    if args.img_ch == 3:
        INFERENCE_3CHANNEL = True
    else:
        INFERENCE_3CHANNEL = False

    model_path = args.weight
    SOURCE = args.source
    
    if INFERENCE_3CHANNEL:
        image = cv2.imread(SOURCE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB
    else:
        image = cv2.imread(SOURCE, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception(f"It is not an image file")
    #imageH, imageW, ch = image.shape # HWC
    # Invode tflite model
    output = tflite_run(model_path, image)
    print("output",output.shape)

    # Postprocess of yolov5n
    bbox = yolov5_postprocess(output, confThr=args.conf_thres, iou_thres=args.iou_thres)
    print(bbox)
    # result_img = np.array(input_data1).astype(np.float32)
    # result_img = result_img.reshape(320, 320, 3)
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    # for i, det in enumerate(bbox): 
    #     xyxy = (det[:4]*320.0).round()
    #     conf = det[4].round()
    #     cv2.rectangle(result_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 1)
            
    # cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) + '/dataset/yolov5n/result.jpg', result_img)
