"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os,sys
import time

import cv2
import numpy as np
import onnx


from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

onnx_path = "./snet146.onnx"


predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
result_path = "./result"

threshold = 0.7
path = "/mnt/data1/yanghuiyu/dlmodel/Fd/Face-Detector-1MB-with-landmark/images/input"
sum = 0
if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(path)
sum = 0
for file_path in listdir:
    img_path = os.path.join(path, file_path)
    orig_image = cv2.imread("/mnt/data1/yanghuiyu/project/object_detect/Thundernet_new/voc_images/input/2008_000171.jpg")
    image = cv2.resize(orig_image, (224, 224))

    # image = cv2.resize(image, (640, 480))

    # mean = np.array([0.40789654, 0.44719302, 0.47026115],
    #                 dtype=np.float32).reshape(1, 1, 3)
    # std = np.array([0.28863828, 0.27408164, 0.27809835],
    #                dtype=np.float32).reshape(1, 1, 3)

    # print(image)
    mean = np.array([[[0.485 * 255, 0.456 * 255, 0.406 * 255]]])

    image = image - mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    # boxes , confidences, landmark  = ort_session.run(None, {input_name: image})
    cls_prob = predictor.run(image)[0]
    print(np.argmax(cls_prob))
    print(cls_prob[0][np.argmax(cls_prob)])
