from glob import glob
import os
from time import time
from collections import defaultdict

import cv2
import torch
import onnxruntime
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# from onnxruntime.experimental import TensorRTProviderOptions
# 设置超参数
np.random.seed(31193)
torch.manual_seed(97)
torch.cuda.manual_seed_all(97)
torch.backends.cudnn.deterministic = True
nTrainBatchSize = 1
nCalibrationBatch = 4
nHeight = 28
nWidth = 28
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))


class MyData(torch.utils.data.Dataset):

    def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return data.reshape(1, nHeight, nWidth).astype(np.float32), label

    def __len__(self):
        return len(self.data)

onnx_file = ["./model-p.onnx"]#, "./model-p.onnx", "./model-sim.onnx", "./model-p-sim.onnx"]
# onnx_file = ["./model.pt", "./model.pt","./model.pt","./model.pt",]
trainDataset = MyData(True)
testDataset = MyData(False)
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

# onnx 11 之后，必须要指定解释器，目前提供三种可供选择
# providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']#
# providers = ['CUDAExecutionProvider']#
providers = ['CPUExecutionProvider']#
# provider_options = [trt_options]
metric = defaultdict(dict)

for file in onnx_file:
    t0 = time()
    # ort_session = onnxruntime.InferenceSession(file, providers=providers, provider_options=provider_options)
    # 初始化ort运行器
    ort_session = onnxruntime.InferenceSession(file, providers=providers)
    acc = 0
    n = 0
    for xTest, yTest in testLoader:
        # 注意输入的格式
        ort_inputs ={"x": xTest.numpy()}
        # 直接执行推理
        ort_outputs = ort_session.run(output_names=["z"], input_feed=ort_inputs)[0]
        acc += (ort_outputs == yTest.argmax(1).numpy()).sum()
        n += xTest.shape[0]
    metric[file] = {"time": time() - t0, "acc": (acc / n)}
# 需要注意的是，ort（或者基本都）需要指定输入输出的名称，可以通过onnx查看对应的名称或者节点编号

# for file in onnx_file:
#     t0 = time()
#     # ort_session = onnxruntime.InferenceSession(file, providers=providers, provider_options=provider_options)
#     # with open(file, "rb") as f:
#     model = torch.load(file)
#     # model.val()
#     acc = 0
#     n = 0
#     for xTest, yTest in testLoader:
#         # ort_inputs ={"x": xTest.numpy()}
#         # ort_outputs = ort_session.run(output_names=["z"], input_feed=ort_inputs)[0]
#         # acc += (ort_outputs == yTest.argmax(1).numpy()).sum()
#         t_outputs =  model(xTest)
#         acc += (t_outputs == yTest.argmax(1)).sum()
#         n += xTest.shape[0]
#     metric[file] = {"time": time() - t0, "acc": (acc.item() / n)}

print(metric)