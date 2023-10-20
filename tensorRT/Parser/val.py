from cuda import cudart # type: ignore
import cv2
from glob import glob
import numpy as np
import os
import tensorrt as trt
import torch as t


# 设置参数
np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97) # type: ignore
t.backends.cudnn.deterministic = True # type: ignore
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

np.set_printoptions(precision=4, linewidth=200, suppress=True) # type: ignore
cudart.cudaDeviceSynchronize()
# 1.创建logger
logger = trt.Logger(trt.Logger.ERROR) # type: ignore 
# 2.读取trt序列化文件
with open(trtFile, "rb") as f:
    engineString = f.read()
# 3.通过文件反序列化加载引擎
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString) # type: ignore
# 4.构建上下文
context = engine.create_execution_context()
# 5.指定需要申请的内存块(输入输出)的数量
nIO = engine.num_io_tensors
# 6.初始化输入输出名称
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
# 7.计算输入的张量的数量
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT) # type: ignore
# 8.设置输入张量的尺寸
context.set_input_shape(lTensorName[0], [1, 1, nHeight, nWidth])
# for i in range(nIO):
#     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
# 9.声明本地空内存块,用于储存输入输出张量
bufferH = []
# 10.对本地空内存块进行初始化赋值
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
# 11.声明服务端空内存块，用于储存输入输出张量
bufferD = []
# 12.在服务端占用指定大小的内存空间
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
# 13.读取数据
data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth) # type: ignore
# 14.将数据内存拷贝到本地内存中
bufferH[0] = data
# 15.将本地内存中的数据拷贝到对应的服务端内存中
for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
# 16.为对应的输入输出指定内存空间
for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))
# 17.执行异步执行
context.execute_async_v3(0)
# 18.将推理结果从服务端拷贝到对应的本地内存中
for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
# 19.拿到推理结果做后续处理……
for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])
# 20.释放cuda内存
for b in bufferD:
    cudart.cudaFree(b)

print("Succeeded running model in TensorRT!")