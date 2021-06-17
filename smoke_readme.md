# 1.环境安装
##  Python >= 3.6.0 required with all requirements.txt dependencies installed:
+ pip install -r requirements.txt



# 2 转换成onnx
+ 1.[squeezenet1.1](/home/md/cc/dms-train-tools/test_model/smoke/model_86_20200524131058.pth)转换成onnx模型

```
# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
CLASS_NUM=5
model_path='/home/md/cc/dms-train-tools/test_model/dms/model_86_20200524131058.pth'
file_name='smoke.onnx'
net = models.SqueezeNet(version = '1_1', num_classes = num_classes)
net.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
net.cpu()

input = torch.randn(1, 3, 224, 224)
#torch.onnx.export(net, input, file_name)
# torch.onnx.export(net, input, file_name, verbose=False, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])
torch.onnx.export(net, input, file_name, verbose=True,opset_version=11, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])

```
# 3.使用摄像头测试模型
+ [测试代码在这里](/home/md/cc/dms-train-tools/smoke_test/CameraShow.py)
+ 测试的命令帮助，python CameraShow.py -h
+ 测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu ,也可以使用cpu,使用的方式是 --detect-dev cpu
```
python CameraShow.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/dms/model_86_20200524131058.pth --mean-std 1 --gpu 0
```
