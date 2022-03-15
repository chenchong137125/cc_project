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



|  parts_name  | error_other | error | 0.1<threshold<0.5 | 0.5 <threshold<0.7 | threshold>0.7 |
| :-----| :----: | :----: |:----: |:----: |:----: |
|    1-log     |      18     |   0   |         0         |         13        |      764      |
|  2-dangwei   |     149     |   0   |         0         |         79        |      2256     |
| 3-houshijing |     194     |   12  |         10        |        222        |      5851     |
| 4-yibiaopan  |      10     |   1   |         0         |         3         |      1038     |
|    5-ping    |     341     |   0   |         1         |        108        |      3884     |
|   6-dadeng   |     535     |   0   |         0         |        257        |      2122     |
|   7-yushua   |      30     |   11  |         1         |         23        |      892      |
|    8-key     |     697     |  118  |         44        |        584        |      7975     |


|  parts_name  | error_other | error | 0.1<threshold<=0.5 | 0.5<threshold<=0.7 | 0.7<threshold<0.9 | threshold>=0.9 |
| :-----| :----: | :----: |:----: |:----: |:----: |:----: |
|    1-log     |      1      |   0   |         0          |         2          |         5         |      149       |
|  2-dangwei   |      8      |   0   |         0          |         0          |         1         |       42       |
| 3-houshijing |      38     |   0   |         0          |         28         |         70        |      2487      |
| 4-yibiaopan  |      4      |   0   |         0          |         4          |         7         |       38       |
|    5-ping    |      8      |   0   |         0          |         9          |         11        |      1459      |
|   6-dadeng   |     110     |   0   |         0          |         37         |        139        |      4415      |
|   7-yushua   |      77     |   51  |         4          |         47         |         79        |      1212      |
|    8-key     |      65     |   7   |         5          |         59         |        262        |      6355     |


|  parts_name  | error_other | error | 0.1<threshold<=0.5 | 0.5<threshold<=0.7 | 0.7<threshold<0.9 | threshold>=0.9 |
| :-----| :----: | :----: |:----: |:----: |:----: |:----: |
|    1-log     |      1      |   0   |         0          |         1          |         13        |      142       |
|  2-dangwei   |      7      |   0   |         0          |         1          |         2         |       41       |
| 3-houshijing |      38     |   0   |         0          |         21         |         66        |      2498      |
| 4-yibiaopan  |      8      |   0   |         0          |         2          |         9         |       34       |
|    5-ping    |      18     |   0   |         0          |         11         |         20        |      1438      |
|   6-dadeng   |      74     |   0   |         0          |         37         |        126        |      4464      |
|   7-yushua   |      50     |   71  |         4          |         37         |         71        |      1237      |
|    8-key     |      42     |   3   |         3          |         45         |        172        |      6488      |