# 1.环境安装
##  Python >= 3.6.0 required with all requirements.txt dependencies installed:
+ pip install -r requirements.txt

# 2.数据集
+ [训练集](/home/md/dataset/age_dataset/age/train),数量如下：

  |类别|数量|
  | :-----| ----: |
  |0-child|23411|
  |1-adult|306057|
  |2-elderly|11284|

+ [测试集](/home/md/dataset/age_dataset/age/test)。数量如下：

|类别|数量|
| :-----| ----: |
|0-child|500|
|1-adult|15312|
|2-elderly|354|



# 3.训练的文件说明和数据增强
## 训练的文件
+ NN_train_focal_loss.py命令是训练具有focal_loss
+ NN_train_1.py命令是训练不带有focal_loss
## 数据增强
+ 随机颜色增强
+ 随机剪切
+ 随机翻转
+ 随机仿射变换
+ 随机旋转
+ 随机灰度
+ 高斯变换
+ 随机自动对比度
+ 随机调节清晰度
+ 随机透视变换

# 4.测试的结果
## 1. 使用[mobilenetv2](/home/md/cc/dms-train-tools/test_model/age/model_23_20210615225612.pth)训练结果如下：
+ 1.混淆矩阵如下：
```
       [[  500     0     0]
       [    2 15241    69]
       [    0   120   234]]
```
+ 2.指标如下：
```
                 precision    recall  f1-score   support

       child     0.9960    1.0000    0.9980       500
       adult     0.9922    0.9954    0.9938     15312
      elderly     0.7723    0.6610    0.7123       354

     accuracy                         0.9882     16166
   macro avg     0.9202    0.8855    0.9014     16166
weighted avg     0.9875    0.9882    0.9877     16166
```
+ 3.训练参数命令
```
python NN_train_1.py -n mobilenetv2 -mp ./train_age_model/age_model/ -ds 2 --data-root /home/md/dataset/age_dataset/age/train  --test-root /home/md/dataset/age_dataset/age/test/ -s 32 -dx 7 --mean-std 1 --gpu 0
```


# 5 测试模型
+ 首先在/home/md/cc/dms-train-tools 目录下，输入
+ 1.workon cc_pytorch ，测试的命令帮助，python test_gender_camer.py -h
+ 2.测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu,下面是使用例子： 
```
 python eval_model.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/age/model_23_20210615225612.pth  -t /home/md/dataset/age_dataset/age/test --model-category classic --mean-std 1 --gpu 0
```
# 6 转换成onnx
+ 1.mobilenetv2转换成onnx模型

```

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
CLASS_NUM=2
model_path='model_23_20210615225612.pth'
file_name='age.onnx'
net = models.mobilenet_v2(num_classes=CLASS_NUM)
net.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
net.cpu()

input = torch.randn(1, 3, 224, 224)
#torch.onnx.export(net, input, file_name)
# torch.onnx.export(net, input, file_name, verbose=False, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])
torch.onnx.export(net, input, file_name, verbose=True,opset_version=11, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])

```
# 7.使用摄像头测试模型
+ 测试代码在(/home/md/cc/dms-train-tools/eye_train/test_age_camer.py)
+ 测试的命令帮助，python test_age_camer.py -h
+ 测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu ,也可以使用cpu,使用的方式是 --detect-dev cpu
 ```
 python test_age_camer.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/age/model_23_20210615225612.pth --mean-std 1 --gpu 0
```
