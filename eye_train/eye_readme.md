# 1.环境安装
##  Python >= 3.6.0 required with all requirements.txt dependencies installed:
+ pip install -r requirements.txt

# 2.数据集
+ [训练集](/home/md/dataset/eye_dataset/eye/train_eye),数量如下：

|类别|数量|
| :-----| ----: |
|0-close_eye|45690|
|1-open_eye|39820|

+ [测试集](/home/md/dataset/eye_dataset/eye/test_eye)。数量如下：

|类别|数量|
| :-----| ----: |
|0-close_eye|355|
|1-open_eye|427|



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
## 1. 使用[mobilenetv2](/home/md/cc/dms-train-tools/test_model/eye/eye2.pth)训练结果如下：
+ 1.混淆矩阵如下:
```
      [[344  11]
      [ 5 422]]
```
+ 2.指标如下：
```

                precision    recall  f1-score   support

  closed_eye     0.9857    0.9690    0.9773       355
    open_eye     0.9746    0.9883    0.9814       427

    accuracy                         0.9795       782
   macro avg     0.9801    0.9787    0.9793       782
weighted avg     0.9796    0.9795    0.9795       782
```
+ 3.训练参数命令
 ```
 NN_train_1.py -n mobilenetv2 -mp ./eye_model/ -ds 2 --data-root /home/md/dataset/eye_dataset/eye/train_eye  --test-root /home/md/dataset/eye_dataset/eye/test_eye/ -s 32 -dx 7 --mean-std 1 --gpu 0
 ```



# 5 测试模型
+ 首先在/home/md/cc/dms-train-tools 目录下，输入
+ 1.workon cc_pytorch ，测试的命令帮助，python test_gender_camer.py -h
+ 2.测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu ,下面是使用的例子：

```
python eval_model.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/eye/eye2.pth  -t /home/md/dataset/eye_dataset/eye/test_eye --model-category classic --mean-std 1 --gpu 0
```

# 6 转换成onnx
+ 1.mobilenetv2转换成onnx模型

```
# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
CLASS_NUM=2
model_path='eye2.pth'
file_name='eye.onnx'
net = models.mobilenet_v2(num_classes=CLASS_NUM)
net.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
net.cpu()

input = torch.randn(1, 3, 224, 224)
#torch.onnx.export(net, input, file_name)
# torch.onnx.export(net, input, file_name, verbose=False, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])
torch.onnx.export(net, input, file_name, verbose=True,opset_version=11, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])

```

# 7.使用摄像头测试模型
+ [测试代码](/home/md/cc/dms-train-tools/eye_train/test_eye_camer.py)
+ 测试的命令帮助，python test_eye_camer.py -h
+ 测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu ,也可以使用cpu,使用的方式是 --detect-dev cpu,下面是使用的例子：
```
python test_eye_camer.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/eye/eye2.pth --mean-std 1 --gpu 0
```

