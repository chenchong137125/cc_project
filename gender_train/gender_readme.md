# 1.环境安装
##  Python >= 3.6.0 required with all requirements.txt dependencies installed:
+ pip install -r requirements.txt

# 2.数据集
+ [训练集](/home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/train),训练集是包括AFAD数据集，筛选后的Celeba数据集,以及筛选后imdb数据集。数量如下：

|类别|数量|
| :-----| ----: |
|0-female|274633|
|1-male|334565|

+ [测试集](/home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/test)测试集是模拟大屏下采集的。数量如下：

|类别|数量|
| :-----| ----: |
|0-female|1262|
|1-male|2561|



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
## 1. 使用[mobilenetv2](/home/md/cc/dms-train-tools/test_model/gender/model_46_20210516124136.pth)训练结果如下：
+ 1.混淆矩阵如下：
```
    [[1202   60]
    [53 2508]]
```
+ 2.指标如下：
```
              precision    recall  f1-score   support

      Female     0.9578    0.9525    0.9551      1262
        Male     0.9766    0.9793    0.9780      2561

    accuracy                         0.9704      3823
   macro avg     0.9672    0.9659    0.9665      3823
weighted avg     0.9704    0.9704    0.9704      3823
```

+ 3.训练参数命令
```
python NN_train_1.py-n mobilenetv2 -mp ./train_gender_model/gender_model/ -ds 2 --data-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/train/ --test-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/test -s 224 -dx 15 --mean-std 1 -b 32 --gpu 3 --pretrain
```
## 2.使用[desnet169](/home/md/cc/dms-train-tools/test_model/gender/model_36_20210608171138.pth)训练结果如下：
+ 1.混淆矩阵如下：
```
    [[1231   31]
    [  23 2538]]
```
+ 2.指标如下：
```

               precision    recall  f1-score   support

      Female     0.9817    0.9754    0.9785      1262
        Male     0.9879    0.9910    0.9895      2561

    accuracy                         0.9859      3823
   macro avg     0.9848    0.9832    0.9840      3823
weighted avg     0.9859    0.9859    0.9859      3823
```
+ 3.训练参数命令
```
python NN_train_focal_loss.py -n shufflenet10 -mp ./train_gender_model/gender_model/ -ds 2 --data-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/train/ --test-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/test -s 224 -dx 15 --mean-std 1 -b 32 --gpu 3   --focal_loss_weight 0:1.3,1:1 --pretrain
```

## 3.使用[resnet50](/home/md/cc/dms-train-tools/test_model/gender/model_33_20210602055020.pth)训练结果如下：
+ 1. 混淆矩阵如下：
```
    [[1205   57]
    [  34 2527]]
```
+ 2. 指标如下：

```

               precision    recall  f1-score   support

      Female     0.9726    0.9548    0.9636      1262
        Male     0.9779    0.9867    0.9823      2561

    accuracy                         0.9762      3823
   macro avg     0.9752    0.9708    0.9730      3823
weighted avg     0.9762    0.9762    0.9761      3823
```
+ 3.训练参数命令
+ 测试的命令帮助，python NN_train_1.py -h
```
python NN_train_1.py -n shufflenet10 -mp ./train_gender_model/gender_model/ -ds 2 --data-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/train/ --test-root /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/test -s 224 -dx 15 --mean-std 1 -b 32 --gpu 3 --pretrain
```

# 5 测试模型
+ 首先在/home/md/cc/dms-train-tools 目录下，输入
+ 1.workon cc_pytorch ，测试的命令帮助，python test_gender_camer.py -h
+ 2.测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu 

```
python eval_model.py -n mobilenetv2 -m /home/md/cc/dms-train-tools/test_model/gender/model_46_20210516124136.pth  -t /home/md/dataset/gender_dataset/AFAD_celeba_imdb_select_dataset/test --model-category classic --mean-std 1 --gpu 0
```
# 6 转换成onnx
+ 1.mobilenetv2转换成onnx模型

```
# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
CLASS_NUM=2
model_path='model_46_20210516124136.pth'
file_name='gender.onnx'
net = models.mobilenet_v2(num_classes=CLASS_NUM)
net.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
net.cpu()

input = torch.randn(1, 3, 224, 224)
#torch.onnx.export(net, input, file_name)
# torch.onnx.export(net, input, file_name, verbose=False, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])
torch.onnx.export(net, input, file_name, verbose=True,opset_version=11, keep_initializers_as_inputs=True, input_names=["input"], output_names=["output"])

```
# 7.使用摄像头测试模型
+ [测试代码在这里](/home/md/cc/dms-train-tools/gender_train/test_gender_camer.py)
+ 测试的命令帮助，python test_gender_camer.py -h
+ 测试的例子 -n 是使用的网络，-m是使用的模型的路径，--mean-std是均值，--gpu是使用哪一块gpu ,也可以使用cpu,使用的方式是 --detect-dev cpu
```
python test_gender_camer.py -n mobilenetv2 -m /home/chenchong/project/data_set/mutil-task/model/test_model/gender/model_0093_20200608055919.pth --mean-std 1 --gpu 0
```

