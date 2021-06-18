# -*- coding: utf-8 -*-

#%%
import cv2
from PIL import Image
import time
import os
import sys, getopt
import torchvision.models as models
import torch
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import mobilenetv3 as mn
import numpy as np
from common import *
from eval_model import INVERTED_RESIDUAL_SETTING

curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(curr_dir, 'RetinaFace'))
from retinaface import RetinaFace
import torch.nn as nn

CENTER_SIZE = 320
# CLASSES = ('normal', 'drinking', 'phoning', 'smoking', 'muting')
# CLASSES = ('Female', 'Male')
# CLASSES = ('normal', 'happy', 'sad', 'anger', 'surprised', 'fear', 'disgust')
CLASSES = ('child', 'adult','elderly')
# CLASSES = ('closed_eye', 'closed_eye')

def detect(detector, img):
    thresh = 0.5
    scales = [512, 512]
    target_size = scales[0]
    max_size = scales[1]
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

    return faces, landmarks   

    
def draw_info(frame, detect_region, faces, predicted, data):
    font = cv2.FONT_HERSHEY_PLAIN

    ####
    for face in faces:
        left = int(face[0])
        top = int(face[1])
        right = int(face[2])
        bottom = int(face[3])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 1)
        
    # Draw a box around the face
    if detect_region != None:
        left = detect_region[0]
        top = detect_region[1]
        right = detect_region[2]
        bottom = detect_region[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 1)

    # Draw bar
    if data != None and len(data) > 0:
        for idx in range(len(data)):
            left = 120
            top = 10 + 25*idx
            right = left + 50 + int(data[idx] * 2)
            bottom = top + 15

            if idx == predicted:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, '%d %s' % (idx, CLASSES[idx]), (10, bottom - 2), font, 1.2, color, 2)
        # end of: for idx in len(data):
    # end of: if data != None:

def show(cam_id, res, dms_net, detect_net, size, mean_std, output_dir, device = -1):
    # Get a reference to webcam #0 (the default one)
    font = cv2.FONT_HERSHEY_PLAIN
    video_capture = cv2.VideoCapture(cam_id)
    # video_capture = cv2.VideoCapture('/home/cc/project/dms-train-tools-R-12058/test_model/video/test.mp4')

    if res != None:
        res = res.split('x')
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(res[0]))
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(res[1]))

    count = 0
    transform = get_transform(0, size, mean_std)
    detect_region = None
    saved_frame = None
    while True:
        # video capture
        t0 = time.time() * 1000
        ret, frame = video_capture.read()
        if not ret:
            print('video_capture failed.')
            break
            
        # flip
        t1 = time.time() * 1000
        frame = cv2.flip(frame, 1)

        # face detect
        # if detect_region == None:
        #     frame_size = frame.shape[:2]
        #     w = frame_size[1]
        #     h = frame_size[0]
        #     left = (w - CENTER_SIZE) // 2
        #     top = (h - CENTER_SIZE) // 2
        #     right = left + CENTER_SIZE
        #     bottom = top + CENTER_SIZE

        #     detect_region = [left, top, right, bottom]
        # end of: if detect_region == None:
            
        t2 = time.time() * 1000
        # detect_frame = frame[detect_region[1]:detect_region[3],detect_region[0]:detect_region[2]]
        faces, landmarks = detect(detect_net, frame)

        if len(faces) > 0:
            for face in faces:
                # face[0] += detect_region[0]
                # face[1] += detect_region[1]
                # face[2] += detect_region[0]
                # face[3] += detect_region[1]

                # face = faces[0]

                w = int(face[2] - face[0])
                h = int(face[3] - face[1])
                # detect_region = [
                #     max(0, int(face[0]) - w // 2),
                #     max(0, int(face[1]) - 30),
                #     int(face[2]) + w // 2,
                #     int(face[3]) + w // 2]
                detect_region = [
                    max(0, int(face[0]) - w // 3),
                    max(0, int(face[1]) - 30),
                    int(face[2]) + w // 3,
                    int(face[3]) + w // 3]
            
                t3 = time.time() * 1000
                detect_frame = frame[detect_region[1]:detect_region[3],detect_region[0]:detect_region[2]]
                img = Image.fromarray(cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB))

                t4 = time.time() * 1000
                data = transform(img).unsqueeze(0)
                if torch.cuda.is_available():
                    data = data.cuda(arg_gpu)

                t5 = time.time() * 1000
                outputs = dms_net(Variable(data))
                if torch.cuda.is_available():
                    outputs = outputs.cpu()

                _, predicted = torch.max(outputs.data, 1)

                t6 = time.time() * 1000
                print('gender_classifer:',outputs)
                # Display the resulting image
                predict_indicator = predicted.item()
                predict_values = outputs.data.squeeze(0).numpy().tolist()
                cv2.rectangle(frame, (detect_region[0], detect_region[1]), (detect_region[2], detect_region[3]), (0, 255, 255), 1)
                cv2.putText(frame, '%d %s' % (predict_indicator, CLASSES[predict_indicator]), (detect_region[0], detect_region[1]), font, 1.2, (0, 255, 255), 2)
        # else:
        #     detect_region = None
        #     t3 = time.time() * 1000
        #     t4 = time.time() * 1000
        #     t5 = time.time() * 1000
        #     t6 = time.time() * 1000
        #     predict_indicator = None
        #     predict_values = None

        saved_frame = frame.copy()
        # draw_info(frame, detect_region, faces, predict_indicator, predict_values)
        cv2.imshow('Video', frame)
        t7 = time.time() * 1000

        # count += 1
        # if count % 20 == 0:
        #     print('frame: %.1f ms; (capture: %.1f ms; flip: %.1f ms; detect: %.1f ms; convert: %.1f ms; transform: %.1f ms; inference: %.1f ms; show: %.1f ms)' % (t7-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6))

        # Hit 'q' on the keyboard to quit!
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key >= ord('0') and key <= ord('4'):
            save_jpg(saved_frame, predict_indicator, key - ord('0'), output_dir)
            saved_frame = None
        

    video_capture.release()
    cv2.destroyAllWindows()
    # saved_frame = None

def save_jpg(frame, predicted, expected, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_dir = os.path.join(output_dir, '%s-%s' % (expected, CLASSES[expected]))
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    jpg_file = os.path.join(sub_dir, '%s_%s-%s.jpg' % (now, predicted, CLASSES[predicted]))

    cv2.imwrite(jpg_file, frame)

def load_detect_net(device):
    if device == None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    device_map = {
        'cpu': -1,
        'cuda': 0
    }
    return RetinaFace("./model/mnet.25", 0, device_map[device])

def load_net(nn_type, num_classes, model_path, mobilenetv2_fix_irs, mobilenetv2_width_mult):
    if nn_type == 'mobilenetv3_small':
        net = mn.MobileNetV3_Small(num_classes = num_classes)
    elif nn_type == 'mobilenetv3_large':
        net = mn.MobileNetV3_Large(num_classes = num_classes)
    elif nn_type == 'mobilenetv2':
        inverted_residual_setting = None
        if mobilenetv2_fix_irs:
            inverted_residual_setting = INVERTED_RESIDUAL_SETTING
        net = models.MobileNetV2(num_classes = num_classes,
                                inverted_residual_setting = inverted_residual_setting,
                                width_mult = mobilenetv2_width_mult)
    elif nn_type == 'squeezenet1':
        net = models.SqueezeNet(version = '1_0', num_classes = num_classes)
    elif nn_type == 'squeezenet11':
        net = models.SqueezeNet(version = '1_1', num_classes = num_classes)
    elif arg_nn_type == 'shufflenet10':
        #shufflenet10
        # net = models.shufflenet_v2_x1_0(num_classes = num_classes)

        #googlenet
        # net = models.googlenet(pretrained=True)
        # net.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Conv2d(512, num_classes, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )

        #googlenet
        # net = models.googlenet(pretrained=True)
        # net.fc = nn.Linear(in_features=1024, out_features=2, bias=True)

        #resnet50
        # net = models.resnet50(pretrained=True)
        # num_ftrs = net.fc.in_features
        # net.fc = nn.Linear(num_ftrs, 2)

        # densenet121
        # net = models.densenet121(pretrained=arg_pretrain)
        # net.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
        # densenet169
        net = models.densenet169(pretrained=True)
        net.classifier = nn.Linear(in_features=1664, out_features=2, bias=True)
    else:
        print("Invalid nn type")
        net = None
    
    if net != None:
        if torch.cuda.is_available():
            device = torch.device('cuda:%s' % arg_gpu)
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.cuda(arg_gpu)
        else:
            device = torch.device('cpu')
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.cpu()

        net.eval()

    return net

def help():
    print('python CameraShow.py [options]')
    print(' options:')
    print('     -c: Camera id。缺省为0')
    print('     -n, --nntype: 网络类型，可以为:')
    print('             mobilenetv3_small')
    print('             mobilenetv3_large')
    print('             mobilenetv2')
    print('             squeezenet1 (version 1.0)')
    print('             squeezenet11 (version 1.1)')
    print('             shufflenet10')
    print('     -m, --mp: model文件路径')
    print('     --mean-std: Normalize的时候均值和方差')
    print('             0: mean=[.5, .5, .5], std=[.5, .5, .5] (缺省)')
    print('             1: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]')
    print('     -s, --size: 图像像素点size，图像为正方形，缺省为 224 x 224。')
    print('     --res: 图片输出路径。缺省为摄像头默认分辨率')
    print('     --detect-dev: 指定人脸检测网络运行设备，可以为cpu或者cuda，缺省按设备能力自动选择。DMS网络按硬件设备能力自动选择。')
    print('     --output-dir: 图片输出路径。缺省为: ./pictures/')
    print('     --gpu: 指定gpu设备编号。如果不指定，缺省为0')
    print('     --mobilenetv2-fix-irs: 是否要修改mobilenetv2的网络结构（论文可能错误，不确定pytorch实现是否修改）')
    print('     --mobilenetv2-width-mult: mobilenetv2 网络宽度系数。缺省为1.0')
    print('     -h: 显示帮助信息')

#%%
if __name__ == '__main__':
    arg_cam_id = 0
    arg_nn_type = None
    arg_model_path = None
    arg_mean_std = 0
    arg_size = 224
    arg_output_dir = './pictures'
    arg_res = None
    arg_detect_dev = None
    arg_gpu = None
    arg_mobilenetv2_fix_irs = False
    arg_mobilenetv2_width_mult = 1.0

    FOR_TEST = False
    if FOR_TEST:
        # TEST ONLY
        arg_cam_id = 0
        arg_nn_type = 'mobilenetv2'
        #arg_model_path = os.path.expanduser('/media/data-disk/models/iauto_bigcut_20191227_merged/model_20200101080804/model_210_20200102094439.pth')
        #arg_model_path = os.path.expanduser('~/Documents/model_210_20200102094439.pth')
        arg_model_path = os.path.expanduser('~/Documents/model_231_20200118185254.pth')
        arg_mean_std = 1
        arg_detect_dev = 'cpu'
        # END TEST ONLY
    else:
        try:
            argv = sys.argv[1:]
            opts, args = getopt.getopt(argv,"hc:n:m:s:", ["nntype=", "mp=", "mean-std=", "size=", "output-dir=", "res=",
                                                        "detect-dev=", "gpu=", "mobilenetv2-fix-irs", "mobilenetv2-width-mult="])
        except getopt.GetoptError:
            help()
            sys.exit(1)
        for opt, arg in opts:
            if opt == '-h':
                help()
                sys.exit()
            elif opt in ("-c",):
                arg_cam_id = int(arg)
            elif opt in ("-n", "--nntype",):
                arg_nn_type = arg
            elif opt in ("-m", "--mp",):
                arg_model_path = arg
            elif opt in ("--mean-std",):
                arg_mean_std = int(arg)
            elif opt in ("-s", "--size",):
                arg_size = int(arg)
            elif opt in ("--output-dir",):
                arg_output_dir = arg
            elif opt in ("--res",):
                arg_res = arg
            elif opt in ("--detect-dev",):
                arg_detect_dev = arg
            elif opt in ("--gpu",):
                arg_gpu = int(arg)
            elif opt in ("--mobilenetv2-fix-irs",):
                arg_mobilenetv2_fix_irs = True
            elif opt in ("--mobilenetv2-width-mult",):
                arg_mobilenetv2_width_mult = float(arg)
            else:
                print("Invalid argument!")
                help()
                sys.exit(1)
    # end of: if FOR_TEST:

    if arg_gpu != None:
        gpu_count = torch.cuda.device_count()
        if arg_gpu < 0 or arg_gpu >= gpu_count:
            print('Invalid gpu number. Available devices:')
            for gpu_idx in range(torch.cuda.device_count()):
                print('    %s %s' % (gpu_idx, torch.cuda.get_device_name(gpu_idx)))
            sys.exit(1)
    # end of: if arg_gpu != None:

    detect_net = load_detect_net(arg_detect_dev)

    dms_net = load_net(arg_nn_type, len(CLASSES), arg_model_path, arg_mobilenetv2_fix_irs, arg_mobilenetv2_width_mult)
    if dms_net == None:
        print("Invalid nn type!")
        help()
        sys.exit(1)

    show(arg_cam_id, arg_res, dms_net, detect_net, arg_size, arg_mean_std, arg_output_dir)
