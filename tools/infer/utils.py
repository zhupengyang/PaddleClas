# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import base64
import shutil
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--multilabel", type=str2bool, default=False)

    # params for preprocess
    parser.add_argument("--resize_short", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # params for predict
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_profile", type=str2bool, default=False)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_num_threads", type=int, default=10)
    parser.add_argument("--hubserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--xpu_device_id", type=int, default=0)
    parser.add_argument("--all_xpu_device", type=int, default=2)
    parser.add_argument("--log_file", type=str, default="log.txt")

    # params for infer
    parser.add_argument("--model", type=str)
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument(
        "--load_static_weights",
        type=str2bool,
        default=False,
        help='Whether to load the pretrained weights saved in static mode')

    # parameters for pre-label the images
    parser.add_argument(
        "--pre_label_image",
        type=str2bool,
        default=False,
        help="Whether to pre-label the images using the loaded weights")
    parser.add_argument("--pre_label_out_idr", type=str, default=None)

    # parameters for test hubserving
    parser.add_argument("--server_url", type=str)

    # enable_calc_metric, when set as true, topk acc will be calculated
    parser.add_argument("--enable_calc_topk", type=str2bool, default=False)
    # groudtruth label path
    # data format for each line: $image_name $class_id
    parser.add_argument("--gt_label_path", type=str, default=None)

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    elif args.use_xpu:
        config.disable_gpu()
        config.enable_lite_engine()
        config.enable_xpu()
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
    config.set_cpu_math_library_num_threads(args.cpu_num_threads)

    if args.enable_profile:
        config.enable_profile()
    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=Config.Precision.Half
            if args.use_fp16 else Config.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


def preprocess(img, args):
    resize_op = ResizeImage(resize_short=args.resize_short)
    img = resize_op(img)
    crop_op = CropImage(size=(args.resize, args.resize))
    img = crop_op(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0
        normalize_op = NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        img = normalize_op(img)
    tensor_op = ToTensor()
    img = tensor_op(img)
    return img


def postprocess(batch_outputs, topk=5, multilabel=False):
    batch_results = []
    for probs in batch_outputs:
        if multilabel:
            index = np.where(probs >= 0.5)[0].astype('int32')
        else:
            index = probs.argsort(axis=0)[-topk:][::-1].astype("int32")
        clas_id_list = []
        score_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(probs[i].item())
        batch_results.append({"clas_ids": clas_id_list, "scores": score_list})
    return batch_results


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def get_image_list_from_label_file(image_path, label_file_path):
    imgs_lists = []
    gt_labels = []
    with open(label_file_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            image_name, label = line.strip("\n").split()
            label = int(label)
            imgs_lists.append(os.path.join(image_path, image_name))
            gt_labels.append(int(label))
    return imgs_lists, gt_labels


def calc_topk_acc(info_map):
    '''
    calc_topk_acc
    input:
        info_map(dict): keys are prediction and gt_label
    output:
        topk_acc(list): top-k accuracy list
    '''
    gt_label = np.array(info_map["gt_label"])
    prediction = np.array(info_map["prediction"])

    gt_label = np.reshape(gt_label, (-1, 1)).repeat(
        prediction.shape[1], axis=1)
    correct = np.equal(prediction, gt_label)
    topk_acc = []
    for idx in range(prediction.shape[1]):
        if idx > 0:
            correct[:, idx] = np.logical_or(correct[:, idx],
                                            correct[:, idx - 1])
        topk_acc.append(1.0 * np.sum(correct[:, idx]) / correct.shape[0])
    return topk_acc


def save_prelabel_results(class_id, input_file_path, output_dir):
    output_dir = os.path.join(output_dir, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_file_path, output_dir)


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


def b64_to_np(b64str, revert_params):
    shape = revert_params["shape"]
    dtype = revert_params["dtype"]
    dtype = getattr(np, dtype) if isinstance(str, type(dtype)) else dtype
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, dtype).reshape(shape)
    return data


def np_to_b64(images):
    img_str = base64.b64encode(images).decode('utf8')
    return img_str, images.shape
