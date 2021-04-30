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
import numpy as np
import cv2
import time

import sys
sys.path.insert(0, ".")
from ppcls.utils import logger
from tools.infer.utils import parse_args, create_paddle_predictor, preprocess, postprocess
from tools.infer.utils import get_image_list, get_image_list_from_label_file, calc_topk_acc


class Predictor(object):
    def __init__(self, args):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args

        self.paddle_predictor = create_paddle_predictor(args)
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def predict(self, batch_input):
        self.input_tensor.copy_from_cpu(batch_input)
        self.paddle_predictor.run()
        batch_output = self.output_tensor.copy_to_cpu()
        return batch_output

    def normal_predict(self):
        images_num = 0
        if self.args.enable_calc_topk:
            assert self.args.gt_label_path is not None and os.path.exists(self.args.gt_label_path), \
                "gt_label_path shoule not be None and must exist, please check its path."
            image_list, gt_labels = get_image_list_from_label_file(
                self.args.image_file, self.args.gt_label_path)
            xpu_device_id = self.args.xpu_device_id
            images_num = int(len(image_list) / self.args.all_xpu_device)
            image_list = image_list[xpu_device_id * images_num : (xpu_device_id + 1) * images_num]
            gt_labels = gt_labels[xpu_device_id * images_num : (xpu_device_id + 1) * images_num]
            predicts_map = {
                "prediction": [],
                "gt_label": [],
            }
        else:
            image_list = get_image_list(self.args.image_file)
            gt_labels = None

        batch_input_list = []
        img_name_list = []
        cnt = 0
        all_time = 0.
        for idx, img_path in enumerate(image_list):
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(
                    "Image file failed to read and has been skipped. The path: {}".
                    format(img_path))
                continue
            else:
                img = img[:, :, ::-1]
                img = preprocess(img, args)
                batch_input_list.append(img)
                img_name = img_path.split("/")[-1]
                img_name_list.append(img_name)
                cnt += 1
                if self.args.enable_calc_topk:
                    predicts_map["gt_label"].append(gt_labels[idx])

            if cnt % args.batch_size == 0 or (idx + 1) == len(image_list):
                if cnt / args.batch_size <= 1 or (idx + 1) == len(image_list):
                    for i in range(self.args.warmup):
                        batch_outputs = self.predict(np.array(batch_input_list))
                        print("warmup {}, done".format(i))

                start_time = time.time()
                batch_outputs = self.predict(np.array(batch_input_list))
                all_time = all_time + time.time() - start_time
                avg_time = all_time / cnt
                print("device {} cost time avg: {} ms/images, throughput: {} images/s".format(self.args.xpu_device_id, avg_time * 1000, 1 / avg_time))

                batch_result_list = postprocess(batch_outputs, self.args.top_k)

                for number, result_dict in enumerate(batch_result_list):
                    filename = img_name_list[number]
                    clas_ids = result_dict["clas_ids"]
                    # scores_str = "[{}]".format(", ".join("{:.2f}".format(
                    #     r) for r in result_dict["scores"]))
                    # logger.info(
                    #     "File:{}, Top-{} result: class id(s): {}, score(s): {}".
                    #     format(filename, self.args.top_k, clas_ids,
                    #            scores_str))

                    if self.args.enable_calc_topk:
                        predicts_map["prediction"].append(clas_ids)

                batch_input_list = []
                img_name_list = []

        avg_time = all_time / cnt
        out_log = "device {} summary:\n".format(self.args.xpu_device_id) \
                + "\twarmup: {}\n".format(self.args.warmup) \
                + "\timages num: {}\n".format(images_num) \
                + "\taverage: {} ms/batch\n".format(avg_time * 1000) \
                + "\tthroughput: {} batch/s\n".format(1 / avg_time)
        print(out_log)
        np.savetxt(self.args.log_file, np.array(out_log, str), "%s")

        if self.args.enable_calc_topk:
            topk_acc = calc_topk_acc(predicts_map)
            for idx, acc in enumerate(topk_acc):
                logger.info("Top-{} acc: {:.5f}".format(idx + 1, acc))
                with open(self.args.log_file, "ab") as f:
                    log_str = "Top-{} acc: {:.5f}".format(idx + 1, acc)
                    np.savetxt(f, np.array([log_str], str), "%s")

    def benchmark_predict(self):
        test_num = 500
        test_time = 0.0
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            batch_output = self.predict(inputs).flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if args.use_fp16 else "FP32"
        trt_msg = "using tensorrt" if args.use_tensorrt else "not using tensorrt"
        print("{0}\t{1}\t{2}\tbatch size: {3}\ttime(ms): {4}".format(
            args.model, trt_msg, fp_message, args.batch_size, 1000 * test_time
            / test_num))


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(
        args.model_file), "The path of 'model_file' does not exist: {}".format(
            args.model_file)
    assert os.path.exists(
        args.params_file
    ), "The path of 'params_file' does not exist: {}".format(args.params_file)

    predictor = Predictor(args)
    if not args.enable_benchmark:
        predictor.normal_predict()
    else:
        assert args.model is not None
        predictor.benchmark_predict()
