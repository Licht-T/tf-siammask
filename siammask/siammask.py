"""
MIT License

Copyright (c) 2020 Licht Takeuchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import itertools

import tensorflow as tf
import numpy as np

from .model import Prediction


class SiamMask:
    def __init__(self):
        # CustomResNet50 output stride
        self.stride = 8

        self.anchor_box_ratios = np.array([0.33, 0.5, 1, 2, 3])
        self.anchor_box_scales = np.array([8])

        self.exampler_size = 127
        self.search_size = 255

        self.kernel_cut_off_size = 8

        self.num_anchors = None
        self.output_wh_size = None
        self.model = None
        self.anchors = None

        self.init_model()

    def init_model(self):
        self.num_anchors = len(self.anchor_box_ratios) * len(self.anchor_box_scales)
        self.output_wh_size = (self.search_size - self.exampler_size) // self.stride + self.kernel_cut_off_size + 1

        self.model = Prediction(self.num_anchors, self.kernel_cut_off_size)

        exampler_input = tf.keras.Input((self.exampler_size, self.exampler_size, 3))
        search_input = tf.keras.Input((self.search_size, self.search_size, 3))
        self.model([exampler_input, search_input])

        anchors_wh = np.zeros((self.num_anchors, 2))
        for i, (ratio, scale) in enumerate(itertools.product(self.anchor_box_ratios, self.anchor_box_scales)):
            anchors_wh[i, 0] = int(self.stride / np.sqrt(ratio)) * scale
            anchors_wh[i, 1] = int(self.stride * np.sqrt(ratio)) * scale

        anchors_wh = np.tile(anchors_wh, self.output_wh_size * self.output_wh_size).reshape((self.output_wh_size, self.output_wh_size, self.num_anchors, 2))

        xy_grid = np.meshgrid(range(self.output_wh_size), range(self.output_wh_size))
        xy_grid = self.stride * (np.stack(xy_grid, -1) - self.output_wh_size // 2)
        xy_grid = np.broadcast_to(xy_grid[..., np.newaxis, :], (self.output_wh_size, self.output_wh_size, self.num_anchors, 2))

        self.anchors = np.concatenate([xy_grid, anchors_wh], -1)

    def load_weights(self, fp: str):
        self.model.load_weights(fp)

    def predict(self, exampler, search):
        scores, boxes, masks = self.model([exampler, search], 1)

        scores = np.squeeze(scores.numpy())
        boxes = np.squeeze(boxes.numpy())
        masks = np.squeeze(masks.numpy())

        boxes[..., :2] = boxes[..., :2] * self.anchors[..., 2:] + self.anchors[..., :2]
        boxes[..., 2:] = boxes[..., 2:] * self.anchors[..., 2:]

        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]

        print(boxes.reshape((-1, 4)).shape, scores.reshape((-1,)).shape)

        idx = tf.image.non_max_suppression(boxes.reshape((-1, 4)), scores.reshape((-1,)), 1)[0]

        return boxes.reshape((-1, 4))[idx], 255 * np.clip(masks.reshape((-1, 63, 63))[idx//self.num_anchors], 0, 1)
