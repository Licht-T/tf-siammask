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
import PIL.Image
import PIL.ImageDraw

from .model import BaseNet, MaskRefinementNet


class SiamMask:
    def __init__(self):
        # CustomResNet50 output stride
        self.stride = 8

        self.anchor_box_ratios = np.array([0.33, 0.5, 1, 2, 3])
        self.anchor_box_scales = np.array([8])

        self.exampler_size = 127
        self.search_size = 255

        self.mask_pixels = 63

        self.kernel_cut_off_size = 8

        self.mask_threshold = 10

        self.num_anchors = None
        self.output_wh_size = None
        self.base_model = None
        self.mask_refinement_model = None
        self.anchors = None

        self.init_model()

    def init_model(self):
        self.num_anchors = len(self.anchor_box_ratios) * len(self.anchor_box_scales)
        self.output_wh_size = (self.search_size - self.exampler_size) // self.stride + self.kernel_cut_off_size + 1

        self.base_model = BaseNet(self.num_anchors, self.mask_pixels, self.kernel_cut_off_size)

        exampler_input = tf.keras.Input((self.exampler_size, self.exampler_size, 3))
        search_input = tf.keras.Input((self.search_size, self.search_size, 3))
        self.base_model([exampler_input, search_input])

        self.mask_refinement_model = MaskRefinementNet()
        mask_feature_input = tf.keras.Input((1, 1, 256))
        # From CustomResNet50 parameters
        res3_input = tf.keras.Input((15, 15, 512))
        res2_input = tf.keras.Input((31, 31, 256))
        res1_input = tf.keras.Input((61, 61, 64))
        self.mask_refinement_model([mask_feature_input, res3_input, res2_input, res1_input])

        anchors_wh = np.zeros((self.num_anchors, 2))
        for i, (ratio, scale) in enumerate(itertools.product(self.anchor_box_ratios, self.anchor_box_scales)):
            anchors_wh[i, 0] = int(self.stride / np.sqrt(ratio)) * scale
            anchors_wh[i, 1] = int(self.stride * np.sqrt(ratio)) * scale

        anchors_wh = np.tile(
            anchors_wh,
            self.output_wh_size * self.output_wh_size
        ).reshape((self.output_wh_size, self.output_wh_size, self.num_anchors, 2))

        xy_grid = np.meshgrid(range(self.output_wh_size), range(self.output_wh_size))
        xy_grid = self.stride * (np.stack(xy_grid, -1) - self.output_wh_size // 2)
        xy_grid = np.broadcast_to(
            xy_grid[..., np.newaxis, :],
            (self.output_wh_size, self.output_wh_size, self.num_anchors, 2)
        )

        self.anchors = np.concatenate([xy_grid, anchors_wh], -1)

    def load_weights(self, base_model_fp: str, mask_refinement_model_fp: str):
        self.base_model.load_weights(base_model_fp)
        self.mask_refinement_model.load_weights(mask_refinement_model_fp)

    def predict(self, img: np.ndarray, box: np.ndarray, debug=False):
        box_wh = (box[1] - box[0]).astype(np.int64)
        box_center = box.mean(0).astype(np.int64)

        exampler_box_size = np.array([box_wh.max() * 1.1, box_wh.max() * 1.1], dtype=np.int64)

        exampler_box = np.array([box_center - exampler_box_size/2, box_center + exampler_box_size/2], dtype=np.int64)
        search_box = np.array([box_center - exampler_box_size, box_center + exampler_box_size], dtype=np.int64)

        im = PIL.Image.fromarray(img[..., ::-1])

        scale = exampler_box_size / self.exampler_size
        box_ratio = box_wh[0] / box_wh[1]
        box_area = np.sqrt(box_wh[0] * box_wh[1] / (scale[0] * scale[1]))

        im_exampler = im.crop(exampler_box.flatten().tolist()).resize((self.exampler_size, self.exampler_size))
        im_search = im.crop(search_box.flatten().tolist()).resize((self.search_size, self.search_size))
        exampler = np.array(im_exampler)[..., ::-1]
        search = np.array(im_search)[..., ::-1]

        predicted_box, _, predicted_mask = self._predict(exampler, search, box_ratio, box_area)

        predicted_box = scale[np.newaxis, ...] * predicted_box + box_center[np.newaxis, ...]
        predicted_box = predicted_box.astype(np.int64)

        # predicted_mask[predicted_mask >= self.mask_threshold] = 255
        # predicted_mask[predicted_mask < self.mask_threshold] = 0

        im_predicted_mask = PIL.Image.fromarray(predicted_mask).resize((*exampler_box_size,))

        im_mask = PIL.Image.new('L', im.size)
        # FIXME: Box回帰ずみのものをつかわないようにする
        im_mask.paste(
            im_predicted_mask,
            (*(predicted_box.mean(axis=0) - self.exampler_size * scale / 2).astype(np.int64),)
        )

        if debug:
            im_exampler.convert('RGB').save('exampler.png')
            im_search.convert('RGB').save('search.png')

            draw = PIL.ImageDraw.Draw(im)
            draw.rectangle((*predicted_box.flatten(),))
            im.save('predicted_box.png')

            im_predicted_mask.convert('RGB').save('predicted_mask_patch.png')
            im_mask.convert('RGB').save('predicted_mask.png')

        return predicted_box, np.array(im_mask)

    def _predict(self, exampler: np.ndarray, search: np.ndarray, prev_box_ratio, prev_box_area):
        exampler = exampler[np.newaxis, ...].astype(np.float32)
        search = search[np.newaxis, ...].astype(np.float32)
        scores, boxes, masks, mask_features, residuals = self.base_model([exampler, search])

        scores = np.squeeze(scores.numpy())
        boxes = np.squeeze(boxes.numpy())
        masks = np.squeeze(masks.numpy())

        boxes[..., :2] = boxes[..., :2] * self.anchors[..., 2:] + self.anchors[..., :2]
        boxes[..., 2:] = boxes[..., 2:] * self.anchors[..., 2:]

        box_ratios = boxes[..., 2] / boxes[..., 3]
        box_areas = np.sqrt(boxes[..., 2] * boxes[..., 3])

        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]

        def diff(r):
            return np.maximum(r, 1. / r)

        penalty = np.exp(-(diff(box_ratios / prev_box_ratio) * diff(box_areas / prev_box_area) - 1) * 0.04)

        box_idx = np.unravel_index((scores * penalty).argmax(), boxes.shape[:-1])
        idx = box_idx[:2]

        box = boxes[box_idx].reshape((2, 2))
        mask = masks[idx].reshape((self.mask_pixels, self.mask_pixels))
        mask = np.clip(255 * mask, 0, 255).astype(np.uint8)

        mask_refinement_inputs = [
            tf.reshape(mask_features[[0, *idx]], (1, 1, 1, tf.shape(mask_features)[-1]))
        ]

        # FIXME: もっとわかりやすくかけるはず
        for residual, pad, mn in zip(residuals, [4, 8, 16], [15, 31, 61]):
            frm_h = (pad // 4) * idx[0]
            to_h = frm_h + mn
            frm_w = (pad // 4) * idx[1]
            to_w = frm_w + mn

            residual = tf.pad(residual, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'CONSTANT')
            mask_refinement_inputs.append(residual[:, frm_h:to_h, frm_w:to_w, :])

        refined_mask = np.squeeze(self.mask_refinement_model(mask_refinement_inputs).numpy())
        refined_mask = np.clip(255 * refined_mask, 0, 255).astype(np.uint8)

        return box, mask, refined_mask
