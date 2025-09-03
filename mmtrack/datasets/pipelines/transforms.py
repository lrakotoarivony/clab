# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Normalize, Pad, RandomFlip, Resize

from mmtrack.core import crop_image

from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms.autoaugment import _apply_op

import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image


@PIPELINES.register_module()
class SeqCropLikeSiamFC(object):
    """Crop images as SiamFC did.

    The way of cropping an image is proposed in
    "Fully-Convolutional Siamese Networks for Object Tracking."
    `SiamFC <https://arxiv.org/abs/1606.09549>`_.

    Args:
        context_amount (float): The context amount around a bounding box.
            Defaults to 0.5.
        exemplar_size (int): Exemplar size. Defaults to 127.
        crop_size (int): Crop size. Defaults to 511.
    """

    def __init__(self, context_amount=0.5, exemplar_size=127, crop_size=511):
        self.context_amount = context_amount
        self.exemplar_size = exemplar_size
        self.crop_size = crop_size

    def crop_like_SiamFC(self,
                         image,
                         bbox,
                         context_amount=0.5,
                         exemplar_size=127,
                         crop_size=511):
        """Crop an image as SiamFC did.

        Args:
            image (ndarray): of shape (H, W, 3).
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
                Defaults to 0.5.
            exemplar_size (int): Exemplar size. Defaults to 127.
            crop_size (int): Crop size. Defaults to 511.

        Returns:
            ndarray: The cropped image of shape (crop_size, crop_size, 3).
        """
        padding = np.mean(image, axis=(0, 1)).tolist()

        bbox = np.array([
            0.5 * (bbox[2] + bbox[0]), 0.5 * (bbox[3] + bbox[1]),
            bbox[2] - bbox[0], bbox[3] - bbox[1]
        ])
        z_width = bbox[2] + context_amount * (bbox[2] + bbox[3])
        z_height = bbox[3] + context_amount * (bbox[2] + bbox[3])
        z_size = np.sqrt(z_width * z_height)

        z_scale = exemplar_size / z_size
        d_search = (crop_size - exemplar_size) / 2.
        pad = d_search / z_scale
        x_size = z_size + 2 * pad
        x_bbox = np.array([
            bbox[0] - 0.5 * x_size, bbox[1] - 0.5 * x_size,
            bbox[0] + 0.5 * x_size, bbox[1] + 0.5 * x_size
        ])

        x_crop_img = crop_image(image, x_bbox, crop_size, padding)
        return x_crop_img

    def generate_box(self, image, gt_bbox, context_amount, exemplar_size):
        """Generate box based on cropped image.

        Args:
            image (ndarray): The cropped image of shape
                (self.crop_size, self.crop_size, 3).
            gt_bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
            exemplar_size (int): Exemplar size. Defaults to 127.

        Returns:
            ndarray: Generated box of shape (4, ) in [x1, y1, x2, y2] format.
        """
        img_h, img_w = image.shape[:2]
        w, h = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]

        z_width = w + context_amount * (w + h)
        z_height = h + context_amount * (w + h)
        z_scale = np.sqrt(z_width * z_height)
        z_scale_factor = exemplar_size / z_scale
        w = w * z_scale_factor
        h = h * z_scale_factor
        cx, cy = img_w // 2, img_h // 2
        bbox = np.array(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            dtype=np.float32)

        return bbox

    def __call__(self, results):
        """Call function.

        For each dict in results, crop image like SiamFC did.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains cropped image and
            corresponding ground truth box.
        """
        outs = []
        for _results in results:
            image = _results['img']
            gt_bbox = _results['gt_bboxes'][0]

            crop_img = self.crop_like_SiamFC(image, gt_bbox,
                                             self.context_amount,
                                             self.exemplar_size,
                                             self.crop_size)
            generated_bbox = self.generate_box(crop_img, gt_bbox,
                                               self.context_amount,
                                               self.exemplar_size)
            generated_bbox = generated_bbox[None]

            _results['img'] = crop_img
            if 'img_shape' in _results:
                _results['img_shape'] = crop_img.shape
            _results['gt_bboxes'] = generated_bbox

            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqCropLikeStark(object):
    """Crop images as Stark did.

    The way of cropping an image is proposed in
    "Learning Spatio-Temporal Transformer for Visual Tracking."
    `Stark <https://arxiv.org/abs/2103.17154>`_.

    Args:
        crop_size_factor (list[int | float]): contains the ratio of crop size
            to bbox size.
        output_size (list[int | float]): contains the size of resized image
            (always square).
    """

    def __init__(self, crop_size_factor, output_size):
        self.crop_size_factor = crop_size_factor
        self.output_size = output_size

    def crop_like_stark(self, img, bbox, crop_size_factor, output_size):
        """Crop an image as Stark did.

        Args:
            image (ndarray): of shape (H, W, 3).
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            crop_size_factor (float): the ratio of crop size to bbox size
            output_size (int): the size of resized image (always square).

        Returns:
            img_crop_padded (ndarray): the cropped image of shape
                (crop_size, crop_size, 3).
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            pdding_mask (ndarray): the padding mask caused by cropping.
        """
        x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
        bbox_w, bbox_h = x2 - x1, y2 - y1
        cx, cy = x1 + bbox_w / 2., y1 + bbox_h / 2.

        img_h, img_w, _ = img.shape
        # 1. Crop image
        # 1.1 calculate crop size and pad size
        crop_size = math.ceil(math.sqrt(bbox_w * bbox_h) * crop_size_factor)
        crop_size = max(crop_size, 1)

        x1 = int(np.round(cx - crop_size * 0.5))
        x2 = x1 + crop_size
        y1 = int(np.round(cy - crop_size * 0.5))
        y2 = y1 + crop_size

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img_h + 1, 0)

        # 1.2 crop image
        img_crop = img[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # 1.3 pad image
        img_crop_padded = cv2.copyMakeBorder(img_crop, y1_pad, y2_pad, x1_pad,
                                             x2_pad, cv2.BORDER_CONSTANT)
        # 1.4 generate padding mask
        img_h, img_w, _ = img_crop_padded.shape
        pdding_mask = np.ones((img_h, img_w))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        pdding_mask[y1_pad:end_y, x1_pad:end_x] = 0

        # 2. Resize image and padding mask
        resize_factor = output_size / crop_size
        img_crop_padded = cv2.resize(img_crop_padded,
                                     (output_size, output_size))
        pdding_mask = cv2.resize(pdding_mask,
                                 (output_size, output_size)).astype(np.bool_)

        return img_crop_padded, resize_factor, pdding_mask

    def generate_box(self,
                     bbox_gt,
                     bbox_cropped,
                     resize_factor,
                     output_size,
                     normalize=False):
        """Transform the box coordinates from the original image coordinates to
        the coordinates of the cropped image.

        Args:
            bbox_gt (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            bbox_cropped (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            output_size (float): the size of output image.
            normalize (bool): whether to normalize the output box.
                Default to True.

        Returns:
            ndarray: generated box of shape (4, ) in [x1, y1, x2, y2] format.
        """
        assert output_size > 0
        bbox_gt_center = (bbox_gt[0:2] + bbox_gt[2:4]) * 0.5
        bbox_cropped_center = (bbox_cropped[0:2] + bbox_cropped[2:4]) * 0.5

        bbox_out_center = (output_size - 1) / 2. + (
            bbox_gt_center - bbox_cropped_center) * resize_factor
        bbox_out_wh = (bbox_gt[2:4] - bbox_gt[0:2]) * resize_factor
        bbox_out = np.concatenate((bbox_out_center - 0.5 * bbox_out_wh,
                                   bbox_out_center + 0.5 * bbox_out_wh),
                                  axis=-1)

        return bbox_out / output_size if normalize else bbox_out

    def __call__(self, results):
        """Call function. For each dict in results, crop image like Stark did.

        Args:
            results (list[dict]): list of dict from
                :obj:`mmtrack.base_sot_dataset`.

        Returns:
            List[dict]: list of dict that contains cropped image and
                the corresponding groundtruth bbox.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']
            gt_bbox = _results['gt_bboxes'][0]
            jittered_bboxes = _results['jittered_bboxes'][0]
            crop_img, resize_factor, padding_mask = self.crop_like_stark(
                image, jittered_bboxes, self.crop_size_factor[i],
                self.output_size[i])

            generated_bbox = self.generate_box(
                gt_bbox,
                jittered_bboxes,
                resize_factor,
                self.output_size[i],
                normalize=False)

            generated_bbox = generated_bbox[None]

            _results['img'] = crop_img
            if 'img_shape' in _results:
                _results['img_shape'] = crop_img.shape
            _results['gt_bboxes'] = generated_bbox
            _results['seg_fields'] = ['padding_mask']
            _results['padding_mask'] = padding_mask
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBboxJitter(object):
    """Bounding box jitter augmentation. The jittered bboxes are used for
    subsequent image cropping, like `SeqCropLikeStark`.

    Args:
        scale_jitter_factor (list[int | float]): contains the factor of scale
            jitter.
        center_jitter_factor (list[int | float]): contains the factor of center
            jitter.
        crop_size_factor (list[int | float]): contains the ratio of crop size
            to bbox size.
    """

    def __init__(self, scale_jitter_factor, center_jitter_factor,
                 crop_size_factor):
        self.scale_jitter_factor = scale_jitter_factor
        self.center_jitter_factor = center_jitter_factor
        self.crop_size_factor = crop_size_factor

    def __call__(self, results):
        """Call function.

        Args:
            results (list[dict]): list of dict from
                :obj:`mmtrack.base_sot_dataset`.

        Returns:
            list[dict]: list of dict that contains augmented images.
        """
        outs = []
        for i, _results in enumerate(results):
            gt_bbox = _results['gt_bboxes'][0]
            x1, y1, x2, y2 = np.split(gt_bbox, 4, axis=-1)
            bbox_w, bbox_h = x2 - x1, y2 - y1
            gt_bbox_cxcywh = np.concatenate(
                [x1 + bbox_w / 2., y1 + bbox_h / 2., bbox_w, bbox_h], axis=-1)

            crop_img_size = -1
            # avoid croped image size too small.
            count = 0
            while crop_img_size < 1:
                count += 1
                if count > 100:
                    print_log(
                        f'-------- bbox {gt_bbox_cxcywh} is invalid -------')
                    return None
                jittered_wh = gt_bbox_cxcywh[2:4] * np.exp(
                    np.random.randn(2) * self.scale_jitter_factor[i])
                crop_img_size = np.ceil(
                    np.sqrt(jittered_wh.prod()) * self.crop_size_factor[i])

            max_offset = np.sqrt(
                jittered_wh.prod()) * self.center_jitter_factor[i]
            jittered_center = gt_bbox_cxcywh[0:2] + max_offset * (
                np.random.rand(2) - 0.5)

            jittered_bboxes = np.concatenate(
                (jittered_center - 0.5 * jittered_wh,
                 jittered_center + 0.5 * jittered_wh),
                axis=-1)

            _results['jittered_bboxes'] = jittered_bboxes[None]
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBrightnessAug(object):
    """Brightness augmention for images.

    Args:
        jitter_range (float): The range of brightness jitter.
            Defaults to 0..
    """

    def __init__(self, jitter_range=0):
        self.jitter_range = jitter_range

    def __call__(self, results):
        """Call function.

        For each dict in results, perform brightness augmention for image in
        the dict.

        Args:
            results (list[dict]): list of dict that from
                :obj:`mmtrack.base_sot_dataset`.
        Returns:
            list[dict]: list of dict that contains augmented image.
        """
        brightness_factor = np.random.uniform(
            max(0, 1 - self.jitter_range), 1 + self.jitter_range)
        outs = []
        for _results in results:
            image = _results['img']
            image = np.dot(image, brightness_factor).clip(0, 255.0)
            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqGrayAug(object):
    """Gray augmention for images.

    Args:
        prob (float): The probability to perform gray augmention.
            Defaults to 0..
    """

    def __init__(self, prob=0.):
        self.prob = prob

    def __call__(self, results):
        """Call function.

        For each dict in results, perform gray augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented gray image.
        """
        outs = []
        gray_prob = np.random.random()
        for _results in results:
            if self.prob > gray_prob:
                grayed = cv2.cvtColor(_results['img'], cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
                _results['img'] = image

            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqShiftScaleAug(object):
    """Shift and rescale images and bounding boxes.

    Args:
        target_size (list[int]): list of int denoting exemplar size and search
            size, respectively. Defaults to [127, 255].
        shift (list[int]): list of int denoting the max shift offset. Defaults
            to [4, 64].
        scale (list[float]): list of float denoting the max rescale factor.
            Defaults to [0.05, 0.18].
    """

    def __init__(self,
                 target_size=[127, 255],
                 shift=[4, 64],
                 scale=[0.05, 0.18]):
        self.target_size = target_size
        self.shift = shift
        self.scale = scale

    def _shift_scale_aug(self, image, bbox, target_size, shift, scale):
        """Shift and rescale an image and corresponding bounding box.

        Args:
            image (ndarray): of shape (H, W, 3). Typically H and W equal to
                511.
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            target_size (int): Exemplar size or search size.
            shift (int): The max shift offset.
            scale (float): The max rescale factor.

        Returns:
            tuple(crop_img, bbox): crop_img is a ndarray of shape
            (target_size, target_size, 3), bbox is the corresponding ground
            truth box in [x1, y1, x2, y2] format.
        """
        img_h, img_w = image.shape[:2]

        scale_x = (2 * np.random.random() - 1) * scale + 1
        scale_y = (2 * np.random.random() - 1) * scale + 1
        scale_x = min(scale_x, float(img_w) / target_size)
        scale_y = min(scale_y, float(img_h) / target_size)
        crop_region = np.array([
            img_w // 2 - 0.5 * scale_x * target_size,
            img_h // 2 - 0.5 * scale_y * target_size,
            img_w // 2 + 0.5 * scale_x * target_size,
            img_h // 2 + 0.5 * scale_y * target_size
        ])

        shift_x = (2 * np.random.random() - 1) * shift
        shift_y = (2 * np.random.random() - 1) * shift
        shift_x = max(-crop_region[0], min(img_w - crop_region[2], shift_x))
        shift_y = max(-crop_region[1], min(img_h - crop_region[3], shift_y))
        shift = np.array([shift_x, shift_y, shift_x, shift_y])
        crop_region += shift

        crop_img = crop_image(image, crop_region, target_size)
        bbox -= np.array(
            [crop_region[0], crop_region[1], crop_region[0], crop_region[1]])
        bbox /= np.array([scale_x, scale_y, scale_x, scale_y],
                         dtype=np.float32)
        return crop_img, bbox

    def __call__(self, results):
        """Call function.

        For each dict in results, shift and rescale the image and the bounding
        box in the dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains cropped image and
            corresponding ground truth box.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']
            gt_bbox = _results['gt_bboxes'][0]

            crop_img, crop_bbox = self._shift_scale_aug(
                image, gt_bbox, self.target_size[i], self.shift[i],
                self.scale[i])
            crop_bbox = crop_bbox[None]

            _results['img'] = crop_img
            if 'img_shape' in _results:
                _results['img_shape'] = crop_img.shape
            _results['gt_bboxes'] = crop_bbox
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqColorAug(object):
    """Color augmention for images.

    Args:
        prob (list[float]): The probability to perform color augmention for
            each image. Defaults to [1.0, 1.0].
        rgb_var (list[list]]): The values of color augmentaion. Defaults to
            [[-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118]].
    """

    def __init__(self,
                 prob=[1.0, 1.0],
                 rgb_var=[[-0.55919361, 0.98062831, -0.41940627],
                          [1.72091413, 0.19879334, -1.82968581],
                          [4.64467907, 4.73710203, 4.88324118]]):
        self.prob = prob
        self.rgb_var = np.array(rgb_var, dtype=np.float32)

    def __call__(self, results):
        """Call function.

        For each dict in results, perform color augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented color image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                offset = np.dot(self.rgb_var, np.random.randn(3, 1))
                # bgr to rgb
                offset = offset[::-1]
                offset = offset.reshape(3)
                image = (image - offset).astype(np.float32)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBlurAug(object):
    """Blur augmention for images.

    Args:
        prob (list[float]): The probability to perform blur augmention for
            each image. Defaults to [0.0, 0.2].
    """

    def __init__(self, prob=[0.0, 0.2]):
        self.prob = prob

    def __call__(self, results):
        """Call function.

        For each dict in results, perform blur augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented blur image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                sizes = np.arange(5, 46, 2)
                size = np.random.choice(sizes)
                kernel = np.zeros((size, size))
                c = int(size / 2)
                wx = np.random.random()
                kernel[:, c] += 1. / size * wx
                kernel[c, :] += 1. / size * (1 - wx)
                image = cv2.filter2D(image, -1, kernel)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqResize(Resize):
    """Resize images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Resize` for
    detailed docstring.

    Args:
        share_params (bool): If True, share the resize parameters for all
            images. Defaults to True.
    """

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Resize` to resize
        image and corresponding annotations.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains resized results,
            'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys
            are added into result dict.
        """
        outs, scale = [], None
        for i, _results in enumerate(results):
            if self.share_params and i > 0:
                _results['scale'] = scale
            _results = super().__call__(_results)
            if self.share_params and i == 0:
                scale = _results['scale']
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqNormalize(Normalize):
    """Normalize images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Normalize` for
    detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Normalize` to
        normalize image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains normalized results,
            'img_norm_cfg' key is added into result dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    """Randomly flip for images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:RandomFlip` for
    detailed docstring.

    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.

        For each dict in results, call `RandomFlip` to randomly flip image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains flipped results, 'flip',
            'flip_direction' keys are added into the dict.
        """
        if self.share_params:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            flip = cur_dir is not None
            flip_direction = cur_dir

            for _results in results:
                _results['flip'] = flip
                _results['flip_direction'] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqPad(Pad):
    """Pad images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Pad` for detailed
    docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Pad` to pad image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains padding results,
            'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' keys are
            added into the dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):
    """Sequentially random crop the images & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether share the cropping parameters
            for the images.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 share_params=False,
                 bbox_clip_border=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': ['gt_labels', 'gt_instance_ids'],
            'gt_bboxes_ignore': ['gt_labels_ignore', 'gt_instance_ids_ignore']
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def get_offsets(self, img):
        """Random generate the offsets for cropping."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            offsets (tuple, optional): Pre-defined offsets for cropping.
                Default to None.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results['img_info']['crop_offsets'] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results

    def __call__(self, results):
        """Call function to sequentially randomly crop images, bounding boxes,
        masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        if self.share_params:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs

@PIPELINES.register_module()
class SeqRotate(object):
    """Sequentially rotate each image in a sequence by 90 degrees, 
    including its bounding boxes, masks, and segmentation maps.
    
    Args:
        prob (float): Probability of applying the rotation. Defaults to 1.0.
        angle (int): Rotation angle, set to 90 for this use case.
        img_fill_val (int | float | tuple): Fill value for image border. 
            Defaults to 128.
        seg_ignore_label (int): Ignore label for segmentation maps. Defaults to 255.
    """

    def __init__(self, prob=1.0, angle=90, img_fill_val=128, seg_ignore_label=255):
        self.prob = prob
        self.angle = angle
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image by 90 degrees."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_rotated = mmcv.imrotate(img, angle, center, scale, auto_bound=True, border_value=self.img_fill_val)
            results[key] = img_rotated
            results['img_shape'] = img_rotated.shape
        return results

    def _rotate_bboxes(self, results):
        """Rotate the bounding boxes by 90 degrees."""
        h, w, _ = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            # Convert bbox coordinates for a 90-degree rotation
            rotated_bboxes = np.zeros_like(bboxes)
            rotated_bboxes[:, 0] = bboxes[:, 1]
            rotated_bboxes[:, 1] = w - bboxes[:, 2]
            rotated_bboxes[:, 2] = bboxes[:, 3]
            rotated_bboxes[:, 3] = w - bboxes[:, 0]
            results[key] = rotated_bboxes
        return results

    def _rotate_masks(self, results, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks by 90 degrees."""
        h, w, _ = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)
        return results

    def _rotate_seg(self, results, angle, center=None, scale=1.0, fill_val=255):
        """Rotate the segmentation map by 90 degrees."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            results[key] = mmcv.imrotate(seg, angle, center, scale, auto_bound=True, border_value=fill_val).astype(seg.dtype)
        return results

    def __call__(self, results):
        """Apply a 90-degree rotation to each image in the sequence and adjust bboxes, masks, and segmentation maps.
        
        Args:
            results (list[dict]): List of result dicts, each containing image and annotation information.

        Returns:
            list[dict]: Rotated results.
        """
        img_shape = results[0]['img'].shape[:2]
        height, width = img_shape

        if height < width:
            return results

        rotated_results = []
        print('Need Rotate')
        for _results in results:
            h, w = _results['img'].shape[:2]
            #center = ((w - 1) * 0.5, (h - 1) * 0.5)
            center = None
            _results = self._rotate_img(_results, self.angle, center)
            _results = self._rotate_bboxes(_results)
            _results = self._rotate_masks(_results, self.angle, center)
            _results = self._rotate_seg(_results, self.angle, center, fill_val=self.seg_ignore_label)
            rotated_results.append(_results)
        
        print(rotated_results[0]['img'].shape[:2])
        return rotated_results

@PIPELINES.register_module()
class SeqRandomFlipAccurate(RandomFlip):
    """Randomly flip for images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:RandomFlip` for
    detailed docstring.

    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.

        For each dict in results, call `RandomFlip` to randomly flip image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains flipped results, 'flip',
            'flip_direction' keys are added into the dict.
        """
        img_shape = results[0]['img'].shape[:2]
        height, width = img_shape
        if self.share_params:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            if height > width:
                flip = True
                flip_direction = 'vertical'
            else:
                flip = cur_dir is not None
                flip_direction = cur_dir

            for _results in results:
                _results['flip'] = flip
                _results['flip_direction'] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs



@PIPELINES.register_module()
class SeqPad(Pad):
    """Pad images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Pad` for detailed
    docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Pad` to pad image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains padding results,
            'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' keys are
            added into the dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):
    """Sequentially random crop the images & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether share the cropping parameters
            for the images.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 share_params=False,
                 bbox_clip_border=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': ['gt_labels', 'gt_instance_ids'],
            'gt_bboxes_ignore': ['gt_labels_ignore', 'gt_instance_ids_ignore']
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def get_offsets(self, img):
        """Random generate the offsets for cropping."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            offsets (tuple, optional): Pre-defined offsets for cropping.
                Default to None.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results['img_info']['crop_offsets'] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results

    def __call__(self, results):
        """Call function to sequentially randomly crop images, bounding boxes,
        masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        if self.share_params:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs

@PIPELINES.register_module()
class SeqSquareRandomCrop(object):
    """Sequentially crop square images exceeding a certain size.

    This augmentation checks if the image is square and exceeds a given
    size threshold. If both conditions are met, the image is cropped to
    (height, 608). Otherwise, the image is returned unchanged.

    Args:
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether to share cropping parameters
            across the sequence of images. Default False.
        bbox_clip_border (bool, optional): Whether to clip bboxes that fall
            outside the cropped region. Default True.
        size_threshold (int, optional): The area threshold in pixels. If the
            square image's area exceeds this, it will be cropped. Default 612864.
    """

    def __init__(self,
                 allow_negative_crop=False,
                 share_params=False,
                 bbox_clip_border=True,
                 size_threshold=612864):
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        self.bbox_clip_border = bbox_clip_border
        self.size_threshold = size_threshold
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': ['gt_labels', 'gt_instance_ids'],
            'gt_bboxes_ignore': ['gt_labels_ignore', 'gt_instance_ids_ignore']
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Apply cropping if the image is square and exceeds the size threshold.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped results if conditions are met, or original images.
        """
        # Determine the height and width of the image
        img_shape = results[0]['img'].shape[:2]
        height, width = img_shape

        # Check if the image is square and exceeds the size threshold
        if height == width and height * width > self.size_threshold:
            print('Need Cropping')
            # Crop to (height, 608)
            crop_size = (height, 608)
            
            if self.share_params:
                # Generate offsets only once if sharing parameters
                offsets = self.get_offsets(results[0]['img'], crop_size)
            else:
                offsets = None
            
            outs = []
            for _results in results:
                _results = self.random_crop(_results, crop_size, offsets)
                if _results is None:
                    return None
                outs.append(_results)
            return outs
        
        # If not square or under the size threshold, return original results
        return results

    def get_offsets(self, img, crop_size):
        """Generate the offsets for cropping."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, crop_size, offsets=None):
        """Randomly crop images, bounding boxes, masks, and segmentation maps."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img, crop_size)
            results['img_info']['crop_offsets'] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # Crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # Crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # Update labels and masks
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # Crop semantic segmentation maps
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results


@PIPELINES.register_module()
class SeqAutoAugment(object):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def __call__(self, results):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(results[0], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(results[0]['img'])
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        transform_id, probs, signs = self.get_params(len(self.policies))


        outs = []
        for _results in results:
            img = _results['img']
            img_pil = Image.fromarray(img)
            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                if probs[i] <= p:
                    op_meta = self._augmentation_space(10, F.get_image_size(img_pil))
                    #op_meta = self._augmentation_space(10, img.shape[:2])
                    magnitudes, signed = op_meta[op_name]
                    magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    if signed and signs[i] == 0:
                        magnitude *= -1.0
                    img_pil = _apply_op(img_pil, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            img = np.asarray(img_pil)
            _results['img'] = img
            outs.append(_results)
            
        return outs


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"

@PIPELINES.register_module()
class SeqRandAugment(object):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, results):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(results[0], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(results[0]['img'])
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        selected_ops = []
        img_pil_start = Image.fromarray(results[0]['img'])
        op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img_pil_start))
        
        outs = []
        for idx, _results in enumerate(results):
            img = _results['img']
            img_pil = Image.fromarray(img)
            if idx == 0:
            # For the first image, select operations and magnitudes randomly
                for _ in range(self.num_ops):
                    op_index = int(torch.randint(len(op_meta), (1,)).item())
                    op_name = list(op_meta.keys())[op_index]
                    magnitudes, signed = op_meta[op_name]
                    magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0

                    # Save the selected operation and magnitude
                    selected_ops.append((op_name, magnitude))
                    
                    # Apply the selected operation to the first image
                    img_pil = _apply_op(img_pil, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            else:
                # For all other images, apply the same operations and magnitudes as the first image
                for op_name, magnitude in selected_ops:
                    img_pil = _apply_op(img_pil, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            img = np.asarray(img_pil)
            _results['img'] = img
            outs.append(_results)
            
        return outs

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

@PIPELINES.register_module()
class SeqAugMix(object):
    r"""AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity (int): The severity of base augmentation operators. Default is ``3``.
        mixture_width (int): The number of augmentation chains. Default is ``3``.
        chain_depth (int): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3].
            Default is ``-1``.
        alpha (float): The hyperparameter for the probability distributions. Default is ``1.0``.
        all_ops (bool): Use all operations (including brightness, contrast, color and sharpness). Default is ``True``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
        return s

    @torch.jit.unused
    def _pil_to_tensor(self, img) -> Tensor:
        return F.pil_to_tensor(img)

    @torch.jit.unused
    def _tensor_to_pil(self, img: Tensor):
        return F.to_pil_image(img)

    def _sample_dirichlet(self, params: Tensor) -> Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)
    
    def __call__(self, results):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(results[0], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(results[0]['img'])
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        img_pil_start = Image.fromarray(results[0]['img'])
        img_tensor_start = self._pil_to_tensor(img_pil_start)
        op_meta = self._augmentation_space(self._PARAMETER_MAX, F.get_image_size(img_pil_start))

        orig_dims = list(img_tensor_start.shape)
        batch = img_tensor_start.view([1] * max(4 - img_tensor_start.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        
        outs = []
        aug_params = []
        for i in range(self.mixture_width):
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            aug_chain = []
            for _ in range(depth):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[torch.randint(self.severity, (1,), dtype=torch.long)].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                aug_chain.append((op_name, magnitude))
            aug_params.append(aug_chain)

        outs = []
        for idx, _results in enumerate(results):
            img = _results['img']
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)

            mix = m[:, 0].view(batch_dims) * img.view([1] * max(4 - img.ndim, 0) + list(img.shape))
            
            for i, aug_chain in enumerate(aug_params):
                aug = img
                for op_name, magnitude in aug_chain:
                    aug = _apply_op(aug, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                mix.add_(combined_weights[:, i].view(batch_dims) * aug)
            
            mix = mix.view(orig_dims).to(dtype=img.dtype)
            #mix = mix.permute(1, 2, 0)
            mix = self._tensor_to_pil(mix)
            img = np.asarray(mix)
            _results['img'] = img
            outs.append(_results)
            
        return outs

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"severity={self.severity}"
            f", mixture_width={self.mixture_width}"
            f", chain_depth={self.chain_depth}"
            f", alpha={self.alpha}"
            f", all_ops={self.all_ops}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


@PIPELINES.register_module()
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 share_params=True,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        """Generate parameters."""
        params = dict()
        # delta
        if np.random.randint(2):
            params['delta'] = np.random.uniform(-self.brightness_delta,
                                                self.brightness_delta)
        else:
            params['delta'] = None
        # mode
        mode = np.random.randint(2)
        params['contrast_first'] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params['alpha'] = np.random.uniform(self.contrast_lower,
                                                self.contrast_upper)
        else:
            params['alpha'] = None
        # saturation
        if np.random.randint(2):
            params['saturation'] = np.random.uniform(self.saturation_lower,
                                                     self.saturation_upper)
        else:
            params['saturation'] = None
        # hue
        if np.random.randint(2):
            params['hue'] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params['hue'] = None
        # swap
        if np.random.randint(2):
            params['permutation'] = np.random.permutation(3)
        else:
            params['permutation'] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
            params (dict, optional): Pre-defined parameters. Default to None.

        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results['img_info']['color_jitter'] = params

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if params['delta'] is not None:
            img += params['delta']

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params['saturation'] is not None:
            img[..., 1] *= params['saturation']

        # random hue
        if params['hue'] is not None:
            img[..., 0] += params['hue']
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # randomly swap channels
        if params['permutation'] is not None:
            img = img[..., params['permutation']]

        results['img'] = img
        return results

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str
