'''
encode object boxes and labels
'''
from __future__ import division
import torch
import math

import utils


class DataEncoder:
    def __init__(self):
        self.anchor_sizes = [32 * 32.0, 64 * 64.0, 128 * 128.0, 256 * 256.0, 512 * 512.0]
        self.aspect_ratios = [1 / 2.0, 1.0, 2.0]
        self.scale_ratios = [1.0, pow(2, 1 / 3.0), pow(2, 2 / 3.0)]
        self.anchor_wh = self._get_anchor_wh()
        # based on input image

    def _get_anchor_wh(self):
        anchor_wh = []
        for size in self.anchor_sizes:
            for ar in self.aspect_ratios:
                h = math.sqrt(size / ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_w = w * sr
                    anchor_h = h * sr
                    anchor_wh.append([anchor_w, anchor_h])
        fms_num = len(self.anchor_sizes)

        return torch.Tensor(anchor_wh).view(fms_num, -1, 2)

    def _get_anchor_boxes(self, input_size):
        boxes = []
        input_size = torch.Tensor(input_size)
        fms_num = len(self.anchor_sizes)
        fms_size = [(input_size / pow(2.0, i + 3)).ceil() for i in xrange(fms_num)]
        for i in xrange(fms_num):
            fm_size = fms_size[i]
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            scale = input_size / fm_size
            xy = utils.makegrid(fm_w, fm_h) * scale + 0.5  # h*w 2
            xy = xy.view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)  # h*w,2 => h,w,1,2 => h,w,9,2
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)  # mapping to original image
            box = torch.cat([xy, wh], 3)
            boxes.append(box.view(-1, 4))

        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):

        """We obey the Faster RCNN box coder:
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)
        args:
        boxes:Tensor(xmin, ymin, xmax, ymax) size(boxes_num, 4)
        labels:Tensor size(boxes_num,)
        return:
        target_cls:Tensor(anchor_num,)
        target_loc:Tensor(anchor_num, 4)
        """

        anchor_boxes = self._get_anchor_boxes(input_size)  # [anchor_num, 4]
        boxes = utils.change_box_order(boxes, 'xyxy2xywh')
        ious = utils.box_iou(anchor_boxes, boxes, order='xywh')  # [anchor_num, boxes_num]
        max_ious, max_ids = ious.max(1)  # (anchor_num,)
        boxes = boxes[max_ids]  # (anchor_num, 4), groundtruth
        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        target_loc = torch.cat([loc_xy, loc_wh], 1)
        target_cls = labels[max_ids]
        target_cls[max_ious < 0.5] = 0
        ignore = (max_ious < 0.5) & (max_ious >= 0.4)
        target_cls[ignore] = -1

        return target_loc, target_cls


'''
encoder = DataEncoder()
print(encoder._get_anchor_boxes(torch.Tensor([600, 600])))
print(encoder.anchor_wh)
'''
