import math
import torch


def makegrid(width, height):
    r = torch.arange(0, width)
    c = torch.arange(0, height)
    x = r.repeat(height).view(-1, 1)
    y = c.view(-1, 1).repeat(1, width).view(-1, 1)

    return torch.cat([x, y], 1)


def change_box_order(boxes, order):
    '''
	args:
	box: Tensor(n, 4)
	'''

    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a + b) / 2, b - a], 1)
    else:
        return torch.cat([a - b / 2, a + b / 2], 1)


def box_iou(box1, box2, order='xyxy'):
    # type: (object, object, object) -> object
    '''
	args:
	box1:Tensor(n, 4)
	box2:Tensor(m, 4)
	order: str, either xyxy or xywh

	returns:
	iou:Tensor(n, m)
	'''

    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')
    box1_num = box1.size(0)
    box2_num = box2.size(0)

    left_top = torch.max(box1[:, None, :2], box2[:, :2])
    right_bottom = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (box2_num, )

    iou = inter_area / (box1_area[:, None] + box2_area - inter_area)

    return iou


def resize(img, boxes, size):
    w, h = img.size
    nw, nh = size
    w_ratio, h_ratio = float(nw) / w, float(nh) / h
    img = img.resize(size)
    boxes[:, 0] = boxes[:, 0] * w_ratio
    boxes[:, 2] = boxes[:, 2] * w_ratio
    boxes[:, 1] = boxes[:, 1] * h_ratio
    boxes[:, 3] = boxes[:, 3] * h_ratio

    return img, boxes




'''
a = makegrid(5, 6)
print(a)

box1 = torch.Tensor([100,100,200,200]).view(-1,4)
box2 = torch.Tensor([150,150,250,250]).view(-1,4)
print(box1)
iou = box_iou(box1, box2)
print(iou)
'''
