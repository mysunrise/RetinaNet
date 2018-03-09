import os
import shutil
import xml.dom.minidom

voc07_imgdir = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2007/JPEGImages'
voc07_annotations = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2007/Annotations'
voc07_trainval = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
voc07_test = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
voc12_imgdir = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2012/JPEGImages'
voc12_annotations = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2012/Annotations'
voc12_trainval = '/media/zhaoyang/01b8193f-c9cb-4777-b187-b4b5f28ed2db/data/data/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
train_dir = '/home/zhaoyang/data/voc0712/train'
test_dir = '/home/zhaoyang/data/voc0712/test'
annotation_dir = '/home/zhaoyang/data/voc0712/annotation'
train_annotaion = '/home/zhaoyang/data/voc0712/annotation/train_annotation.txt'
test_annotaion = '/home/zhaoyang/data/voc0712/annotation/test_annotation.txt'
label_annotaion = '/home/zhaoyang/data/voc0712/annotation/label_annotation.txt'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

train_ann = open(train_annotaion, 'w')
test_ann = open(test_annotaion, 'w')
label_ann = open(label_annotaion, 'r')

labels = label_ann.readlines()
label_num = len(labels)
label = {}
for l in labels:
    l = l.split()
    label[l[0]] = int(l[1])

with open(voc07_trainval) as f:
    lines = f.readlines()
    for line in lines:
        img_name = line.strip()
        train_ann.write(img_name + '.jpg' + ' ')
        img_file = os.path.join(voc07_imgdir, img_name + '.jpg')
        annotation_file = voc07_annotations + '/' + img_name + '.xml'
        if os.path.isfile(annotation_file):
            shutil.copy(os.path.join(voc07_imgdir, img_name + '.jpg'),
                        os.path.join(train_dir, img_name + '.jpg'))
            dom_tree = xml.dom.minidom.parse(annotation_file)
            annotation = dom_tree.documentElement
            object_list = annotation.getElementsByTagName('object')
            for obj in object_list:
                name_list = obj.getElementsByTagName('name')
                object_name = name_list[0].childNodes[0].data
                bndbox = obj.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(float(x1_list[0].childNodes[0].data))
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(float(y1_list[0].childNodes[0].data))
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(float(x2_list[0].childNodes[0].data))
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(float(y2_list[0].childNodes[0].data))
                    train_ann.write(
                        str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(label[object_name]) + ' ')
            train_ann.write('\n')

with open(voc12_trainval) as f:
    lines = f.readlines()
    for line in lines:
        img_name = line.strip()
        train_ann.write(img_name + '.jpg' + ' ')
        img_file = os.path.join(voc12_imgdir, img_name + '.jpg')
        annotation_file = voc12_annotations + '/' + img_name + '.xml'
        if os.path.isfile(annotation_file):
            shutil.copy(os.path.join(voc12_imgdir, img_name + '.jpg'),
                        os.path.join(train_dir, img_name + '.jpg'))
            dom_tree = xml.dom.minidom.parse(annotation_file)
            annotation = dom_tree.documentElement
            object_list = annotation.getElementsByTagName('object')
            for obj in object_list:
                name_list = obj.getElementsByTagName('name')
                object_name = name_list[0].childNodes[0].data
                bndbox = obj.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(float(x1_list[0].childNodes[0].data))
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(float(y1_list[0].childNodes[0].data))
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(float(x2_list[0].childNodes[0].data))
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(float(y2_list[0].childNodes[0].data))
                    train_ann.write(
                        str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(label[object_name]) + ' ')
            train_ann.write('\n')

train_ann.close()

with open(voc07_test) as f:
    lines = f.readlines()
    for line in lines:
        img_name = line.strip()
        test_ann.write(img_name + '.jpg' + ' ')
        img_file = os.path.join(voc07_imgdir, img_name + '.jpg')
        annotation_file = voc07_annotations + '/' + img_name + '.xml'
        if os.path.isfile(annotation_file):
            shutil.copy(os.path.join(voc07_imgdir, img_name + '.jpg'),
                        os.path.join(test_dir, img_name + '.jpg'))
            dom_tree = xml.dom.minidom.parse(annotation_file)
            annotation = dom_tree.documentElement
            object_list = annotation.getElementsByTagName('object')
            for obj in object_list:
                name_list = obj.getElementsByTagName('name')
                object_name = name_list[0].childNodes[0].data
                bndbox = obj.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(float(x1_list[0].childNodes[0].data))
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(float(y1_list[0].childNodes[0].data))
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(float(x2_list[0].childNodes[0].data))
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(float(y2_list[0].childNodes[0].data))
                    test_ann.write(
                        str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(label[object_name]) + ' ')
            test_ann.write('\n')

test_ann.close()
