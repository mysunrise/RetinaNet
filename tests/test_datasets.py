from context import *
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt


def test():
    train_imgdir = '/home/zhaoyang/data/voc0712/train'
    train_annotation_file = '/home/zhaoyang/data/voc0712/annotation/train_annotation.txt'
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    train_set = VocDataset(train_imgdir, train_annotation_file, input_size=[300, 300],
                           transform=transforms.Compose([transforms.ToTensor(), normalizer]))

    img, target_loc, target_cls = train_set[2]

    '''
    print(target_cls)
    print(target_loc)
    img = transforms.ToPILImage()(img).convert('RGB')
    img.show()
    '''
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                                               num_workers=8, collate_fn=train_set.collate_fn)

    for idx, (input, target_loc, target_cls) in enumerate(train_loader):
        if idx < 1:
            print("image size:", input.size())
            print("target loc size:", target_loc.size())
            print("target cls size:", target_cls.size())

            samples = target_cls > 0
            target_cls_masked = target_cls[samples].view(-1, 1)
            print(target_cls_masked)


if __name__ == '__main__':
    test()
