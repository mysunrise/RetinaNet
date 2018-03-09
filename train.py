from __future__ import print_function
from __future__ import division

import time
import os
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from datasets import VocDataset
from models.retinanet import retinanet101
from focalloss import FocalLoss

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--num_classes', default=21, type=int, metavar='N',
                    help='number of total class')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--alpha', '--alpha', default=1.0, type=float,
                    metavar='Alpha', help='alpha')
parser.add_argument('--gamma', '--gamma', default=2.0, type=float,
                    metavar='Gamma', help='gamma')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='print_freq', help='print frequency (default: 10)')
parser.add_argument('--save_freq', '-s', default=10, type=int,
                    metavar='save_freq', help='save frequency (default: 10)')
parser.add_argument('--test_freq', '-t', default=5, type=int,
                    metavar='test_freq', help='test frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='.', type=str, metavar='PATH',
                    help='path to save the models')

best_loss = float('inf')


def main():
    torch.backends.cudnn.benchmark = True
    global best_loss, args, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    train_imgdir = '/home/zhaoyang/data/voc0712/train'
    test_imgdir = '/home/zhaoyang/data/voc0712/test'
    train_annotation_file = '/home/zhaoyang/data/voc0712/annotation/train_annotation.txt'
    test_annotaiion_file = '/home/zhaoyang/data/voc0712/annotation/test_annotation.txt'
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    train_set = VocDataset(train_imgdir, train_annotation_file, input_size=[600, 600],
                           transform=transforms.Compose([transforms.ToTensor(), normalizer]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True,
                                               num_workers=8, collate_fn=train_set.collate_fn)
    test_set = VocDataset(test_imgdir, test_annotaiion_file, input_size=[600, 600],
                          transform=transforms.Compose([transforms.ToTensor(), normalizer]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True,
                                              num_workers=8, collate_fn=test_set.collate_fn)

    model = retinanet101(pretrained=True, num_classes=args.num_classes)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    #criterion = FocalLoss(args.alpha * torch.ones(args.num_classes, 1), args.gamma)
    criterion = FocalLoss()

    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading model from {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model = model.load_state_dict(checkpoint['model'])
            optimizer = optimizer.load_state_dict((checkpoint['optimizer']))
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print('Loaded model from {} (epoch {})'.format(args.resume, args.start_epoch))
        else:
            print('No checkpoint founded in {}'.format(args.resume))

    if use_gpu:
        model.cuda()
        criterion.cuda()

    for epoch in xrange(args.start_epoch, args.epochs):
        #test_loss = test_model(model, test_loader, criterion)
        #print("test loss", test_loss)
        lr_scheduler.step()
        train_model(model, train_loader, optimizer, criterion, epoch)
        if (epoch + 1) % args.test_freq == 0:
            test_loss = test_model(model, test_loader, criterion)
            if test_loss <= best_loss:
                best_loss = test_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, epoch + 1, True)
                print('best test loss: {}'.format(best_loss))
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss
            }, epoch + 1, False)


def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    batch_time = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()
    losses = AverageMeter()
    since = time.time()

    for idx, (input, target_loc, target_cls) in enumerate(train_loader):
        if use_gpu:
            input, target_loc, target_cls = input.cuda(), target_loc.cuda(), target_cls.cuda()
        input, target_loc, target_cls = Variable(input), Variable(target_loc), Variable(target_cls)
        pred_loc, pred_cls = model(input)  # (n, anchor_num, classes_num), (n, anchor_num, 4)
        pos_samples = target_cls > 0

        pos_samples_num = pos_samples.data.long().sum()
        if pos_samples_num == 0:
            continue
        #print("pos sample:", pos_samples_num)
        mask = pos_samples.unsqueeze(2).expand_as(pred_loc)
        pred_loc_masked = pred_loc[mask].view(-1, 4)
        target_loc_masked = target_loc[mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pred_loc_masked, target_loc_masked, size_average=False) / pos_samples_num

        samples = target_cls > -1
        #print("sample num:", samples.data.long().sum())
        mask = samples.unsqueeze(2).expand_as(pred_cls)
        pred_cls_masked = pred_cls[mask].view(-1, args.num_classes)
        target_cls_masked = target_cls[samples].view(-1, 1)
        #print("sample size:", samples.size())
        #print("size:", pred_cls_masked.size(), target_cls_masked.size())
        #print("target cls:", target_cls)
        #print("pred clc masked:", pred_cls_masked)
        #print("target clc masked:", target_cls_masked)
        focalloss = criterion(pred_cls_masked, target_cls_masked)
        #print("focal loss:", focalloss)

        cls_loss = focalloss / pos_samples_num
        loss = loc_loss + cls_loss
        cls_losses.update(cls_loss.data[0], pos_samples_num)
        loc_losses.update(loc_loss.data[0], pos_samples_num)
        losses.update(loss.data[0], pos_samples_num)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - since)
        since = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\t'
                  'Time: {time} {batch_time.val: .3f} {batch_time.avg: .3f}\t'
                  'focolloss: {cls_loss.val: .4f} {cls_loss.avg: .4f}\t'
                  'smooth_l1_loss: {loc_loss.val: .4f} {loc_loss.avg: .4f}\t'
                  'Loss: {loss.val: .4f} {loss.avg: .4f}\t'.format(
                    epoch, idx, len(train_loader), time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    batch_time=batch_time, cls_loss=cls_losses, loc_loss=loc_losses, loss=losses))


def test_model(model, test_loader, criterion):
    model.eval()
    batch_time = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()
    losses = AverageMeter()
    since = time.time()

    for idx, (input, target_loc, target_cls) in enumerate(test_loader):
        if use_gpu:
            input, target_loc, target_cls = input.cuda(), target_loc.cuda(), target_cls.cuda()
        input, target_loc, target_cls = Variable(input), Variable(target_loc), Variable(target_cls)
        pred_loc, pred_cls = model(input)  # (n, anchor_num, classes_num), (n, anchor_num, 4)
        pos_samples = target_cls > 0
        pos_samples_num = pos_samples.data.long().sum()
        mask = pos_samples.unsqueeze(2).expand_as(pred_loc)
        pred_loc_masked = pred_loc[mask].view(-1, 4)
        target_loc_masked = target_loc[mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pred_loc_masked, target_loc_masked, size_average=False) / pos_samples_num

        samples = target_cls > -1
        mask = samples.unsqueeze(2).expand_as(pred_cls)
        pred_cls_masked = pred_cls[mask].view(-1, args.num_classes)
        target_cls_masked = target_cls[samples].view(-1, 1)
        #print("pred clc masked:", pred_cls_masked)
        #print("target clc masked:", target_cls_masked)
        cls_loss = criterion(pred_cls_masked, target_cls_masked) / pos_samples_num
        loss = loc_loss + cls_loss
        cls_losses.update(cls_loss.data[0], pos_samples_num)
        loc_losses.update(loc_loss.data[0], pos_samples_num)
        losses.update(loss.data[0], pos_samples_num)
        batch_time.update(time.time() - since)
        since = time.time()

        if idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time: {time} {batch_time.val: .3f} {batch_time.avg: .3f}\t'
                  'focolloss: {cls_loss.val: .4f} {cls_loss.avg: .4f}\t'
                  'smooth_l1_loss: {loc_loss.val: .4f} {loc_loss.avg: .4f}\t'
                  'Loss: {loss.val: .4f} {loss.avg: .4f}\t'.format(
                    idx, len(test_loader), time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    batch_time=batch_time, cls_loss=cls_losses, loc_loss=loc_losses, loss=losses))

    return losses.avg


def save_checkpoint(state, epoch, is_best):
    if is_best:
        torch.save(state, os.path.join(args.save_dir, 'model_best.pth'))
        return
    model_save_path = os.path.join(args.save_dir, 'model_epoch_{}'.format(epoch))
    torch.save(state, model_save_path)
    print('checkpoint saved to {}'.format(model_save_path))


class AverageMeter(object):
    """docstring for AverageMeter"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, value, n=1):
        self.val = value
        self.sum = self.sum + self.val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
