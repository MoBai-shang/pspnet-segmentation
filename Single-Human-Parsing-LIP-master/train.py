#!/usr/local/bin/python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.summaries import TensorboardSummary
from dataset.lip import LIPWithClass
from net.pspnet import PSPNet

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

parser = argparse.ArgumentParser(description="Human Parsing")
parser.add_argument('--data-path', type=str, default='F:\code\pycharm\Datasets\LIP',help='Path to dataset folder')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--snapshot', type=str, default=None, help='Path to pre-trained weights')
parser.add_argument('--batch-size', type=int, default=16, help="Number of images sent to the network in one step.")
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs to run')
parser.add_argument('--crop_x', type=int, default=128, help='Horizontal random crop size')
parser.add_argument('--crop_y', type=int, default=128, help='Vertical random crop size')
parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
parser.add_argument('--start-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
parser.add_argument('--eval_step', type=int, default=1, help='per eval_step to eval the model')
args = parser.parse_args()
colormap = np.array([(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ])*128

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


if __name__ == '__main__':
    models_path = os.path.join('./checkpoints', args.backend)
    os.makedirs(models_path, exist_ok=True)
    transform_image_list = [
        transforms.CenterCrop((args.crop_x,args.crop_y)),#transforms.Resize((args.crop_x,args.crop_y), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.CenterCrop((args.crop_x,args.crop_y)),#transforms.Resize((args.crop_x,args.crop_y), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]

    train_loader = DataLoader(LIPWithClass(root=args.data_path, dtype='train',transform=transforms.Compose(transform_image_list),tg_transform=transforms.Compose(transform_gt_list)),batch_size=args.batch_size,shuffle=True,)
    eval_loader = DataLoader(LIPWithClass(root=args.data_path, dtype='val',transform=transforms.Compose(transform_image_list),tg_transform=transforms.Compose(transform_gt_list)),batch_size=args.batch_size,shuffle=False,)

    net, starting_epoch = build_network(args.snapshot, args.backend)
    optimizer = optim.Adam(net.parameters(), lr=args.start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in args.milestones.split(',')])

    net.train()
    seg_criterion = nn.NLLLoss(weight=None)
    cls_criterion = nn.BCEWithLogitsLoss(weight=None)
    best_loss=200
    epochs_losses = []
    pen=TensorboardSummary('run')
    eval_vis_init_state=True
    eval_show_step=0
    for epoch in range(1+starting_epoch, 1+starting_epoch+args.epochs):
        epoch_losses = []
        tbar=tqdm(train_loader)
        for count, (x, y, y_cls) in enumerate(tbar):
            # input data
            x, y, y_cls = x.cuda(), y.cuda().long(), y_cls.cuda().float()
            # forward
            out, out_cls = net(x)

            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + args.alpha * cls_loss
            #pen.add_scalar(tag='train/seg loss', scalar_value=seg_loss.item(), global_step=global_step,display_name='seg-loss', summary_description='loss at seg')
            #pen.add_scalar(tag='train/cls loss', scalar_value=cls_loss.item(), global_step=global_step,display_name='cls-loss', summary_description='loss at class')
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print
            epoch_losses.append(loss.item())
            status = 'Train-->loss = %.4f avg = %.4f, LR = %.6f'%(loss.item(), np.mean(epoch_losses), scheduler.get_last_lr()[0])
            #print(status)
            tbar.set_description(status)
            #pen.add_scalar(tag='train/total loss iter',scalar_value=loss.item(),global_step=global_step,display_name='total iter',summary_description='summary at train')
            #global_step+=1

        scheduler.step()
        epochs_losses.append(np.mean(epoch_losses))
        pen.add_scalar(tag='train/epoch loss', scalar_value=epochs_losses[-1], global_step=epoch,
                       display_name='epoch iter', summary_description='epoch at train')

        if epoch%args.eval_step==0:
            eval_vis_init_state=not eval_vis_init_state
            eval_vis=eval_vis_init_state
            net.eval()
            epoch_losses = []
            tbar = tqdm(eval_loader)
            for count, (x, y, y_cls) in enumerate(tbar):
                # input data
                x, y, y_cls = x.cuda(), y.cuda().long(), y_cls.cuda().float()
                # forward
                out, out_cls = net(x)
                seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
                loss = seg_loss + args.alpha * cls_loss
                #pen.add_scalar(tag='eval/seg loss', scalar_value=seg_loss.item(), display_name='seg-loss', summary_description='loss at seg')
                #pen.add_scalar(tag='eval/cls loss', scalar_value=cls_loss.item(), display_name='cls-loss', summary_description='loss at class')

                epoch_losses.append(loss.item())
                status = 'Eval-->loss = {0:0.4f} avg = {1:0.4f}'.format(loss.item(), np.mean(epoch_losses))
                # print(status)
                tbar.set_description(status)
                if eval_vis:
                    pen.add_image_by_src_pre_tar(x, out, y, n_classes=20, label_colours=colormap,
                                                 tag='source-predict-target',global_step=eval_show_step)
                    eval_show_step+=1
                eval_vis=not eval_vis
                #pen.add_scalar(tag='eval/total loss iter', scalar_value=loss.item(), display_name='total iter', summary_description='summary at train')
            pen.add_scalar(tag='eval/epoch loss', scalar_value=np.mean(epoch_losses),global_step=epoch, display_name='epoch iter', summary_description='epoch at train')

            net.train()
        if epochs_losses[-1]<best_loss:
            best_loss=epochs_losses[-1]
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch)])))
    pen.writer.close()
