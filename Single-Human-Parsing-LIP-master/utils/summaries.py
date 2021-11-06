
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torch
import numpy as np
from os import path
import glob
class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.runs = sorted(glob.glob(path.join(self.directory, 'experiment_*')))
        self.run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        self.writer = SummaryWriter(log_dir=path.join(self.directory,'experiment_%d'%self.run_id))
        print('log dir at experiment_%d'%self.run_id)
    def add_image(self,tensor,tag,global_step= None,walltime = None,dataformats='CHW',nrow=4,padding= 2,normalize= False,value_range= None,scale_each= False,pad_value= 0,**kwargs):
        grid_image = make_grid(tensor=tensor,nrow=nrow,padding=padding,normalize=normalize,value_range=value_range,scale_each=scale_each,pad_value=pad_value,kwargs=kwargs)
        self.writer.add_image(tag=tag,img_tensor=grid_image,global_step=global_step,walltime=walltime,dataformats=dataformats)

    def add_scalar(self,tag,scalar_value,global_step= None,walltime= None,display_name= "",summary_description= ""):
        self.writer.add_scalar(tag=tag,scalar_value=scalar_value,global_step=global_step,walltime=walltime,display_name=display_name,summary_description=summary_description)

    def add_image_by_src_pre_tar(self,image, output, target,n_classes,label_colours,tag,display_num=12,global_step= None,walltime = None,dataformats='CHW',nrow=None,padding= 2,normalize= False,value_range= None,scale_each= False,pad_value= 0,**kwargs):

        display_num = min(display_num, image.shape[0])
        if nrow is None:
            nrow = int(torch.sqrt(torch.tensor(display_num * 3)))
        src = image[:display_num].clone().cpu().data

        predict = decode_seg_map_sequence(torch.max(output[:display_num], 1)[1].detach().cpu().numpy(),n_classes,label_colours)
        true = decode_seg_map_sequence(torch.squeeze(target[:display_num], 1).detach().cpu().numpy(),n_classes,label_colours)

        norm_ip(src)

        combine_imgs = torch.stack([src, predict, true], -3)  # or dim=1
        shape = list(src.shape)
        shape[-2] = -1  # or shape[0]=-1
        combine_imgs = combine_imgs.reshape(*shape)
        '''import matplotlib.pyplot as plt
        print(true.shape)
        plt.imshow(true[0].transpose(0,2))
        plt.show()
        plt.imshow(predict[0].transpose(0,2))
        plt.show()
        plt.imshow(combine_imgs[0].transpose(0,2))
        plt.show()
        exit()'''
        grid_image = make_grid(tensor=combine_imgs,nrow=nrow,padding=padding,normalize=normalize,value_range=value_range,scale_each=scale_each,pad_value=pad_value,kwargs=kwargs)
        self.writer.add_image(tag=tag,img_tensor=grid_image,global_step=global_step,walltime=walltime,dataformats=dataformats)

def norm_ip(img):
    low=float(img.min())
    high=float(img.max())
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img
def decode_seg_map_sequence(label_masks,n_classes,label_colours):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask,n_classes,label_colours)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask,n_classes,label_colours):

    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """


    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r/ 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def encode_segmap(mask,label_colours):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(label_colours):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask
