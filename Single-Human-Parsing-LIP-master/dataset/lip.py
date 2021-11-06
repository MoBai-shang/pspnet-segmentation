from os import path as osp
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np


class LIP(data.Dataset):

    def __init__(self, root,dtype,  transform=None, tg_transform=None):
        self.root = root
        self.transform = transform
        self.tg_transform = tg_transform
        self.dtype = dtype  # trainval set or val set

        if self.dtype in ['train','tra','Tra']:
            self.image_path, self.tg_path = self.read_labeled_image_list(osp.join(root, 'TrainVal_images'),tg_dir=osp.join(root, 'TrainVal_parsing_annotations'),dtype=self.dtype)
        elif self.dtype in ['validation','val','Val']:
            self.image_path, self.tg_path = self.read_labeled_image_list(osp.join(root, 'TrainVal_images'),tg_dir=osp.join(root, 'TrainVal_parsing_annotations'),dtype=self.dtype)
        elif self.dtype in ['test','tes','Tes']:
            self.image_path, self.tg_path = self.read_labeled_image_list(osp.join(root, 'Testing_images'),dtype=self.dtype)

    def __getitem__(self, index):
        # get PIL Image
        img = Image.open(self.image_path[index])
        if len(img.getbands()) != 3:
            img = img.convert('RGB')

        tg = Image.open(self.tg_path[index])
        if len(tg.getbands()) != 1:
            tg = tg.convert('L')

        if self.transform is not None:
            img = self.transform(img)
        if self.tg_transform is not None:
            tg = self.tg_transform(tg)
        return img, tg

    def __len__(self):
        return len(self.image_path)

    def read_labeled_image_list(self,image_dir,tg_dir=None, dtype='',img_format='.jpg',tg_format='.png'):
        # return img path list and groundtruth path list
        with open(osp.join(image_dir,dtype+ '_id.txt' ), 'r') as f:
            ids=f.read().splitlines()
            image_path=[ osp.join(image_dir,dtype+'_images',img+img_format) for img in ids]

        if dtype in ['test','tes','Tes']:
            return image_path,[]
        if tg_dir is None:
            tg_dir = image_dir
        tg_path=[ osp.join(tg_dir,dtype+'_segmentations',img+tg_format) for img in ids]

        return image_path, tg_path


class LIPWithClass(LIP):

    def __init__(self, root, dtype, num_cls=20, transform=None, tg_transform=None):
        LIP.__init__(self, root, dtype, transform, tg_transform)
        self.num_cls = num_cls

    def __getitem__(self, index):
        img, tg=LIP.__getitem__(self,index=index)
        # compute tg_cls
        tg_np = np.asarray(tg, dtype=np.uint8)
        tg_cls, _ = np.histogram(tg_np, bins=self.num_cls, range=(-0.5, self.num_cls - 0.5), )
        tg_cls = np.asarray(np.asarray(tg_cls, dtype=np.bool), dtype=np.uint8)

        return img, tg, tg_cls

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    path = 'K:\Dataset\LIP\single'

    transform_image_list = [
        transforms.Resize((512, 512), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_tg_list = [
        transforms.Resize((30, 30), Image.NEAREST),
        transforms.Lambda(lambda image: Image.fromarray(np.uint8(np.asarray(image)*(255.0/19.0)))),
        transforms.ToTensor(),
    ]

    data_transforms = {
        'image': transforms.Compose(transform_image_list),
        'tg': transforms.Compose(transform_tg_list),
    }

    loader = data.DataLoader(LIP(path, transform=data_transforms['image'], tg_transform=data_transforms['tg']),
                             batch_size=2, shuffle=False)

    for count, (src, lab) in enumerate(loader):
        src = src[0, :, :, :].numpy()
        lab = lab[0, :, :, :].numpy().transpose(1, 2, 0)


        def denormalize(image, mean, std):
            c, _, _ = image.shape
            for idx in range(c):
                image[idx, :, :] = image[idx, :, :] * std[idx] + mean[idx]
            return image

        src = denormalize(src, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).transpose(1, 2, 0)

        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(np.concatenate([lab, lab, lab], axis=2), cmap='gray')
        plt.show()
        print(src.shape)
        if count+1 == 4:
            break

