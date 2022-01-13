import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import cv2
import torch

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.test_phase1 + 'img'))
    self.img = [os.path.join(self.dataroot, opts.test_phase1 + 'img', x) for x in images]
    images2 = os.listdir(os.path.join(self.dataroot, opts.test_phase2 + 'img'))
    self.img2 = [os.path.join(self.dataroot, opts.test_phase2 + 'img', x) for x in images2]
    gts = os.listdir(os.path.join(self.dataroot, opts.test_phase1 + 'gt'))
    self.gt = [os.path.join(self.dataroot, opts.test_phase1 + 'gt', x) for x in gts]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data1 = self.load_img(self.img[index], self.input_dim, 1)
    data2 = self.load_img(self.img2[index], self.input_dim, 2)
    gt = cv2.imread(self.img[index].replace('img', 'gt'), 0)
    gt = np.where(gt == 255, 1, 0)
    return data1, data2, gt

  def load_img(self, img_name, input_dim, seq):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.direction = opts.direction
    self.dataroot = opts.dataroot
    self.seed = random.randint(0,6546498451351)
    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase1 + 'img'))
    self.A = [os.path.join(self.dataroot, opts.phase1 + 'img', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase2 + 'img'))
    self.B = [os.path.join(self.dataroot, opts.phase2 + 'img', x) for x in images_B]

    #gt
    gts_A = os.listdir(os.path.join(self.dataroot, opts.phase1 + 'gt'))
    self.gt = [os.path.join(self.dataroot, opts.phase1 + 'gt', x) for x in gts_A]

    gts_B = os.listdir(os.path.join(self.dataroot, opts.phase2 + 'gt'))
    self.gt2 = [os.path.join(self.dataroot, opts.phase2 + 'gt', x) for x in gts_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    gt_transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    self.gt_transforms = Compose(gt_transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    gt = cv2.imread(self.A[index].replace('img', 'gt'), 0)
    gt = np.where(gt == 255, 1, 0)
    gt = Image.fromarray(np.uint8(gt))
    gt = self.gt_transforms(gt)
    gt = np.array(gt)

    gt2 = cv2.imread(self.B[index].replace('img', 'gt'), 0)
    gt2 = np.where(gt2 == 255, 1, 0)
    gt2 = Image.fromarray(np.uint8(gt2))
    gt2 = self.gt_transforms(gt2)
    gt2 = np.array(gt2)
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A, 'A')
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B, 'B')
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A, 'A')
      data_B = self.load_img(self.B[index], self.input_dim_B, 'B')
    return data_A, data_B, gt, gt2

  def load_img(self, img_name, input_dim, domain='A'):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
