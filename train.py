from os import pardir
import torch
from torch.nn.modules import loss
from options import TrainOptions
from dataset import dataset_unpair
from model import DCDA
from saver import Saver
from numpy import mean
import numpy as np
import segmentation_models_pytorch as smp
import cv2

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  pre_seg_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  pre_seg_model = smp.Unet(encoder_name="resnet34", encoder_depth=5, encoder_weights='imagenet', decoder_channels=[256, 128, 64, 32, 16], in_channels=1, classes=1, activation='sigmoid')
  pre_seg_optimizer = torch.optim.Adam([
      dict(params=pre_seg_model.parameters(), lr=1e-4)
  ])
  pre_seg_model.cuda(opts.gpu)
  pre_loss = smp.losses.DiceLoss(mode='binary', from_logits=False)

  model = DCDA(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')

  for ep in range(ep0, 100):
    for it, (images_a, images_b, gt, _) in enumerate(pre_seg_loader):
      images_a = images_a.cuda(opts.gpu)
      images_b = images_b.cuda(opts.gpu)
      gt = gt.cuda(opts.gpu)
      pre_seg_optimizer.zero_grad()
      outputs = pre_seg_model(images_a)
      losses = pre_loss(outputs, gt)
      losses.backward()
      pre_seg_optimizer.step()

  torch.autograd.set_detect_anomaly(True)
  max_it = 500000
  for ep in range(ep0, opts.n_ep):
    for it, (images_a, images_b, gt, _) in enumerate(train_loader):
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()
      # update model
      if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
        model.update_D_content(images_a[:, 0, :, :].unsqueeze(1), images_b[:, 0, :, :].unsqueeze(1))
        continue
      else:
        model.update_D(images_a[:, 0, :, :].unsqueeze(1), images_b[:, 0, :, :].unsqueeze(1))
        model.update_EG()
      if ep >= 100:
        model.update_dual_seg(images_a, images_b, gt, pre_seg_model, True)
      # save to display file
      if not opts.no_display_img:
        saver.write_display(total_it, model)

      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, model)
        break

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

  return

if __name__ == '__main__':
  main()
