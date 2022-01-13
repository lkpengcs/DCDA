import torch
from options import TestOptions
from dataset import dataset_single, dataset_unpair
from model import DCDA
from saver import save_imgs
import os
import numpy as np
from numpy import mean, std
import pandas as pd
import scipy.spatial
import surface_distance


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getJaccard(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.jaccard(testArray, resultArray)


def getPrecisionAndRecall(testImage, resultImage):

    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    TP = np.sum(testArray*resultArray)
    FP = np.sum((1-testArray)*resultArray)
    FN = np.sum(testArray*(1-resultArray))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision, recall


def getHD_ASSD(seg_preds, seg_labels):
    label_seg = np.array(seg_labels, dtype=bool)
    predict = np.array(seg_preds, dtype=bool)

    surface_distances = surface_distance.compute_surface_distances(
        label_seg, predict, spacing_mm=(1, 1))

    HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

    return HD, ASSD


def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  if opts.a2b:
    dataset = dataset_single(opts, 'A', opts.input_dim_a)
  else:
    dataset = dataset_single(opts, 'B', opts.input_dim_b)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads, shuffle=False)

  # model
  print('\n--- load model ---')
  model = DCDA(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  dices = []
  pd_names = []
  jaccards = []
  hds = []
  assds = []
  for idx1, (img1, img2, gt1) in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    pd_names.append(str(idx1))
    img1 = img1.cuda(opts.gpu)
    img2 = img2.cuda(opts.gpu)
    gt1 = gt1.cuda(opts.gpu)
    imgs = [img1]
    names = ['input' + str(idx1)]
    for idx2 in range(opts.num):
      with torch.no_grad():
        img = model.test_forward(img2, a2b=opts.a2b)
        outputs = model.tar_seg_model(img1)
        preds = torch.round(outputs)
        preds = preds.squeeze().cpu().detach().numpy()
        gt1 = gt1.squeeze().cpu().detach().numpy()
        dice = getDSC(gt1, preds)
        jac = getJaccard(gt1, preds)
        hd, assd = getHD_ASSD(preds, gt1)
        dices.append(dice)
        jaccards.append(jac)
        hds.append(hd)
        assds.append(assd)
      imgs.append(img)
      names.append('output_{}'.format(idx1))
    #save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(idx1)))
    save_imgs(imgs, names, os.path.join(result_dir))
  
  dataframe = pd.DataFrame({'case': pd_names,
                            'rv_dice': dices,
                            'rv_jaccard': jaccards,
                            'rv_HD': hds, 'rv_ASSD': assds
                            })
  dataframe.to_csv(opts.result_dir + "/detail_metrics.csv",
                     index=False, sep=',')
  print('Counting CSV generated!')
  mean_resultframe = pd.DataFrame({
        'rv_dice': mean(dices), 'rv_jaccard': mean(jaccards),
        'rv_HD': mean(hds), 'rv_ASSD': mean(assds)}, index=[1])
  mean_resultframe.to_csv(opts.result_dir + "/mean_metrics.csv", index=0)
  std_resultframe = pd.DataFrame({
        'rv_dice': std(dices, ddof=1), 'rv_jaccard': std(jaccards, ddof=1),
        'rv_HD': std(hds, ddof=1), 'rv_ASSD': std(assds, ddof=1)}, index=[1])
  std_resultframe.to_csv(opts.result_dir + "/std_metrics.csv", index=0)
  print('Calculating CSV generated!')
  return

if __name__ == '__main__':
  main()
