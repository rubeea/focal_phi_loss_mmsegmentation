import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def perf_measure(preds, labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # print(preds.shape)
    # print(labels.shape)

    rows= labels.shape[0]
    cols=labels.shape[1]
    total_pixels= rows*cols

    for i in range(0, rows):
      for j in range(0, cols):
        if preds[i,j]==1 and labels[i,j]==1:
           TP += 1
        elif preds[i,j]==1 and labels[i,j]==0:
           FP += 1
        elif preds[i,j]==0 and labels[i,j]==0:
           TN += 1
        elif preds[i,j]==0 and labels[i,j]==1:
           FN += 1

#     TP= TP/total_pixels
#     FP= FP/total_pixels
#     TN= TN/total_pixels
#     FN= FN/total_pixels

    return (TP, FP, TN, FN)

def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    TP_list=[]
    FP_list=[]
    TN_list=[]
    FN_list=[]
   
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

        TP,FP,TN,FN= perf_measure(results[i],gt_seg_maps[i])
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    dice= total_area_intersect*2.00/(total_area_pred_label+total_area_label)
    
    # print(type(results))
    # print(type(gt_seg_maps))
    
    
    
    if nan_to_num is not None:
        return all_acc, sum(TP_list)/num_imgs, sum(FP_list)/num_imgs, sum(TN_list)/num_imgs, sum(FN_list)/num_imgs, np.nan_to_num(acc, nan=nan_to_num), \
            np.nan_to_num(iou, nan=nan_to_num), np.nan_to_num(dice, nan=nan_to_num)
    return all_acc, sum(TP_list)/num_imgs, sum(FP_list)/num_imgs, sum(TN_list)/num_imgs, sum(FN_list)/num_imgs, acc, iou, dice
