import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_my import RetinaFace
import time
from skimage import transform as trans


def iou(box1,box2,thresh):
    if box2.shape[0] ==0:
        return None
    x1,y1,w1,h1 = box1[0],box1[1],box1[2]-box1[0],box1[3]-box1[1]
    x2,y2,w2,h2 = box2[:,0],box2[:,1],box2[:,2]-box2[:,0],box2[:,3]-box2[:,1]
    area1 = w1*h1
    area2 = w2*h2
    sum_area = area2+area1
    left = np.maximum(x1,x2)
    right = np.minimum(x1+w1,x2+w2)
    top = np.maximum(y1,y2)
    bottom = np.minimum(y1+h1,y2+h2)
    w = np.maximum(0.0, right-left+1)
    h = np.maximum(0.0, bottom-top+1)
    inter = w*h
    ovr = inter / (sum_area-inter)
    max_idx = np.argmax(ovr)
    if(ovr[max_idx]>thresh):
        return max_idx
    return None

if __name__ == '__main__':

    thresh = 0.8
    scales = [1024, 1980]

    gpuid = 2
    detector = RetinaFace('./model/mnet25', 0, gpuid, 'net3')

    img_root_dir = '..'
    img_list_path = '../success_file.txt'
    output_dir = '../output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(img_list_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(',') for line in lines]
    total_lines = len(lines)

    # total_time = 0.0

    if total_lines % 100 == 0:
        per_percent = total_lines // 100
    else:
        per_percent = total_lines // 100 + 1
    for idx in range(total_lines):

        stdstr = '\r%s%% |' % (idx // per_percent + 1) + '>' * (idx // (4 * per_percent)) + ' ' * (
            (100 // 4 - idx // (4 * per_percent))) + '| %s/%s' % (idx, total_lines)
        sys.stdout.write(stdstr)
        sys.stdout.flush()

        img_path,gt_boxes,old_hw = lines[idx]

        img_path_split = img_path.split('/')
        img_save_dir = os.path.join(output_dir, img_path_split[1])
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)
        img_save_path = os.path.join(img_save_dir,img_path_split[2])
        if os.path.exists(img_save_path):
            continue

        old_hw = old_hw.split(' ')
        gt_boxes = gt_boxes.split(' ')
        gtboxes = np.array([float(gt_boxes[0]),float(gt_boxes[1]),float(gt_boxes[2]),float(gt_boxes[3])])
        img = cv2.imread(os.path.join(img_root_dir,img_path))
        if int(old_hw[0])!=img.shape[0] or int(old_hw[1])!=img.shape[1]:
            sc_h = float(img.shape[0]) / float(old_hw[0])
            sc_w = float(img.shape[1]) / float(old_hw[1])
            gtboxes[0] = gtboxes[0] * sc_w
            gtboxes[1] = gtboxes[1] * sc_h
            gtboxes[2] = gtboxes[2] * sc_w
            gtboxes[3] = gtboxes[3] * sc_h

        # time_start = time.time()

        faces, landmarks = detector.detect(img, thresh)

        # time_end = time.time()
        # # print('total lines', total_lines, 'now', idx, 'time cost', time_end - time_start)
        # use_time = time_end - time_start
        # total_time += use_time

        keep = iou(gtboxes,faces,0.3)
        if keep is not None:
            lands = landmarks[keep]
        else:
            continue

        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(lands, src)
        M = tform.params[0:2, :]
        warped0 = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

        cv2.imwrite(img_save_path, warped0)


    # print('total time cost', total_time)

