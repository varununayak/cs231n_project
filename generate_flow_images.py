from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import sys
import glob
sys.path.append('./pyflow/')
import pyflow

IMG_SIZE = 128
SEQUENCES = ['01'] # for local machine


for sequence in SEQUENCES:
    filelist = glob.glob('../dataset/sequences/{}/image_2/*.png'.format(sequence))

    for i in range(len(filelist) - 1):
        im1 = np.array(Image.open(filelist[i]).resize((IMG_SIZE,IMG_SIZE)))
        im2 = np.array(Image.open(filelist[i+1]).resize((IMG_SIZE,IMG_SIZE)))
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0

        u, v, _ = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        imagename = str(i)
        imagename = imagename.zfill(6)
        np.save('../flow_dataset/{}/{}.npy'.format(sequence, imagename), flow)
        print("Images Processed: ", i + 1)

