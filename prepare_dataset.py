import numpy as np
import os
import argparse
from utils import read_train_data, read_test_data

parser = argparse.ArgumentParser(description='Data preperation')
parser.add_argument('--train_data_path', help='path to training data folder', default='train_data', type=str)
parser.add_argument('--test_data_path', help='path to test data folder', default='test_data', type=str)
parser.add_argument('--save_path', help='save path for training and test numpy matrices of images', default='.', type=str)
args = parser.parse_args()

#read training data
train_imgs, train_gts = read_train_data(args.train_data_path)

#remove dublicate training imgs
idx_to_rmv = []
for i in range(len(train_imgs)-1):
    for j in range(i+1, len(train_imgs)):
        if np.all(train_imgs[i] == train_imgs[j]):
            idx_to_rmv.append(i)
            if train_gts[i] != train_gts[j]:
                idx_to_rmv.append(j)

idx = [i for i in range(len(train_imgs)) if not(i in idx_to_rmv)]
print('unique train imgs:',len(idx))

#save unique training imgs
np.save(os.path.join(args.save_path, 'unique_train_imgs_rot_fixed'), np.array(train_imgs)[idx])
np.save(os.path.join(args.save_path, 'unique_train_gts_rot_fixed'), np.array(train_gts)[idx])

#read test data
test_imgs, test_gts, ids = read_test_data(args.test_data_path)

#save test data
np.save(os.path.join(args.save_path, 'test_imgs_rot_fixed'), np.array(test_imgs))
np.save(os.path.join(args.save_path, 'test_gts'), np.array(test_gts))
np.save(os.path.join(args.save_path, 'ids'), np.array(ids))

