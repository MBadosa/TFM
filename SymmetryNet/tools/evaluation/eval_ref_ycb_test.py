import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
sys.path.append('/home/marc.badosa.samso/Desktop/TFM/Codi/SymmetryNet') 
sys.path.append('/home/marc.badosa.samso/Desktop/TFM/Codi/SymmetryNet/datasets/')
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from lib.network import SymNet
from lib.tools import reflect
import sklearn.cluster as skc
import cv2
import copy
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb')
parser.add_argument('--dataset_root', type=str, default = '/home/marc.badosa.samso/Desktop/TFM/Codi/SymmetryNet/data/ycb_dataset')
parser.add_argument('--project_root', type=str, default = '/home/marc.badosa.samso/Desktop/TFM/Codi/SymmetryNet/')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--resume_symnet', type=str, default='ycb_model.pth', help='resume SymNet model')
#parser.add_argument('--resume_symnet', type=str, default='ycb_model.pth', help='resume SymNet model')

opt = parser.parse_args()
proj_dir = opt.project_root

def get_bbox(im):
    np_seg = np.array(im)
    segmentation = np.where(np_seg == 255)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        
    return x_min, x_max, y_min, y_max

def get_data(folder_path):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = cv2.cvtColor(np.array(Image.open("../inference_data/" + folder_path + "/rgb_image.png")),cv2.COLOR_BGRA2BGR)
    cloud = np.load("../inference_data/" + folder_path + "/pointcloud.npy")
    custom_mask = np.invert(np.array(Image.open("../inference_data/" + folder_path + "/rgbd_image.png")))

    custom_mask =  cv2.resize(custom_mask, (640,480))
    
    rmin, rmax, cmin, cmax = get_bbox(custom_mask)

    test = copy.deepcopy(cloud)
    while cloud.shape[0] < 1000:
        missing_values = 1000 - cloud.shape[0]
        test = test[:missing_values,:]
        cloud = np.concatenate((cloud,test),axis=0)

    img_masked = np.transpose(img[cmin:cmax, rmin:rmax])

    choose = custom_mask[cmin:cmax, rmin:rmax].flatten().nonzero()[0]
    num_pt = cloud.shape[0] 
    
    if len(choose) > num_pt:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_pt] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:    
        choose = np.pad(choose, (0, num_pt - len(choose)), 'wrap')

    choose = np.array([choose])     
    img_return = torch.unsqueeze(norm(torch.from_numpy(img_masked.astype(np.float32))), dim=0)
    cloud_return = torch.unsqueeze(torch.from_numpy(cloud.astype(np.float32)), dim=0)
    choose_return = torch.unsqueeze(torch.LongTensor(choose.astype(np.int32)), dim=0)

    return cloud_return, \
            choose_return, \
            img_return

if __name__ == '__main__':

    ### First of all read all folders inside inference_data folder:
    inference_objects_folders_name = [name for name in os.listdir("../inference_data") if os.path.isdir(os.path.join("../inference_data", name))]
    inference_objects_folders_name = sorted(inference_objects_folders_name)

    ### Create output directory if not exists
    if not os.path.exists("../symmetry_results"):
        os.makedirs("../symmetry_results")
    opt.outf = proj_dir + 'trained_models/ycb'


    estimator = SymNet(num_points=1000)

    estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_symnet), map_location=torch.device('cpu')))
    estimator.eval()
    for object_folder_name in inference_objects_folders_name:
        main_object_folder_name = "../symmetry_results/" + object_folder_name

        if not os.path.exists("../symmetry_results/" + object_folder_name):
            os.makedirs(main_object_folder_name)

        points, choose, img = get_data(object_folder_name)
        
        pred_cent, pred_ref, pred_foot_ref, _, _, _, _ = estimator(img, points, choose)

        points = points.view(1000, 3)
        points = points.data.numpy()

        pred_cent = pred_cent.data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        pred_cent = (points + pred_cent)

        pred_ref = pred_ref.data.numpy().reshape(1000, -1, 3)

        pred_foot_ref = pred_foot_ref.data.numpy().reshape(1000, -1, 3)
        foot_pred = pred_foot_ref + points.reshape(1000,1,3)

        my_sym = pred_ref
        my_norm = np.zeros(my_sym.shape)
        for i in range(my_sym.shape[1]):
            for k in range(3):
                my_norm[:, i, k] = my_sym[:, i, k] / np.linalg.norm(my_sym[:, i, :], axis=1)
        
        mean_norm = np.mean(my_norm, axis=0)

        mean_cent = np.mean(pred_cent, axis=0)
        out_cent = mean_cent

        ###### DBSCAN Symmetry plane
        out_sym = np.zeros(mean_norm.shape)
        for i in range(my_norm.shape[1]):
            this_norm = my_norm[:, i, :].reshape(1000, 3)
            for t in range(3):
                this_dim = this_norm[:, t].reshape(1000, 1)
                db = skc.DBSCAN(eps=0.2, min_samples=500).fit(this_dim)
                labels = db.labels_
                clster_center = np.mean(this_dim[labels[:] == 0], axis=0)
                out_sym[i,t] = clster_center

        ######## REFLECT POINTCLOUD
        my_ref = reflect(points, out_cent, out_sym)
        
        ######## STORE THE OUTPUT
        np.save(main_object_folder_name + "/final.npy",my_ref)
        np.save(main_object_folder_name + "/center.npy", out_cent)
        np.save(main_object_folder_name + "/original.npy", points)        
        np.save(main_object_folder_name + "/foot.npy", foot_pred)
        np.save(main_object_folder_name + "/symmetri.npy", out_sym)