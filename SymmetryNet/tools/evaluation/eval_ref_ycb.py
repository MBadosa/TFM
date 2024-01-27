import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
sys.path.append('/home/marc.badosa.samso/SymmetryNet') # to run on a server
sys.path.append('/home/marc.badosa.samso/SymmetryNet/datasets/')
import numpy as np
import random
import time
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset_ycb_eval import SymDataset as SymDataset_ycb
from lib.network import SymNet
from lib.tools import reflect
import sklearn.cluster as skc  

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb')
parser.add_argument('--dataset_root', type=str, default = '/home/marc.badosa.samso/SymmetryNet/data/ycb_dataset')
parser.add_argument('--project_root', type=str, default = '/home/marc.badosa.samso/SymmetryNet/')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--resume_symnet', type=str, default='ycb_model.pth', help='resume SymNet model')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
proj_dir = opt.project_root
sym_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]

def output_symmetry():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_objects = 21  # number of object classes in the dataset
    opt.num_points = 1000  # number of points on the input pointcloud
    opt.outf = proj_dir + 'trained_models/ycb'  # folder to save trained models
    opt.repeat_epoch = 1  # number of repeat times for one epoch training

    estimator = SymNet(num_points=opt.num_points)
    estimator.cuda()

    estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_symnet)))
    opt.refine_start = False
    opt.decay_start = False

    test_dataset = SymDataset_ycb('test', opt.num_points, False, opt.dataset_root,proj_dir, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    estimator.eval()
    total_fream = 0
    total_num = 0
    pred_c = 1
    ref_list = [1, 3, 4, 5, 6, 8, 10, 11, 13, 15, 18, 20]
    # rot_list = [0, 2, 7, 19]
    for j, data in enumerate(testdataloader, 0):
        points, choose, img, idx, target_s, target_num, target_mode,depth,cam_intri,pt_num = data
        
        np.save('test.npy', points)
        if idx not in ref_list:
            continue
        # if target_mode != 1:
        total_fream += 1
        tmp_s = target_s.data.cpu().numpy()
        tmp_s = tmp_s.reshape(-1, 3)
        target_cent = tmp_s[0, :]
        target_sym = tmp_s[1:, :]  # target symmetry point
        np.save('copy/1.npy',points)
        np.save('copy/2.npy',img)

        points, choose, img, idx, target_s, target_num, target_mode = Variable(points).cuda(), \
                                                                      Variable(choose).cuda(), \
                                                                      Variable(img).cuda(), \
                                                                      Variable(idx).cuda(), \
                                                                      Variable(target_s).cuda(), \
                                                                      Variable(target_num).cuda(), \
                                                                      Variable(target_mode).cuda()

        pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode, emb = estimator(img, points, choose)

        points = points.view(1000, 3)
        points = points.detach().cpu().data.numpy()

        pred_cent = pred_cent.detach().cpu().data.numpy()
        pred_cent = pred_cent.reshape(1000, 3)
        pred_cent = (points + pred_cent)

        pred_num = pred_num.view(1000, 3).detach()
        my_num = torch.mean(pred_num, dim=0)
        my_num = my_num.data.cpu().numpy().reshape(3)

        pred_ref = pred_ref.detach().cpu().data.numpy().reshape(1000, -1, 3)
        ref_pred = pred_ref+points.reshape(1000,1,3)

        pred_foot_ref = pred_foot_ref.detach().cpu().data.numpy().reshape(1000, -1, 3)
        foot_pred = pred_foot_ref + points.reshape(1000,1,3)
        np.save('foot.npy',foot_pred)
        mode_pred = torch.mean(pred_mode.view(1000,3),dim=0).detach().cpu().data.numpy()

        my_cent = np.mean(pred_cent, axis=0)

        target_sym = target_sym - target_cent

        my_sym = pred_ref
        my_norm = np.zeros(my_sym.shape)
        for i in range(my_sym.shape[1]):
            for k in range(3):
                my_norm[:, i, k] = my_sym[:, i, k] / np.linalg.norm(my_sym[:, i, :], axis=1)


        mean_norm = np.mean(my_norm, axis=0)    # n*3
        mean_cent = np.mean(pred_cent, axis=0)  # 1*3
        out_cent = mean_cent

        ######DBSCAN
        out_sym = np.zeros(mean_norm.shape)
        sym_conf = np.zeros(mean_norm.shape[0])
        for i in range(my_norm.shape[1]):
            this_norm = my_norm[:, i, :].reshape(1000, 3)
            dim_conf = 0
            for t in range(3):
                this_dim = this_norm[:, t].reshape(1000, 1)
                # target_dim = target_sym[i,j]
                mean_dim = np.mean(this_dim, axis=0)
                db = skc.DBSCAN(eps=0.2, min_samples=500).fit(this_dim)
                labels = db.labels_
                clster_center = np.mean(this_dim[labels[:] == 0], axis=0)
                out_sym[i,t] = clster_center
                dim_conf += len(labels[labels[:] == 0]) / len(labels)
            norm_conf = dim_conf/3
            mode_conf = max(mode_pred[:2])
            sym_conf[i] = my_num[i]* norm_conf

        ########  verification

        my_ref = reflect(points, out_cent, out_sym)
        target_ref = reflect(points, target_cent, target_sym)
        print(pred_rot)
        np.save('copy/final_gt.npy',target_ref)
        np.save("copy/center.npy", out_cent)
        np.save("copy/final.npy",my_ref)
        target_sym = target_sym.reshape(-1, 3)
        target_vector = target_ref - points.reshape(1000,1,3).repeat(target_ref.shape[1], axis=1)





if __name__ == '__main__':
    st_time = time.time()
    savedir = proj_dir + 'tools/'

    output_symmetry()
    
