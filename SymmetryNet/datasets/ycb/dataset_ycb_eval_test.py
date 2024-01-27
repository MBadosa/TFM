import torch.utils.data as data
from PIL import Image
import torch
import copy
import cv2
import numpy as np
import torchvision.transforms as transforms

class SymDatasetTest(data.Dataset):
    def __init__(self, proj_dir):
        self.projdir = proj_dir
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        index = index +1
        img = cv2.cvtColor(np.array(Image.open('test_YCB/{}/{}.png'.format(index, index))),cv2.COLOR_BGRA2BGR)
        cloud = np.load("test_YCB/{}/depthCam_new.npy".format(index))
        custom_mask = np.invert(np.array(Image.open('test_YCB/{}/{}_mask.png'.format(index,index))))

        custom_mask =  cv2.resize(custom_mask, (640,480))
        
        rmin, rmax, cmin, cmax = get_bbox(custom_mask)

        test = copy.deepcopy(cloud)

        while cloud.shape[0] < 1000:
            missing_values = 1000 - cloud.shape[0]
            test = test[:missing_values,:]
            cloud = np.concatenate((cloud,test),axis=0)

        
        img_masked = np.transpose(img[cmin:cmax, rmin:rmax])

        choose = custom_mask[cmin:cmax, rmin:rmax].flatten().nonzero()[0]
        self.num_pt = cloud.shape[0] 
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:    
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        choose = np.array([choose])     
        
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32)))
    
    def __len__(self):
        return 30

def get_bbox(im):
    np_seg = np.array(im)
    segmentation = np.where(np_seg == 255)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        
    return x_min, x_max, y_min, y_max
