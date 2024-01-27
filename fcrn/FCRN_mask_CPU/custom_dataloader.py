import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms

### Custom dataset for gelsight images found on different folders
class GelsightImagesDataset(Dataset):
    def __init__(self, partition_name, network_type='contact'):
        self.data_path = "YCBSight-Sim/" + partition_name + "/"
        items_names_list = np.loadtxt(self.data_path + "names", delimiter=",", dtype='str')
        self.data = []
        self.transforms = transforms.Compose([transforms.PILToTensor()])

        for item in items_names_list:
            for image_index in range(60):
                string_index = str(image_index)
                gelsight_image_path = self.data_path + item + "/gelsight/" + string_index + ".jpg"
                gt_image = None
                gt_image = self.data_path + item + "/gt_height_map/" + string_index + ".npy"

                if (network_type == 'contact'):
                    gt_image = self.data_path + item + "/gt_contact_mask/" + string_index + ".npy"

                self.data.append([gelsight_image_path,gt_image, item + str(image_index)])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        gelsight_image = cv2.imread(self.data[idx][0])
        gelsight_image = torch.from_numpy(gelsight_image).permute(2, 0, 1)
        gt_image = torch.from_numpy(np.load(self.data[idx][1]))

        return gelsight_image, gt_image, self.data[idx][2]
    
def main():
    print("Testing dataloader is able to load all data")
    dataset = GelsightImagesDataset('train', "depth")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for image1, image2, _ in data_loader:
        print(image1)
        print(image2)
    print("All data from partition loaded!!")

if __name__ == '__main__':
    main()
