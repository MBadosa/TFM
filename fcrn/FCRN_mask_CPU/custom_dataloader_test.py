import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import glob

### Custom dataset for gelsight images found on different folders
class GelsightImagesDatasetTest(Dataset):
    def __init__(self):
        self.data_path = "inference_data/"
        items_names_list = glob.glob(self.data_path + "*.png")
        self.data = []

        for item in items_names_list:
            self.data.append([item])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        gelsight_image = cv2.imread(self.data[idx][0])
        gelsight_image = torch.from_numpy(gelsight_image).permute(2, 0, 1)
        
        return gelsight_image
    
def main():
    print("Testing test dataloader is able to load all data")
    dataset = GelsightImagesDatasetTest()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for image1 in data_loader:
        print(image1)
    print("All data from partition loaded!!")

if __name__ == '__main__':
    main()
