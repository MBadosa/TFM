import os
import torch
import numpy as np
from fcrn import FCRN
import matplotlib.pyplot as plot
from custom_dataloader_test import GelsightImagesDatasetTest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = FCRN(1, "contact")

    print("Loading data...")
    test_images = GelsightImagesDatasetTest()

    print("Loading checkpoint...")
    resume_file = os.getcwd() + '/FCRN_mask_CPU/checkpoint.pth.tar'

    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file,  map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))
        
    model.eval()
    idx = 0
    with torch.no_grad():
        for input in test_images:            
            output = model(input.type(torch.FloatTensor).unsqueeze(0))

            print('Predict complete.')

            pred_depth_image = output[0].data.squeeze().numpy().astype(np.uint8)

            plot.imsave('output/pred_contact_mask_{}.png'.format(idx), pred_depth_image, cmap="gray")
            print("Finished infering all images in the inference folder")

if __name__ == '__main__':
    main()

"""

How to perform inference:

Put images inside YCBSight-Sim/inference, the images that we want to perform inference in PNG format

"""