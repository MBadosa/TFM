import torch
import torch.utils.data
import torchvision
import os
from fcrn import FCRN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from custom_dataloader import GelsightImagesDataset 
from torch.utils.data import DataLoader
import sys
import numpy as np
import cv2

dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    training_type = 'depth'
    
    batch_size = 16
    learning_rate = 1.0e-5
    num_epochs = 50
    resume_from_file = False

    print("Loading data......")
    train_dataset = GelsightImagesDataset('train','depth')
    validation_dataset = GelsightImagesDataset('validation','depth')
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2.Load model
    print("Loading model......")
    model = FCRN(batch_size, training_type)
    resnet = torchvision.models.resnet50()
    resnet.load_state_dict(torch.load('./resnet50-19c8e357.pth'))

    print("resnet50 loaded.")
    _ = resnet.state_dict()

    #model.load_state_dict(load_weights(model, weights_file, dtype))
    print("resnet50_pretrained_dict loaded.")
    ### Put model to GPU if exists
    model.to(device)

    # 3.Loss
    loss_fn = torch.nn.MSELoss()
    print("loss_fn set.")

    # 5.Train
    best_val_err = 1.0e3

    start_epoch = 0

    resume_file = 'checkpoint.pth.tar'
    if resume_from_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Optimizer set.")

        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0

        for gelsight_image, gt_image,  _  in train_data_loader:
            input_gelsight_image = gelsight_image.to(device).type(torch.cuda.FloatTensor)

            input_gt_image = gt_image.to(device).type(torch.cuda.FloatTensor)
            
            output = model(input_gelsight_image)
                       
            minibatch_loss =torch.sqrt(loss_fn(output.squeeze(), input_gt_image))

            count += 1
            running_loss += minibatch_loss

            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', float(epoch_loss))

        # Validation
        model.eval()
        num_samples = 0
        loss_local = 0
        with torch.no_grad():
            for gelsight_image, gt_image, _ in validation_data_loader:
                input_gelsight_image = gelsight_image.to(device).type(torch.cuda.FloatTensor)

                input_gt_image = gt_image.to(device).type(torch.cuda.FloatTensor)
                
                output = model(input_gelsight_image)
                minibatch_loss = torch.sqrt(loss_fn(output.squeeze(), input_gt_image.squeeze()))

                input_rgb_image = input_gelsight_image[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                input_gt_image = input_gt_image[0].data.cpu().numpy().astype(np.float32)
                pred_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                input_gt_image /= np.max(input_gt_image)
                pred_image /= np.max(pred_image)

                plot.imsave('train_output/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('train_output/gt_image_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_image, cmap="viridis")
                plot.imsave('train_output/pred_image_epoch_{}.png'.format(start_epoch + epoch + 1), pred_image, cmap="viridis")

                loss_local += minibatch_loss

                num_samples += 1

        err = float(loss_local) / num_samples
        print('val_error:', err)

        if err < best_val_err:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint.pth.tar')

        if epoch % 10 == 0:
            learning_rate = learning_rate * 0.6


if __name__ == '__main__':
    main()
