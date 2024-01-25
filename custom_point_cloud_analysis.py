import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import os, glob
import operator

def load_display_camera(file_path):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.load(file_path))

    o3d.visualization.draw_geometries([pcd1])

def display_gt_images(image_number):
    contact_mask = np.load("test_YCB/gt_contact_mask/{}.npy".format(image_number))
    deapth_map = np.load("test_YCB/gt_height_map/{}.npy".format(image_number))
    gelsight_image = cv2.imread("test_YCB/gelsight/{}.jpg".format(image_number))
    _, axis = plt.subplots(3)
    axis[0].imshow(contact_mask, cmap='gray')
    axis[1].imshow(deapth_map, cmap='viridis')
    axis[2].imshow(gelsight_image)
    plt.show()

def height_map_to_pointcloud(height_map):
    height_map_copy = copy.copy(height_map)
    pixmm = 0.0295; 
    max_depth = 1.0; 
    max_z = max_depth/pixmm; 

    height_map_copy = -1 * (height_map_copy - max_z)
    size_x = height_map_copy.shape[1]
    size_y = height_map_copy.shape[0]

    x, y = np.meshgrid(np.arange(size_x) - size_x / 2, np.arange(size_y) - size_y / 2)

    # Reshape the arrays to column vectors
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    height_map_copy = np.array(height_map_copy).reshape(-1, 1)

    # Concatenate the column vectors to create the heightmap_3d array
    heightmap_3d = np.concatenate((x, y, height_map_copy), axis=1)
    computed_pointcloud = o3d.geometry.PointCloud()
    scaled_computed_pointcloud = np.asarray(heightmap_3d)*pixmm/1000
    computed_pointcloud.points = o3d.utility.Vector3dVector(scaled_computed_pointcloud)

    ### Remove 0 elements by height
    points_to_filter = np.asarray(computed_pointcloud.points)
    unique, counts = np.unique(points_to_filter, return_counts=True)
    index, _ = max(enumerate(counts), key=operator.itemgetter(1))
    mask_height_indices = np.where(points_to_filter == unique[index])[0]
    return  computed_pointcloud.select_by_index(mask_height_indices, invert = True)

def main():
    """
        From gelsight images + contact masks and depth estimations, generate the masked death maps and the pointclouds out of those depth masks

        The information that needs to be infered will be structured as follow:
            inside inference_data folder put all the folders that want to be infered.
                Inside each folder the structure must be as follows:
                    Folder named: gelisight, which contains the gelsight images in png format
                    Folder named: contact_masks, which contains the images of the contact mask in npy format 
                    Folder names: height_map, which contains the images of the height map in npy format
        
                    So the program will look in for all the folders inside "inference_data" and for each one it will extract the masked gelsight images as well
                    as the pointclouds of it, storing it all in a folder names masking_result whith the same folder structure inside as the items to perform 
                    inference on.

    """
    ### First of all read all folders inside inference_data folder:
    inference_objects_folders_name = [name for name in os.listdir("./inference_data") if os.path.isdir(os.path.join("./inference_data", name))]
    inference_objects_folders_name = sorted(inference_objects_folders_name)
    ### Create output directory if not exists
    if not os.path.exists("masking_results"):
        os.makedirs("masking_results")

    for object_folder_name in inference_objects_folders_name:
        main_object_folder_name = "./masking_results/" + object_folder_name

        if not os.path.exists("./masking_results/" + object_folder_name):
            os.makedirs(main_object_folder_name)
            os.makedirs(main_object_folder_name + "/height_masked")
            os.makedirs(main_object_folder_name + "/filtered_pointclouds")
            os.makedirs(main_object_folder_name + "/segmented_gelsight_image")

        images_in_object = glob.glob("./inference_data/" + object_folder_name + '/gelsight/*.jpg')
        for image_index in range(len(images_in_object)):
            ### Load images
            gelsight_image = cv2.imread("./inference_data/" + object_folder_name + "/gelsight/{}.jpg".format(image_index))
            heightmap_image = np.load("./inference_data/" + object_folder_name + "/height_map/{}.npy".format(image_index))
            contact_mask = np.load("./inference_data/" + object_folder_name + "/contact_mask/{}.npy".format(image_index))

            ### Use the masks to segment the gelsightr and heightmap images
            segmented_gelsight_image = copy.copy(gelsight_image)
            segmented_gelsight_image[contact_mask == False] = (0,0,0)

            segmented_heightmap_image = copy.copy(heightmap_image)
            segmented_heightmap_image[contact_mask == False] = 0

            filtered_pointcloud = height_map_to_pointcloud(segmented_heightmap_image)
            ### Store masked images
            np.save(main_object_folder_name + "/height_masked/heighmap_masked_{}.jpg".format(image_index), segmented_heightmap_image)
            cv2.imwrite(main_object_folder_name + "/segmented_gelsight_image/gelsight_masked_{}.jpg".format(image_index), segmented_gelsight_image)
            o3d.io.write_point_cloud(main_object_folder_name + "/filtered_pointclouds/filtered_pointcloud_{}.pcd".format(image_index),filtered_pointcloud,write_ascii=True)
            
if __name__ == "__main__":
    main()
