# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import cv2
import numpy as np

def create_mask(cam_num):
# load image as pixel array
    image_ori = image.imread(f'Camera{cam_num}.jpg')
    mask_np = image.imread(f'Camera{cam_num}_mask.jpg')

    mask = np.all(mask_np == [255, 255, 255], axis=-1)
    binary_mask = np.where(mask, 0, 1)
    # save the binary mask to npy file
    np.save(f'Camera{cam_num}_mask.npy', binary_mask)

    # validate the mask
    covered_image = image_ori *binary_mask[:, :, None]

    pyplot.imshow(covered_image)
    pyplot.show()
if __name__ == '__main__':
    # run the function to create mask for camera 5
    create_mask(5)