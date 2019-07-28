import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

from pneumothorax_segmentation.get_dicom_data import get_dicom_data

def add_masks_on_pixels(input_pixels, masks):
    pixels = copy.deepcopy(input_pixels)
    pixels = np.transpose(pixels, (1, 0))
    pixels = np.reshape(pixels, (-1))

    for mask in masks:
        is_it_a_mask = False
        current_pixel = 0
        for pixel_long_movement in mask:
            if is_it_a_mask:
                pixels[current_pixel] = 255
                pixels[current_pixel + pixel_long_movement] = 255
            current_pixel += pixel_long_movement
            is_it_a_mask = not is_it_a_mask

    final_pixels = np.reshape(pixels, input_pixels.shape)
    final_pixels = np.transpose(final_pixels, (1, 0))
    final_pixels = final_pixels.tolist()

    return final_pixels

def get_masks(name):
    raw_masks = []
    with open('train-rle.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == name:
                raw_masks.append(row[1])

    masks = []
    for raw_mask in raw_masks:
        mask = raw_mask.split(" ")
        mask = mask[1:] # raw_mask starts with a space
        mask = [int(m) for m in mask]
        masks.append(mask)

    return masks

def show_true_mask(index):
    dicom_data, filename = get_dicom_data("train", index) # True mask is only known for the training dataset
    masks = get_masks(filename.replace(".dcm", ""))
    if masks[0][0] == -1:
        print("No Pneumothorax on %s" % filename)
        exit(0)

    pixel_array_with_masks = add_masks_on_pixels(dicom_data.pixel_array, masks)

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show True Mask of %s" % filename)

    plt.imshow(pixel_array_with_masks)

    plt.show()

# Read arguments from python command
index = None
for param in sys.argv:
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show_true_mask.py [id]")
    print("Hint: images 0, 5 and 22 have a mask")
    exit(-1)

show_true_mask(index)
