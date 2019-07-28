import matplotlib.pyplot as plt
import sys

from pneumothorax_segmentation.preprocess import get_dicom_data, get_all_images_list

def show_data(folder, index):
    images_list = get_all_images_list(folder)

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    file_path, filename = images_list[index]
    dicom_data = get_dicom_data(file_path)

    # Display the data and image through matplotlib
    plt.imshow(dicom_data.pixel_array)
    print("File: %s" % filename)
    print(dicom_data)
    plt.show()

# Read arguments from python command
folder = None
index = None
for param in sys.argv:
    if param in ["train", "test"]:
        folder = param
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show_raw_data.py [train|test] [id]")
    exit(-1)
if (folder == None):
    print("No folder specified. Defaults to train")
    folder = "train"

show_data(folder, index)
