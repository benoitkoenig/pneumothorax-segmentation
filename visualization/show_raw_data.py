import matplotlib.pyplot as plt
import sys

from pneumothorax_segmentation.get_dicom_data import get_dicom_data

def show_data(folder, index):
    dicom_data, filename = get_dicom_data(folder, index)

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
