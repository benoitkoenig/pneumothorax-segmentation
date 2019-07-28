import pydicom
import matplotlib.pyplot as plt
import os
import sys

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

def show_data(folder, index):
    # Load all images filenames in folder
    all_images_in_folder = []
    for dirName, _, fileList in os.walk("./dicom-images-%s" % folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                all_images_in_folder.append(os.path.join(dirName,filename))

    # Check index is valid
    if index >= len(all_images_in_folder):
        print("Index %s out of range. Max index is %s" % (index, len(all_images_in_folder) - 1))
        exit(-1)

    # Display the data and image through matplotlib
    ds = pydicom.dcmread(all_images_in_folder[index])
    plt.imshow(ds.pixel_array)
    print("File: %s" % all_images_in_folder[index])
    print(ds)
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
