import pydicom
import os
import sys

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

def get_dicom_data(folder, index):
    # Load all images filenames in folder
    all_images_in_folder = []

    for dirName, _, fileList in os.walk("./dicom-images-%s" % folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                all_images_in_folder.append((os.path.join(dirName,filename), filename))

    # Check index is valid
    if index >= len(all_images_in_folder):
        print("Index %s out of range. Max index is %s" % (index, len(all_images_in_folder) - 1))
        exit(-1)

    # Display the data and image through matplotlib
    ds = pydicom.dcmread(all_images_in_folder[index][0])
    return ds, all_images_in_folder[index][1]
