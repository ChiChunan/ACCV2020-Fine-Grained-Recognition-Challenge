import os
import piexif
import warnings
from PIL import Image
warnings.filterwarnings('error')

files = ['ExifError_test.txt', 'pExifError_test.txt']

for file in files:
    with open(file, 'r')as f:
        for i in f.readlines():
            i = i.strip()
            print(i.strip())
            piexif.remove(i.strip())