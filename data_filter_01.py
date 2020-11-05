import os
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('error')

root = '../../Dataset/ACCV2020/test'

f1 = open('pExifError_test.txt', 'w')
f2 = open('rgbaError_test.txt', 'w')
f3 = open('ExifError_test.txt', 'w')
f4 = open('4chImg_test', 'w')
f5 = open('WebpError_test.txt', 'w')
f6 = open('UnknownError_test.txt', 'w')

idx = 0
for r, d, files in os.walk(root):
    if files != []:
        for i in files:
            fp = os.path.join(r, i)
            try:
                img = Image.open(fp)
                if (len(img.split()) != 3):
                    f4.write('{}\n'.format(fp))
            except Exception as e:
                print('Error:', str(e))
                print(fp)
                if 'Possibly corrupt EXIF data' in str(e):
                    print('Exif error')
                    f1.write('{}\n'.format(fp))
                elif 'Palette images with Transparency' in str(e):
                    print('rgba error')
                    f2.write('{}\n'.format(fp))
                elif 'Corrupt EXIF data' in str(e):
                    print('pExif error')
                    f3.write('{}\n'.format(fp))
                elif 'image file could not be identified because WEBP' in str(e):
                    print('Webp error')
                    f5.write('{}\n'.format(fp))
                else:
                    print('Unknown error')
                    f6.write('{}\n'.format(fp))

            if idx % 5000 == 0:
                print('='*20, idx)

            idx += 1
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
