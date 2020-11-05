import warnings
from PIL import Image
warnings.filterwarnings('error')

f1 = open('rgbaError_test.txt', 'w')
f2 = open('rgbaOK_test', 'w')

with open('4chImg_test.txt', 'r')as f:
    for i in f.readlines():
        i = i.strip()
        try:
            img = Image.open(i).convert('RGB')
            f2.write('{}\n'.format(i))

        except Exception as e:
            print('Error:', str(e))
            print(i)
            f1.write('{}\n'.format(i))

f1.close()
f2.close()
