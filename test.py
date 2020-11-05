import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import time
import os
import csv
import cv2
import PIL.Image as Image
from MyModels import initialize_model
from collections import OrderedDict
from tqdm import tqdm


csvFile = open('test_submit.csv', 'w')
writer = csv.writer(csvFile)
writer.writerow(['image_name', 'class'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 224

data_dir = '../../Dataset/ACCV2020'
img_test = os.listdir(os.path.join(data_dir, 'test', 'test'))
model_name = 'ResNext'
batch_size = 256
num_classes = 5000
torch.distributed.init_process_group('nccl', init_method='file:///tmp/somefile', rank=0, world_size=1)
model_test = torch.load('model.pkl', map_location='cuda:0')
model_test = model_test.to(device)
model_test.eval()
#model_test, input_size = initialize_model(model_name, num_classes, features_extract=True, use_pretrained=True)

#model_test = nn.DataParallel(model_test, device_ids=[0, 1, 2])
#state_dict = torch.load('./model.pkl')
#model_test.load_state_dict(torch.load('model.pkl'))
#model_test = model_test.to(device)
#new_state_dict = OrderedDict()
#for k, v in state_dict.item()
#    name = k[7:]
#    new_state_dict[name] = v
#model_test.load_state_dict(new_state_dict)


data_trainsforms_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_datasets_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_trainsforms_test)
data_loader_test = torch.utils.data.DataLoader(
    image_datasets_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

for idx, (x, _) in tqdm(enumerate(data_loader_test)):
    x = x.to(device)
    y = model_test(x)
    probability = torch.nn.functional.softmax(y, dim=1)
    max_value, index = torch.max(probability, 1)
    for i in range(256):
        writer.writerow([img_test[idx*256+i], index.cpu().view(-1).numpy()[i]])
csvFile.close()
