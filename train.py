import torch
import os
from tqdm import tqdm

from torch import nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import copy
from MyModels import initialize_model
import warnings
warnings.filterwarnings('ignore')

print('Pytorch Version', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model:nn.Module, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            for step, (inputs, labels) in tqdm(enumerate(data_loader[phase])):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.autograd.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                print(loss.item())
                preds = torch.argmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time()-since
        print('One epoch training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        model.load_state_dict(best_model_wts)
    return model




data_dir = '../../Dataset/ACCV2020'
model_name = 'ResNext'
num_classes = 5000
batch_size = 128*3

num_epochs = 10
model_ft, input_size = initialize_model(model_name, num_classes, features_extract=True, use_pretrained=True)
torch.backends.cudnn.benchmark = True
torch.distributed.init_process_group(backend='nccl')
model_ft = model_ft.to(device)
model_ft = nn.parallel.DistributedDataParallel(model_ft)

print(input_size)


data_trainsforms_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_trainsforms_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_trainsforms_train)
image_datasets_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_trainsforms_test)
datasets_size = len(image_datasets)
train_size = int(0.8*len(image_datasets))
val_size = len(image_datasets) - train_size

data_loader_test = torch.utils.data.DataLoader(
    image_datasets_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.00005)
exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00005, max_lr=0.0002, step_size_up=int(2*(train_size // batch_size)), cycle_momentum=False, mode='exp_range')

num_folds = 5
for fold in range(num_folds):
    image_datasets_train, image_datasets_val = torch.utils.data.random_split(image_datasets, [train_size, val_size])
    data_loader_train = torch.utils.data.DataLoader(
        image_datasets_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        pin_memory=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        image_datasets_val,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    data_loader = {'train':data_loader_train, 'val':data_loader_val}
    model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)


torch.save(model_ft, './model.pkl')