import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import os
from torch.utils.data import Dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training script for your model.')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    # parser.add_argument('--save_path', type=str, default='saves/', help='Path to save the trained model')
    parser.add_argument('--load_path', type=str, default='saves/', help='Path to load the trained model')
    parser.add_argument('--load_model', type=bool, default=False, help='Load a model or not')
    parser.add_argument('--save_model', type=bool, default=True, help='Save a model or not')
    parser.add_argument('--lamb', type=float, default=0.25, help='Lambda for loss function')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--split_size', type=float, default=0.9, help='Size of the training set')
    parser.add_argument('--exp', type=str, default='0', help='Name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on')
    parser.add_argument('--tune', type=bool, default=True, help='Fine Tune ResNet or not')
    parser.add_argument('--model', type=str, default="resnet", help='model to extract features from')
    parser.add_argument('--cross_attention', action='store_false', default=True, help='use cross attention or not')
    args = parser.parse_args()

    if not os.path.exists(f"saves/exp{args.exp}"):
        os.makedirs(f"saves/exp{args.exp}")

    with open(f'saves/exp{args.exp}/args_log.txt', 'w') as file:
        # Convert the Namespace object to a dictionary and write to the file
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            file.write(f'{key}: {value}\n')
    print('Arguments saved to args_log.txt')
    print(args.cross_attention)

    return args
    
def extract_center_square(image_path):
    img = Image.open(image_path).convert('RGB')

    width, height = img.size
    size = height

    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2

    center_square = img.crop((left, top, right, bottom))
    return center_square

class CustomDataset(Dataset):
    def __init__(self, file_list_path, fold_path, transform=None):
        self.file_list = self.load_file_list(file_list_path)
        self.transform = transform

    def load_file_list(self, file_list_path):
        with open(file_list_path, 'r') as file:
            file_list = [os.path.join('data', f) for f in file.read().splitlines()]
        # print(file_list[:10])
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = extract_center_square(img_path)
        if self.transform:
            img = self.transform(img)
    
        label = self.get_label_from_path(img_path)
        return img, label

    def get_label_from_path(self, img_path):
        file_name = os.path.basename(img_path)
        directory = os.path.dirname(img_path)
        folder_name = os.path.basename(directory) # Base11...
        df = pd.read_excel(f'{directory}/Annotation {folder_name}.xls')
        
        item = df.iloc[df[df['Image name'] == file_name].index, [2,3]].values[0]

        if item[0] < 2:
            item[0] = 0
        else:
            item[0] = 1
        return item

def loss_fn(output, target, criterion=nn.CrossEntropyLoss(), lamb=0.25, cross_attention=True): # output tuple 4, target 2
 
    l_dr_s = criterion(output[0], target[:,0])
    l_dme_s = criterion(output[3], target[:,1])

    if cross_attention:
        l_dr_r = criterion(output[1], target[:,0])
        l_dme_r = criterion(output[2], target[:,1])
        tot_loss = l_dr_r + l_dme_r + lamb* (l_dr_s + l_dme_s)
    else:
        tot_loss = l_dr_s + l_dme_s

    return tot_loss

def eval(model, dataloader, cross_attention, device):
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if cross_attention:
                dr_pred, dme_pred = F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1)
            else:
                dr_pred, dme_pred = F.softmax(outputs[0], dim=1), F.softmax(outputs[3], dim=1)

            dr, dme = labels[:, 0].cpu().numpy(), labels[:, 1].cpu().numpy()

            dr_pred_class = torch.argmax(dr_pred, 1).cpu().numpy()
            dme_pred_class = torch.argmax(dme_pred, 1).cpu().numpy()
            
            dr_f1 = f1_score(dr, dr_pred_class, average='weighted')
            dme_f1 = f1_score(dme, dme_pred_class, average='weighted')
            dr_accuracy = accuracy_score(dr, dr_pred_class)
            dme_accuracy = accuracy_score(dme, dme_pred_class)

    model.train()
    return dr_f1, dme_f1, dr_accuracy, dme_accuracy

def save_checkpoint(state, save_path="saves"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, "my_checkpoint.pth.tar")
    print("=> Saving checkpoint")
    torch.save(state, file_name)


def load_checkpoint(load_path, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

