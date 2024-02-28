import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataset import random_split
from new_model import CANet_reg
from utils import CustomDataset, loss_fn, eval, save_checkpoint, load_checkpoint, get_args

args = get_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = args.device

lr = args.learning_rate
batch_size = args.batch_size
num_epochs = args.epochs
split_size = args.split_size

# save_path = args.save_path
load_path = args.load_path
save_model = args.save_model
load_model = args.load_model

lamb = args.lamb
seed = args.seed

torch.manual_seed(seed)

# setting up tensorboard
writer = SummaryWriter(os.path.join("runs", f"exp{args.exp}"))
step = 0

best_epoch_loss = float('inf')

transform = transforms.Compose(
        [
            transforms.Resize((350, 350)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        

dataset = CustomDataset(file_list_path='data/file_list.txt', fold_path='data/10fold', transform=transform)

total_size = len(dataset)
train_size = int(split_size * total_size)

train_dataset, test_dataset = random_split(dataset, [train_size, total_size - train_size], generator=torch.Generator().manual_seed(seed))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = CANet().to(device)
model = CANet_reg(args.tune, args.model, args.cross_attention).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

if load_model:
    step = load_checkpoint(load_path, model, optimizer)
    
'''
new_state_dict = new_model.state_dict()
for name, param in saved_state_dict.items():
    if name in new_state_dict and param.size() == new_state_dict[name].size():
        new_state_dict[name].copy_(param)

for name, module in new_model.named_modules():
    if 'cbam_dr.spatial_attention.bn' in name or 'cbam_dme.spatial_attention.bn' in name:
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

new_model.load_state_dict(new_state_dict)

print(new_model)
new_model.train()

print("=> Saving checkpoint")
checkpoint = {
            "state_dict" : new_model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "step" : step,
        }
torch.save(checkpoint, 'my_checkpoint.pth.tar')
print("=> Checkpoint saved")
'''


for epoch in range(num_epochs):
    epoch_loss=0
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        #forward pass
        pred = model(image)
        loss = loss_fn(pred, label, lamb=lamb, cross_attention=args.cross_attention)

        epoch_loss += loss.item()
        #backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("Train/Learning rate", current_lr, global_step=step)
        writer.add_scalar("Train/Training_loss", loss.item(), global_step = step)
        step += 1
    
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}, loss = {loss.item():.4f}')

    epoch_loss /= len(train_loader)
    
    dr_f1, dme_f1, dr_accuracy, dme_accuracy = eval(model, test_loader, args.cross_attention, device)
    writer.add_scalar("Metrics/Val/DR F1", dr_f1, global_step = epoch)
    writer.add_scalar("Metrics/Val/DME F1", dme_f1, global_step = epoch)
    writer.add_scalar("Metrics/Val/DR Accuracy", dr_accuracy, global_step = epoch)
    writer.add_scalar("Metrics/Val/DME Accuracy", dme_accuracy, global_step = epoch)
    # print(f"Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_loss}, DR F1: {dr_f1}, DME F1: {dme_f1}, DR Accuracy: {dr_accuracy}, DME Accuracy: {dme_accuracy}")
    if epoch % 5 == 0:
        dr_f1, dme_f1, dr_accuracy, dme_accuracy = eval(model, train_loader, args.cross_attention, device)
        writer.add_scalar("Metrics/Train/DR F1", dr_f1, global_step = epoch)
        writer.add_scalar("Metrics/Train/DME F1", dme_f1, global_step = epoch)
        writer.add_scalar("Metrics/Train/DR Accuracy", dr_accuracy, global_step = epoch)
        writer.add_scalar("Metrics/Train/DME Accuracy", dme_accuracy, global_step = epoch)

    if epoch_loss < best_epoch_loss and save_model:
        best_epoch_loss = epoch_loss
        checkpoint = {
          "state_dict" : model.state_dict(),
          "optimizer" : optimizer.state_dict(),
          "step" : step,
        }
        save_checkpoint(checkpoint, os.path.join("saves", f"exp{args.exp}"))
