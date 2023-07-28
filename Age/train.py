import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
    get_csv_for_blend,
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        #save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")


def main():
    train_ds = DRDataset(
        images_folder="/home/ec2-user/SageMaker/SolutionAge/data/training/",
        path_to_csv="/home/ec2-user/SageMaker/SolutionAge/data/trainLabels.csv",
        transform=config.train_transforms,
    )
    val_ds = DRDataset(
        images_folder="/home/ec2-user/SageMaker/SolutionAge/data/training/",
        path_to_csv="/home/ec2-user/SageMaker/SolutionAge/data/valLabels.csv",
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder="/home/ec2-user/SageMaker/SolutionAge/data/training",
        path_to_csv="/home/ec2-user/SageMaker/SolutionAge/data/trainLabels.csv",
        transform=config.val_transforms,
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    
    loss_fn = nn.MSELoss()

    model = EfficientNet.from_pretrained("efficientnet-b7")
    model._fc = nn.Linear(2560, 5)
#     model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
#     scaler = torch.cuda.amp.GradScaler()
    
    
#     print("this is OS list dir, ", os.listdir())
    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        print("found existing check point file")
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
#         print("New Print, Model Parameters", model)
        model._fc = nn.Sequential(*list(model._fc.children())[:-2])
        print("After removing the last layer, Model Parameters", model)
        model._fc = nn.Linear(2560, 1)
#         print("this is the model", model)

    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

#     Run after training is done and you've achieved good result
#     on validation set, then run train_blend.py file to use information
#     about both eyes concatenated
#     get_csv_for_blend(val_loader, model, "/home/ec2-user/SageMaker/SolutionAge/data/val_blend.csv")
#     get_csv_for_blend(train_loader, model, "/home/ec2-user/SageMaker/SolutionAge/data/train_blend.csv")
#     get_csv_for_blend(test_loader, model, "/home/ec2-user/SageMaker/SolutionAge/data/test_blend.csv")
#     make_prediction(model, val_loader, "submission_.csv")
    
#     preds, labels = check_accuracy(val_loader, model, config.DEVICE)
#     print(f"MSE (Validation): {mean_squared_error(labels, preds)}")
    
#     print(f"Rsq (Validation): {r2_score(labels, preds)}")
    
#     print(f"Rsq (Validation): {mean_absolute_error(labels, preds)}")
    
#     preds, labels = check_accuracy(train_loader, model, config.DEVICE)
#     print(f"MSE (train_loader): {mean_squared_error(labels, preds)}")
    
#     print(f"Rsq (train_loader): {r2_score(labels, preds)}")
    
#     print(f"Rsq (train_loader): {mean_absolute_error(labels, preds)}")
    
    
#     import sys
    
#     make_prediction(model, val_loader)
#     sys.exit()

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"MSE (Validation): {mean_squared_error(labels, preds)}")
        print(f"Rsq (Validation): {r2_score(labels, preds)}")
        print(f"Mean Abs Error (Validation): {mean_absolute_error(labels, preds)}")
        
#         preds_train, labels_train = check_accuracy(train_loader, model, config.DEVICE)
#         print(f"MSE (train_loader): {mean_squared_error(labels_train, preds_train)}")
    
#         print(f"Rsq (train_loader): {r2_score(labels_train, preds_train)}")

        # get on train
        #preds, labels = check_accuracy(train_loader, model, config.DEVICE)
        #print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"b3_TransfLearning_{epoch}.pth.tar")



if __name__ == "__main__":
    main()