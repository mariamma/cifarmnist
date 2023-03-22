import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
import utils
from resnet_mtl import resnet18, resnet50, resnet34
import random
import math
from logger_utils import Logger
import mnistcifar_utils as mc_utils
from tqdm import trange, tqdm


WEIGHTS_DIR = '/scratch/mariamma/cifar_mnist/minmax-mtl/weights'
RESULTS_DIR = '/scratch/mariamma/cifar_mnist/minmax-mtl/results'

def fit_model(model_name: str, run: str=None, epochs: int=50, fold: int=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_str = "" if run is None or run == "" else f"_{run}"
    num_classes = 10

    checkpoints_dir = f"{WEIGHTS_DIR}/resnet_mtl/{model_name}{run_str}_fold_{fold}"
    tensorboard_dir = f"{RESULTS_DIR}/tensorboard/resnet_mtl/{model_name}{run_str}_fold_{fold}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logger = Logger(tensorboard_dir)
    
    model = resnet34(pretrained=True, num_classes=10).to(device)
    # multilabel_criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    multilabel_criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, verbose=True, factor=0.2
    )

    # load standard, MNIST-randomized, CIFAR-randomized datasets
    mnist_classes = (0, 1)
    cifar_classes = (1, 9)
    batch_size = 128
    # batch_size = 8

    train_loader, val_loader = mc_utils.get_mnist_cifar_dl(mnist_classes=mnist_classes, cifar_classes=cifar_classes, bs=batch_size, 
                                                randomize_mnist=True, randomize_cifar=False)
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    epoch_loss = []

    for epoch_num in trange(epochs, desc="Epochs"):

        model.train()
        train_dataset = iter(train_loader)
        train_losses = torch.zeros(num_classes, dtype=torch.float32)
        train_well_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_to_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_accs = torch.zeros(num_classes, dtype=torch.float32)
        train_acc_arr = []

        for i in range(train_batches):
            # Get data
            data, targets_raw = next(train_dataset)
            targets = F.one_hot(targets_raw, num_classes=10)
            # data, targets = data.to(device), [elt.to(device) for elt in targets]
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            feats = model(data)
            logits = feats
            # print("Logits : ", logits)
            preds = logits.max(1).indices
            # print("Preds : ", preds)
            # print("Raw target : ", targets_raw)
           
            total_loss = multilabel_criterion(logits, targets.float())
            # print("Total loss : ", total_loss)
            class_losses = torch.mean(total_loss, 0)
            # print("Class loss : ", class_losses)

            loss = torch.mean(total_loss)
            loss_classmean = torch.mean(class_losses)
            # print("Loss = ", loss, " Class loss = ", loss_classmean)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))
            logger.scalar_summary("train_loss", np.mean(epoch_loss), epoch_num)
            # Scoring
            # with torch.no_grad():
            #     for task in range(num_classes):
                    # train_losses[task] += class_losses[task].cpu() / train_batches
                    # train_pred[task] += torch.sum(preds[task].cpu(), dim=0)
                    # train_to_pred[task] += torch.sum(targets[task].cpu(), dim=0)
                    # train_well_pred[task] += torch.sum((preds[task]*targets[task]).cpu(), dim=0)
                    # train_accs[task] += torch.mean((preds[task]==targets[task]).cpu().type(torch.float32), axis=0)/train_batches
            
            # Avg scores
            # train_precs = train_well_pred / (train_pred + 1e-7)
            # train_recs = train_well_pred / (train_to_pred + 1e-7)
            # train_fscores = 2*train_precs*train_recs/(train_precs+train_recs+1e-7)
            train_acc =  (preds.cpu() == targets_raw).sum().item() / preds.size(0)
            train_acc_arr.append(train_acc)
            logger.scalar_summary("train_accuracy", train_acc, epoch_num)
            # Out line
            print('Epoch {}, iter {}/{}, Loss : {}, Acc : {}'.format(epoch_num, i+1, train_batches, loss.item(), np.mean(train_acc_arr)), end='\r')
            
        #############
        # Eval loop #
        #############
        model.eval()
        with torch.no_grad(): 
            eval_acc = []
            eval_loss = []
            val_dataset = iter(val_loader)
            val_losses = torch.zeros(num_classes, dtype=torch.float32)
            val_well_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_to_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_accs = torch.zeros(num_classes, dtype=torch.float32)
            for i in range(val_batches):
                print('Eval iter {}/{}'.format(i+1, val_batches), end='\r')
                    
                # Get data
                data, targets_raw = next(val_dataset)
                targets = F.one_hot(targets_raw, num_classes=10)
                data, targets = data.to(device), targets.to(device) # [elt.to(device) for elt in targets]
                
                # Forward
                feats = model(data)
                logits = feats
                # print("Feats : ", feats.shape)
                preds = logits.max(1).indices
                # print("Logits : ", len(logits))
                #total_losses = [torch.mean(criterion(logits[k], targets[k]), 0) for k in range(opt.batch_size)]
                #task_losses = torch.stack([torch.mean(elt) for elt in class_losses])
                total_loss = multilabel_criterion(logits, targets.float())
                class_losses = torch.mean(total_loss, 0)
                # print("Class loss : ", class_losses.shape)

                val_loss = torch.mean(total_loss)
                # print("Loss = ", val_loss, " Class loss = ", class_losses)
                eval_loss.append(val_loss.cpu())
                
                
            
                # Scoring
                # for task in range(num_classes):
                #     val_losses[task] += class_losses[task].cpu() / val_batches
                    # val_pred[task] += torch.sum(preds[task].cpu(), dim=0)
                    # val_to_pred[task] += torch.sum(targets[task].cpu(), dim=0)
                    # val_well_pred[task] += torch.sum((preds[task]*targets[task]).cpu(), dim=0)
                    # val_accs[task] += torch.mean((preds[task]==targets[task]).cpu().type(torch.float32), axis=0)/val_batches
                    
                    
            # Avg scores
            # val_precs = val_well_pred / (val_pred + 1e-7)
            # val_recs = val_well_pred / (val_to_pred + 1e-7)
            # val_fscores = 2*val_precs*val_recs/(val_precs+val_recs+1e-7)
            val_acc =  (preds.cpu() == targets_raw).sum().item() / preds.size(0)
            eval_acc.append(val_acc)
            logger.scalar_summary("val_loss", val_loss.cpu(), epoch_num)
            logger.scalar_summary("val_accuracy", val_acc, epoch_num)

            # Out line
            print('EVAL EPOCH {}, Loss : {}, acc : {},'.format(epoch_num, np.mean(eval_loss), np.mean(eval_acc)))
            scheduler.step(val_loss)
        if np.mean(eval_acc) > 0.70:
            break    



def main():
    model_name = 'resnet34'
    run = 1
    epochs = 2000
    fit_model(model_name, run, epochs)



main()    