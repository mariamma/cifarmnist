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
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # gpu = utils.check_gpu()
    # device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    run_str = "" if run is None or run == "" else f"_{run}"
    num_classes = 10

    checkpoints_dir = f"{WEIGHTS_DIR}/resnet34_minmax_ce1/cifar10_mnist10"
    tensorboard_dir = f"{RESULTS_DIR}/tensorboard/resnet34_minmax_ce1/cifar10_mnist10"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logger = Logger(tensorboard_dir)
    
    model = resnet34(pretrained=True, num_classes=num_classes).to(device)
    # multilabel_criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    multilabel_criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, verbose=True, factor=0.2
    )

    # load standard, MNIST-randomized, CIFAR-randomized datasets
    mnist_classes = (0, 1)
    cifar_classes = (1, 9)
    batch_size = 64
    # batch_size = 8

    train_loader, val_loader = mc_utils.get_mnist_cifar_dl(mnist_classes=mnist_classes, cifar_classes=cifar_classes, bs=batch_size, 
                                                randomize_mnist=True, randomize_cifar=False)
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    best_acc = 0

    for epoch_num in trange(epochs, desc="Epochs"):

        model.train()
        train_dataset = iter(train_loader)
        train_acc_arr = []
        epoch_loss = []

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

            loss = torch.max(total_loss)
            loss_classmean = torch.mean(class_losses)
            # print("Loss = ", loss, " Class loss = ", loss_classmean)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))
            logger.scalar_summary("train_loss", np.mean(epoch_loss), epoch_num)
            train_acc =  (preds.cpu() == targets_raw).sum().item() / preds.size(0)
            train_acc_arr.append(train_acc)
            logger.scalar_summary("train_accuracy", train_acc, epoch_num)
            # Out line
            #   
            print('Epoch {}, iter {}/{}, Loss : {}, Acc : {}'.format(epoch_num, i+1, train_batches, loss.item(), np.mean(train_acc_arr)), end='\r')
            
        #############
        # Eval loop #
        #############
        model.eval()
        with torch.no_grad(): 
            eval_acc = []
            eval_loss = []
            val_dataset = iter(val_loader)
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
                # val_loss = torch.max(total_loss)
                # print("Loss = ", val_loss, " Class loss = ", class_losses)
                eval_loss.append(val_loss.cpu())
            
                val_acc =  (preds.cpu() == targets_raw).sum().item() / preds.size(0)
                eval_acc.append(val_acc)
                logger.scalar_summary("val_loss", val_loss.cpu(), epoch_num)
                logger.scalar_summary("val_accuracy", val_acc, epoch_num)

            # Out line
            print('EVAL EPOCH {}, Loss : {}, acc : {},'.format(epoch_num, np.mean(eval_loss), np.mean(eval_acc)))
            scheduler.step(np.mean(eval_loss))
        if np.mean(eval_acc) >= best_acc:
            print('Saving.. for ', np.mean(eval_acc))
            state = {
                'net': model.state_dict(),
                'acc': np.mean(eval_acc),
                'epoch': epoch_num,
            }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            torch.save(state, f"{checkpoints_dir}/ckpt.pt")
            best_acc = np.mean(eval_acc)    
        



def main():
    model_name = 'resnet34'
    run = 1
    epochs = 500
    fit_model(model_name, run, epochs)



main()    