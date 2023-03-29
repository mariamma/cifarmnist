import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from resnet_mtl import resnet18, resnet50, resnet34
import mnistcifar_utils as mc_utils
import torch.nn as nn
import math

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def get_loss_diff(output_orig, output_perturbed):
    # print("Origin : ", output_orig)
    # print("Perturbed : ", output_perturbed)
    sum = 0
    for i in range(10):
        # print("{} origin={}, perturbed={}".format(i, output_orig.cpu().data.numpy()[0][i], output_perturbed.cpu().data.numpy()[0][i]))
        l = output_orig.cpu().data.numpy()[0][i] - output_perturbed.cpu().data.numpy()[0][i]
        sum += l*l
    sum = sum/10    
    return  math.sqrt(sum)   


def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    original_correct = 0
    adv_correct = 0
    pred_instability = 0
    multilabel_criterion = nn.CrossEntropyLoss(reduction = 'none')
    loss_arr_no_pip = []
    loss_arr_pip = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # print("init_pred.item() : ", init_pred.item())
        # print("target.item() : ", target.item())
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() == target.item():
            original_correct += 1

        # # Calculate the loss
        # loss = F.nll_loss(output, target)
        # print("Output : ", output)
        # print("Target : ", target)
        total_loss = multilabel_criterion(output, target)
        loss = torch.mean(total_loss)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output_perturbed = model(perturbed_data)

        # Check for success
        final_pred = output_perturbed.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            adv_correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

        loss_diff = get_loss_diff(output, output_perturbed)
        
        if init_pred.item() != final_pred.item(): 
            pred_instability += 1
            loss_arr_pip.append(loss_diff)
        else:
            loss_arr_no_pip.append(loss_diff)    
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    # print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print("Epsilon: {}\tOriginal Test Accuracy = {} / {} = {}".format(epsilon, original_correct, len(test_loader), original_correct/float(len(test_loader))))
    print("Epsilon: {}\tAdversarial Test Accuracy = {} / {} = {}".format(epsilon, adv_correct, len(test_loader), adv_correct/float(len(test_loader))))
    print("Epsilon: {}\tPIP Percentage  = {} / {} = {}".format(epsilon, pred_instability, len(test_loader), pred_instability/float(len(test_loader))))
    if len(loss_arr_pip) > 0:
        print("Epsilon: {}\t Percentile of PIP = {}".format(epsilon, np.percentile(loss_arr_pip, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])))
    if len(loss_arr_no_pip) > 0:    
        print("Epsilon: {}\t Percentile without PIP = {}".format(epsilon, np.percentile(loss_arr_no_pip, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])))
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples    


def run_attack(epsilons, model, device, test_loader):
    
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)    
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()    


def main():
    WEIGHTS_DIR = '/scratch/mariamma/cifar_mnist/minmax-mtl/weights'
    pretrained_model = f"{WEIGHTS_DIR}/resnet34_minmax_bce/cifar10_mnist1/ckpt.pt"
    epsilons = [0, .001, .005, .01, .02, .03, .04, .05]
    use_cuda=True  

    # MNIST Test dataset and dataloader declaration
    mnist_classes = (0, 1)
    cifar_classes = (1, 9)
    batch_size = 1
    train_loader, test_loader = mc_utils.get_mnist_cifar_dl(mnist_classes=mnist_classes, cifar_classes=cifar_classes, bs=batch_size, 
                                                randomize_mnist=True, randomize_cifar=False)
    train_batches = len(train_loader)
    val_batches = len(test_loader)
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    model = resnet34(pretrained=True, num_classes=10).to(device)
    checkpoint = torch.load(pretrained_model)
    best_acc = checkpoint['acc']
    best_acc_epoch = checkpoint['epoch']
    print("Model best acc : {} and epoch : {}".format(best_acc, best_acc_epoch))
    # Load the pretrained model
    model.load_state_dict(checkpoint['net'])
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()      
    run_attack(epsilons, model, device, test_loader)


main()