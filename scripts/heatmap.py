import torch, torchvision, os
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import mnistcifar_utils as mc_utils
from resnet_mtl import resnet18, resnet50, resnet34
import cv2
from PIL import Image
import torchvision.transforms
import numpy as np
from torchvision.utils import save_image
import torch.nn as nn


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


def generate_cam(model, use_cuda, test_loader, device, heatmap_dir, apply_padding = True, repeat_channels=True):
    dest_dir = []
    orig_img_folder = os.path.join(heatmap_dir, "orig_img")
    dest_dir.append(os.path.join(heatmap_dir, "epsilon_0"))
    dest_dir.append(os.path.join(heatmap_dir, "epsilon_1"))
    dest_dir.append(os.path.join(heatmap_dir, "epsilon_2"))
    dest_dir.append(os.path.join(heatmap_dir, "epsilon_3"))
    os.makedirs(orig_img_folder, exist_ok=True)
    for i in range(4):
        os.makedirs(dest_dir[i], exist_ok=True)

    img_index = 0
    
    target_layers = [model.layer4[-1]]
    multilabel_criterion = nn.CrossEntropyLoss(reduction = 'none')
    epsilons = [0, .001, .005, .01]

    with GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            input_tensor = data # Create an input tensor image for your model
            # Note: input_tensor can be a batch tensor with several images!

            # Construct the CAM object once, and then re-use it on many images:
            # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

            # You can also use it within a with statement, to make sure it is freed,
            # In case you need to re-create it inside an outer loop:
            
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_loss = multilabel_criterion(output, target)
            loss = torch.mean(total_loss)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            
            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category
            # will be used for every image in the batch.
            # Here we use ClassifierOutputTarget, but you can define your own custom targets
            # That are, for example, combinations of categories, or specific outputs in a non standard model.

            targets = [ClassifierOutputTarget(target.item())]

            filename = 'img_' + str(img_index) + 'target_' + str(target.item()) + 'pred_' + str(init_pred.item()) +'.jpg'
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = data[0].cpu().detach().numpy() 
                
            cam_image = show_cam_on_image(np.moveaxis(rgb_img, 0, -1), grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(orig_img_folder, filename), cam_image)

            for e in range(len(epsilons)):
                # Call FGSM Attack
                epsilon = epsilons[e]
                perturbed_data = fgsm_attack(data, epsilon, data_grad)
                output_perturbed = model(perturbed_data)
                final_pred = output_perturbed.max(1, keepdim=True)[1]

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=perturbed_data, targets=targets)

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                rgb_img = data[0].cpu().detach().numpy() 
                
                cam_image = show_cam_on_image(np.moveaxis(rgb_img, 0, -1), grayscale_cam, use_rgb=True)
                filename = 'img_' + str(img_index) + 'target_' + str(target.item()) + 'pred_' + str(final_pred.item()) +'.jpg'
                cv2.imwrite(os.path.join(dest_dir[e], filename), cam_image)
            img_index += 1    


def main():
    WEIGHTS_DIR = '/scratch/mariamma/cifar_mnist/minmax-mtl/weights'
    HEATMAP_DIR = '/scratch/mariamma/cifar_mnist/minmax-mtl/heatmaps'
    pretrained_model = f"{WEIGHTS_DIR}/resnet34_minmax_bce/cifar10_mnist10/ckpt.pt"
    heatmap_dir = f"{HEATMAP_DIR}/resnet34_minmax_bce/cifar10_mnist10/"
    os.makedirs(heatmap_dir, exist_ok=True)

    epsilons = [0, .001, .005, .01, .02, .03, .04, .05]
    num_classes = 10
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    mnist_classes = (0, 1)
    cifar_classes = (1, 9)
    batch_size = 1
    train_loader, test_loader = mc_utils.get_mnist_cifar_dl(mnist_classes=mnist_classes, cifar_classes=cifar_classes, bs=batch_size, 
                                                randomize_mnist=True, randomize_cifar=False)
    model = resnet34(pretrained=True, num_classes=num_classes).to(device)
    checkpoint = torch.load(pretrained_model)
    best_acc = checkpoint['acc']
    best_acc_epoch = checkpoint['epoch']
    print("Model best acc : {} and epoch : {}".format(best_acc, best_acc_epoch))
    # Load the pretrained model
    model.load_state_dict(checkpoint['net'])
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()      
    generate_cam(model, use_cuda, test_loader, device, heatmap_dir)

main()    
