#!/usr/bin/env python
# coding: utf-8

# In[2]:



#from google.colab import drive
#drive.mount('/content/drive')

# In[11]:


#rgb_img = '/content/drive/Shareddrives/ACM_Project_Team_3/HYTA-master/3GT/training'
#seg_img = '/content/drive/Shareddrives/ACM_Project_Team_3/HYTA-master/images/training'
#rgb_img = '/content/drive/Shareddrives/ACM_Project_Team_3/WSISEG-Database-master/whole sky images'
#seg_img = '/content/drive/Shareddrives/ACM_Project_Team_3/WSISEG-Database-master/annotation'

rgb_img_training = "./Data/WSISEG-Database-master/whole sky images"
seg_img_training = "./Data/WSISEG-Database-master/annotation"
rgb_img_testing = "./Data/HYTA-master/images/training"
seg_img_testing = "./Data/HYTA-master/3GT/training-fixed"

# In[10]:
#!pip3 install segmentation_models_pytorch
#!pip3 install natsort
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
class CustomDataSet(Dataset):
        def __init__(self, main_dir, label_dir, transform):
            self.main_dir = main_dir
            self.label_dir = label_dir
            self.transform = transform
            all_imgs = os.listdir(main_dir)
            all_segs = os.listdir(main_dir)
            self.total_imgs = natsorted(all_imgs)
            self.total_segs = natsorted(all_segs)

        def __len__(self):
            return len(self.total_imgs)

        def __getitem__(self, idx):
            img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            #tensor_image = transform(image)
            #mean, std = tensor_image.mean([1,2]), tensor_image.std([1,2])
            #print(mean, std, type(tensor_image))

            seg_loc = os.path.join(self.label_dir, self.total_segs[idx])
            labeled_image = Image.open(seg_loc).convert("RGB")
            transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            labeled_image = transform(labeled_image)
            labeled_image = labeled_image.float()
            tensor_image = tensor_image.float()
            return tensor_image, labeled_image

if __name__ == '__main__':
    print("importing torch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision


    from  torchvision.models.segmentation import  deeplabv3_resnet50

    import segmentation_models_pytorch as smp
    from numpy import asarray
    import numpy as np

        #/content/drive/Shareddrives/ACM_Project_Team_3 


        # Raw images
        #/content/drive/Shareddrives/ACM_Project_Team_3/HYTA-master/images/training

        # Labels 
        # /content/drive/Shareddrives/ACM_Project_Team_3/HYTA-master/3GT/training



    from natsort import natsorted

    def getDataStats(folder):
        min_abs = 1
        max_abs = 0
        mean_sum = 0
        std_sum = 0
        i = 0
        for image in os.listdir(folder):
            #try:
            img_loc = os.path.join(folder, image)
            image = Image.open(img_loc).convert("RGB")
            transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
            tensor_image = transform(image)
            mean, std, max_, min_ = tensor_image.mean([1,2]), tensor_image.std([1,2]), tensor_image.max(), tensor_image.min()
            mean_sum += mean
            std_sum += std
            i += 1
            min_abs = min(min_abs, min_)
            max_abs = max(max_abs, max_)
            if (max_ < 0 or min_ < 0):
                print(image, min, max)
            #except:
            #  print(image, "failed")
            #  continue
        mean = mean_sum / i
        std = std_sum / i
        return mean, std, min_abs, max_abs


    

    BATCH_SIZE = 16
    ## transformations  
    size = (256, 256)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size), transforms.Normalize([0.3114, 0.3166, 0.3946], [0.2580, 0.2593, 0.2953])])


    #UNCOMMENT TO FIND MEAN AND STD OF DATASET
    #mean, std = getDataStats(rgb_img)
    #print("mean and std before normalize:")
    #print("Mean of the image:", mean)
    #print("Std of the image:", std)


    ## download and load training dataset
    imagenet_data = CustomDataSet(rgb_img_training, seg_img_training, transform=transform)
    trainloader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=2,)


    ## download and load training dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size), transforms.Normalize([0.3663, 0.4620, 0.5813], [0.1548, 0.1313, 0.1024])])
    imagenet_data = CustomDataSet(rgb_img_testing, seg_img_testing, transform=transform)
    testloader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=2)

    #rgb_img_t = '/content/drive/MyDrive/Highway_Dataset/Test/TestSeq04/image'
    #seg_img_t = '/content/drive/MyDrive/Highway_Dataset/Test/TestSeq04/label'
    ### download and load training dataset
    #imagenet_data_test = CustomDataSet(rgb_img_t, seg_img_t, transform=transform)
    #testloader = torch.utils.data.DataLoader(imagenet_data_test,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=False,
    #                                          num_workers=2)
    #

    #for images, labels in trainloader:
        #print("batch size:", images.shape)

    # In[8]:


    import segmentation_models_pytorch as smp
    learning_rate = 0.1
    num_epochs = 10

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = MyModel()
    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # model to eval() model and load onto computation devicce
    #model.eval().to(device)
    #model = model.to(device)
    model = smp.MAnet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        classes=3,
        in_channels=3,
        #encoder_depth=5
    )

    from segmentation_models_pytorch.encoders import get_preprocessing_fn

    preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')

    for images, labels in trainloader:
        print("batch size:", images.shape)
        out = model(images)
        pr_mask = out.sigmoid()
        print(out.shape, labels.shape)
        break

    criterion = nn.MSELoss()

    loss = smp.utils.losses.DiceLoss()
    
    # using multiple metrics to train the model
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
    ]

    # Using Adam optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000007) #0.000075

    DEVICE = 'cuda'#'cuda'

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )




    max_score = 0
    count = 0
    torch.cuda.empty_cache()
  
    for i in range(0, 35):
        torch.cuda.empty_cache()
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(trainloader)
        torch.cuda.empty_cache()
        valid_logs = valid_epoch.run(trainloader)#testloader)
    
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model-MAnet#5.pth')
            print('Model saved!')
            count = 0
        else:
            count += 1
            if (count == 3):
                break 
            
        #if i == 3:
        #    optimizer.param_groups[0]['lr'] = 1e-5
        #    print('Decrease decoder learning rate to 1e-5!')
        model.eval() 
        torch.cuda.empty_cache() 

    # In[27]:


    import matplotlib.pyplot as plt
    #https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

    # In[37]:

    run_images = input("good? y/n: ")
    if (run_images == "y"):
        batch = next(iter(testloader))
        with torch.no_grad():
            model.eval()
            logits = model(batch[0].cuda())
        pr_masks = logits.sigmoid()
        
        for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image.cpu().numpy().transpose(1, 2, 0))  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().transpose(1, 2, 0).squeeze()) # just squeeze classes dim, because we have only one class
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.cpu().numpy().transpose(1, 2, 0).squeeze()) # just squeeze classes dim, because we have only one class
            plt.title("Prediction")
            plt.axis("off")

            plt.show()