import pytorch_lightning as pl
import glob
import torchvision 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
from tqdm.auto import tqdm
from util.misc import nested_tensor_from_tensor_list
import random

import os
from torch.utils.data.sampler import SubsetRandomSampler
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

def collate_fn_st(batch):
    """
    Collate function for the ImagePairsDataset.

    Args:
        batch (list): List of samples in a batch.

    Returns:
        batch (tuple): Tuple of samples in a batch.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = nested_tensor_from_tensor_list(batch[1])
    return tuple(batch)

class ImagePairsDataset(torchvision.datasets.CocoDetection):
    """
    Custom dataset class for loading pairs of content and style images.
    """
    def __init__(self,img_size, content_images, style_images,
                 style_folder,content_folder,
                  transform=None ):
        """
        Initialize the ImagePairsDataset.

        Args:
            img_size (int): Size of the images.
            content_images (list): List of content image filenames.
            style_images (list): List of style image filenames.
            style_folder (str): Path to the style images folder.
            content_folder (str): Path to the content images folder.
            transform: Optional transform to be applied to the images.
        """
        self.img_size = img_size
        self.transform = transform
        self.style_images = style_images
        self.content_images = content_images
        self.style_folder= style_folder
        self.content_folder=content_folder

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            length (int): Number of samples in the dataset.
        """
        return len(self.style_images)
    
    def img_resize(self,im,div=32):#32
        """
        Resize the image to the desired size while maintaining the aspect ratio.

        Args:
            im (PIL.Image): Input image.
            div (int): Divisor for rounding the new size.

        Returns:
            noise_new_im (numpy.ndarray): Resized and converted image array.
        """
        desired_size=self.img_size
        h, w = im.size

        if h>w:
            new_h=desired_size
            new_w=int(new_h/h*w)
        else:
            new_w=desired_size
            new_h=int(new_w/w*h)
            
        new_w = (new_w%div==0) and new_w or (new_w + (div-(new_w%div)))
        new_h = (new_h%div==0) and new_h or (new_h + (div-(new_h%div)))
        new_im  = im.resize((new_h,new_w), Image.ANTIALIAS).convert("RGB")


        new_im=np.array(new_im)
        return new_im
    
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            content_image (Tensor): Content image tensor.
            style_image (Tensor): Style image tensor.
            target (dict): Target dictionary containing the content and style image names.
        """
        content_index = index
        style_index = random.choice(range(len(self.style_images))) 

        s_name = self.style_images[style_index]
        c_name = self.content_images[content_index]

        style_image_path = os.path.join(self.style_folder, s_name)
        content_image_path = os.path.join(self.content_folder, c_name)



        target = {'content_image_name': c_name, 'style_image_name': s_name}

        # Load images
        content_image = Image.open(content_image_path).convert("RGB")
        content_image = self.img_resize(content_image)
       
        style_image = Image.open(style_image_path).convert("RGB")
        style_image = self.img_resize(style_image)

        # Applying transformations if defined
        if self.transform is not None:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image, target
        
class DataModule(pl.LightningDataModule):
    """
    DataModule for loading and preparing the data for training and validation.
    """
    def __init__(self, args):
        """
        Initialize the DataModule.

        Args:
            img_size (int): Size of the images.
            style_folder (str): Path to the style images folder.
            content_folder (str): Path to the content images folder.
            batch_size (int): Batch size for the dataloaders.
        """
        super().__init__()
        style_folder = args.style_folder,#"/data/wikiart",#'/app/STTR/images/style',#"/data/wikiart"
        content_folder= args.content_folder,# "/data/train2014",  #'/app/STTR/images/content',#"/data/train2014",  
                    
        self.std = std
        self.mean = mean
        self.batch_size = args.batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        
        self.style_images = glob.glob(str( style_folder) + '/*'+ '/*')
        self.content_images =glob.glob(str(content_folder) + '/*')
        self.img_size=args.img_size
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Set up the data for training and validation.

        Args:
            stage: Optional stage argument (used by PyTorch Lightning).
        """
 
        dataset = ImagePairsDataset(self.img_size,  self.content_images, self.style_images, 
                                    self.style_folder,self.content_folder,
                                    transform=self.transform)
        # Split dataset into training, validation, and test sets
 
        train_size = int(0.9 * len(dataset))
        val_size = int(0.1 * len(dataset))
        
        indices = list(range(len(dataset)))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices, )
        self.dataset = dataset

    def train_dataloader(self):
        # Return the training dataloader
        dataloader = DataLoader(self.dataset,collate_fn=collate_fn_st,sampler=self.train_sampler, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def val_dataloader(self):
        # Return the validation dataloader
        dataloader = DataLoader(self.dataset,collate_fn=collate_fn_st, sampler=self.val_sampler,batch_size=self.batch_size,   num_workers=2)
        return dataloader

