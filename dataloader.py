#
# PROGRAMMER:       Brandon Ingram
# DATE CREATED:     Thursday, February 13, 2020
# REVISED DATE:     
#

import json
import torch
from torchvision import datasets, transforms


def load_data(data_dir):
    """
    Loads the data from the given 'data_dir' into the torchvision
    datasets and dataloaders.
    
    Parameters:
        data_dir - the directory to load
    Returns:
        A tuple consisting of the torchvision datasets and dataloaders
    """
    train_dir = f'{data_dir}/train'
    valid_dir = f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'
    
    data_transforms = generate_transforms()
    datasets = create_datasets(train_dir, valid_dir, test_dir, data_transforms)
    dataloaders = create_dataloaders(datasets)
    
    return datasets, dataloaders


def load_json(filename):
    """
    Loads the given JSON file and returns it.
    
    Parameters:
        filename - the name of the JSON file to load
    Returns:
        The processed JSON file dictionary
    """
    with open(filename, 'r') as f:
        return json.load(f)
   

def generate_transforms():
    """
    Generates the image transforms for training, validating and testing
    a machine learning model.
    
    Parameters:
        None
    Returns:
        A data transforms dictionary containing the transformations
    """
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    data_transforms = {}
    data_transforms['train'] = train_transforms
    data_transforms['valid'] = test_valid_transforms
    data_transforms['test'] = test_valid_transforms
    
    return data_transforms


def create_datasets(train_dir, valid_dir, test_dir, data_transforms):
    """
    Creates the image datasets via the torchvision.datasets.ImageFolder
    method.
    
    Parameters:
        train_dir - The directory containing training images
        valid_dir - The directory containing validation images
        test_dir - The directory containing test images
        data_transforms - A dictonary containing the image transformations to use
    """
    img_datasets = {}
    img_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    img_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    img_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    return img_datasets
    

def create_dataloaders(datasets):
    """
    Creates and returns a dictionary of torch.utils.data.DataLoader objects.
    
    Parameters:
        datasets - A dictionary of datasets
    Returns:
        A dictionary containing the dataloader objects
    """
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'], batch_size=32)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=32)
    return dataloaders
