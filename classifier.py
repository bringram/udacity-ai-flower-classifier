#
# PROGRAMMER:       Brandon Ingram
# DATE CREATED:     Tuesday, February 11, 2020
# REVISED DATE:     
#

from time import time

import torch
from torch import nn, optim
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = { 'resnet': resnet50, 'vgg': vgg16 }


def train(datasets, dataloaders, model_name, epochs, hidden_units, learning_rate,
          save_dir, use_gpu):
    """
    Trains a new model initialized with various customizations. Saves the model
    checkpoint to the given 'save_dir' after training completes.
    
    Parameters:
        datasets - The torchvision.datasets.ImageFolder object collection
        dataloaders - The torch.utils.data.DataLoader object collection
        model_name - The name of the model to use ('vgg' or 'resnet')
        epochs - The number of epochs to train the model for
        hidden_units - The number of hidden units to use in the model
        learning_rate - The learning rate to use in the model
        save_dir - The directory to save the model checkpoint into
        use_gpu - True if training should happen on the GPU, False otherwise
    Returns:
        None
    """
    model, criterion, optimizer = create_model(model_name, hidden_units,
                                               learning_rate,
                                               datasets['train'].class_to_idx)
    
    # Use GPU if available & the 'use_gpu' variable is True
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f'Using {device} to train the {model_name} model')
    
    start_time = time()
    print('Model training starting')
    model.to(device)
    
    running_loss = 0
    print_every = 10
    
    # Begin the "epoch" loop
    for epoch in range(epochs):
        print()
        print(f'Starting epoch {epoch+1} of {epochs}')
        
        # Loop over the training data to train the model
        for step, (inputs, labels) in enumerate(dataloaders['train']):
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % print_every == 0:
                print(f'Epoch {epoch+1} ',
                      f'Training batch: {step} ',
                      f'Training loss: {running_loss/print_every:.3f}')
                running_loss = 0
        
        # Loop over the validation data to validate the model accuracy
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            accuracy = 0
            
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                logps = model(inputs)
                loss = criterion(logps, labels)
                
                valid_loss += loss.item()
                
                # Calculate the accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print(f'Epoch {epoch+1} ',
                  f'Validation loss: {valid_loss/len(dataloaders["valid"]):.3f} ',
                  f'Accuracy: {accuracy/len(dataloaders["valid"])*100:.3f}%')
            
        model.train()
        running_loss = 0
    
    end_time = time()
    tot_time = end_time - start_time
    print()
    print(f'Model training complete, took {tot_time:.3f} seconds')
    
    # Save the model checkpoint
    checkpoint_path = f'{save_dir}flowers-{model_name}-checkpoint.pth'
    print(f'Saving model checkpoint to {checkpoint_path}')
    save_checkpoint(checkpoint_path, model, optimizer, epochs, hidden_units, learning_rate)


def predict(image, checkpoint_path, topk, use_gpu):
    """
    Loads a model from the given checkpoint file and predicts what
    flower the given image is and the related probability.
    
    Parameters:
        image - A NumPy array representing the desired image
        checkpoint_path - The path of the checkpoint file
        topk - The number of probabilities to return
        use_gpu - True if prediction should happen on the GPU,
                  False otherwise
    Returns:
        A tuple consiting of the probabilities, classes and the
        class_to_idx mapping
    """
    model, criterion, optimizer = load_checkpoint(checkpoint_path)
    
    # Use GPU if available & the 'use_gpu' variable is True
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f'Using {device} to predict the given image')
    
    start_time = time()
    print(f'{model.name.title()} model prediction starting')
    model.to(device)
    
    # Run the image through the model to calculate probabilities
    with torch.no_grad():
        model.eval()
        image = image.view(1, 3, 224, 224)
        image = image.to(device)
        predictions = torch.exp(model(image))
        top_ps, top_class = predictions.topk(topk, dim=1)
        
        end_time = time()
        tot_time = end_time - start_time
        print(f'Model prediction complete, took {tot_time:.3f} seconds')
        
        return top_ps, top_class, model.class_to_idx


def save_checkpoint(checkpoint_path, model, optimizer, epochs, hidden_units, learning_rate):
    """
    Saves a checkpoint file for the given 'model' at the given 'checkpoint_path'.
    
    Parameters:
        checkpoint_path - The path of where to save the checkpoint file
        model - The model to save
        optimizer - The optimizer to save
        epochs - The number of epochs to save
        hidden_units - The number of hidden units in the model
        learning_rate - The learning rate used in the optimizer
    Returns:
        None
    """
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'model_name': model.name,
                  'classifier': model.fc if model.name != 'vgg' else model.classifier,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate}
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path):
    """
    Creates a new machine learning model, criterion and optimizer
    from the values in the given checkpoint file
    
    Parameters:
        checkpoint_path - The path of the checkpoint file
    Returns:
        A tuple consisting of the new model, criterion and optimizer
    """
    checkpoint = torch.load(checkpoint_path)
    
    class_to_idx = checkpoint['class_to_idx']
    model_name = checkpoint['model_name']
    epochs = checkpoint['epochs']    
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    
    model, criterion, optimizer = create_model(model_name, hidden_units,
                                               learning_rate, class_to_idx)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, criterion, optimizer


def vgg_classifier(hidden_units):
    """
    Creates a new VGG classifier with 'hidden_units' hidden units.
    
    Parameters:
        hidden_units - The number of hidden units to use in the new classifier
    Returns:
        A new VGG classifier initilized with the given 'hidden_units'
    """
    return nn.Sequential(nn.Linear(25088, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))


def resnet_classifier(hidden_units):
    """
    Creates a new ResNet classifier with 'hidden_units' hidden units.
    
    Parameters:
        hidden_units - The number of hidden units to use in the new classifier
    Returns:
        A new ResNet classifier initilized with the given 'hidden_units'
    """
    return nn.Sequential(nn.Linear(2048, hidden_units),
                         nn.ReLU(),
                         nn.Dropout(0.3),
                         nn.Linear(hidden_units, 102),
                         nn.LogSoftmax(dim=1))


def create_model(model_name, hidden_units, learning_rate, class_to_idx):
    """
    Creates and returns a new model, criterion and optimizer, all of which
    are initialized with the given parameters.
    
    Parameters:
        model_name - The name of the model to use
        hidden_units - The number of hidden units to use in the model classifier
        learning_rate - The learning rate to set on the new optimizer
        class_to_idx - A classes to index mapping object
    Returns:
        A tuple consisting of the new model, criterion and optimizer
    """
    model = models[model_name]
    
    # Freeze model parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.name = model_name
    model.class_to_idx = class_to_idx
    
    if model_name == 'vgg':
        model.classifier = vgg_classifier(hidden_units)
        optimizer = create_optimizer(model.classifier.parameters(), learning_rate)
    elif model_name == 'resnet':
        model.fc = resnet_classifier(hidden_units)
        optimizer = create_optimizer(model.fc.parameters(), learning_rate)
        
    criterion = nn.NLLLoss()
    return model, criterion, optimizer


def create_optimizer(parameters, learning_rate):
    """
    Returns a new Adam optimizer with the given model parameters
    and learning rate.
    
    Parameters:
        parameters - The parameters of the model to optimize
        learning_rate - The learning rate to set on the new optimizer
    Returns:
        A new Adam optimizer
    """
    return optim.Adam(parameters, lr=learning_rate)
