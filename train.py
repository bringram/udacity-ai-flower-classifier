#
# PROGRAMMER:       Brandon Ingram
# DATE CREATED:     Monday, February 10, 2020
# REVISED DATE:     
#

import argparse

from classifier import train
from dataloader import load_data


def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window.
    
    Command Line Arguments:
        data_dir - The directory containing the training, testing and validation data
        
    Optional Flags:
        --save_dir - The directory path to save model checkpoint files
        --arch - The CNN model architecture to use
        --learning_rate - The learning rate to use while training the model
        --hidden_units - The number of hidden units to use while training the model
        --epochs - The number of epochs to train for
        --gpu - Will train using the GPU if provided, CPU otherwise
    
    Parameters:
        None
    Returns:
        A data structure that stores the values of the command line arguments
    """
    parser = argparse.ArgumentParser()
    
    # Add the data_dir object - this is not a flag
    parser.add_argument('data_dir', type = str,
                        help = 'The directory containing training, testing and validation data')
    
    # Add all the flag based arguments
    parser.add_argument('--save_dir', type = str, default = './',
                        help = 'The directory path to save model checkpoint files')
    parser.add_argument('--arch', type = str, default = 'vgg',
                        help = 'The CNN model architecture to use')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'The learning rate to use while training the model')
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'The number of hidden units to use while training the model')
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'The number of epochs to train for')
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Use the GPU to train the model')
    
    return parser.parse_args()


def main():
    in_args = get_input_args()
    datasets, dataloaders = load_data(in_args.data_dir)
    train(datasets, dataloaders, in_args.arch, in_args.epochs, in_args.hidden_units,
          in_args.learning_rate, in_args.save_dir, in_args.gpu)


# Call the main function to run the program
if __name__ == "__main__":
    main()
    