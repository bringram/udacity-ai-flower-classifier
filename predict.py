#
# PROGRAMMER:       Brandon Ingram
# DATE CREATED:     Monday, February 10, 2020
# REVISED DATE:     
#

import argparse
import numpy as np
from PIL import Image

from classifier import predict
from dataloader import load_json
from imageloader import process_image


def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window.
    
    Command Line Arguments:
        image_path - The path to the image to predict
        checkpoint - The path to the model checkpoint file
        
    Optional Flags:
        --top_k - Returns the top K most likely classes
        --category_names - Use a mapping of categories to real names
        --gpu - Will predict using the GPU if provided, CPU otherwise
    
    Parameters:
        None
    Returns:
        A data structure that stores the values of the command line arguments
    """
    parser = argparse.ArgumentParser()
    
    # Add the non-flag arguments
    parser.add_argument('image_path', type = str,
                        help = 'The path to the image to predict')
    parser.add_argument('checkpoint', type = str,
                        help = 'The path to the model checkpoint file')
    
    # Add all the flag based arguments
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'Return the top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'Use a mapping of categories to real names')
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Will predict using the GPU if provided, CPU otherwise')
    
    return parser.parse_args()


def main():
    in_args = get_input_args()
    
    # Process the input image and run the model prediction
    image = process_image(Image.open(in_args.image_path))
    top_ps, top_class, class_to_idx = predict(image, in_args.checkpoint,
                                              in_args.top_k, in_args.gpu)
    
    flower_to_name = load_json(in_args.category_names)
    
    # Map the class_to_idx labels to flower names from the given JSON file
    flower_name_dict = {}
    for flower, label in class_to_idx.items():
        flower_name_dict[label] = flower_to_name.get(flower)
    
    # Create NumPy 1D arrays out of the probabilities & class labels
    probs = np.atleast_1d(top_ps.data.cpu().numpy().squeeze())
    classes = np.atleast_1d(top_class.data.cpu().numpy().squeeze())
    
    # Convert the labels to flower names & combine with the probabilities
    # then sort them in descending order by the probabilities
    class_labels = [flower_name_dict[class_label] for class_label in classes]
    results = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)
    
    # Print the prediction results
    print()
    for i, (label, prob) in enumerate(results):
        print(f'{i+1}. {label}: {prob*100:.3f}%')


# Call the main function to run the program
if __name__ == "__main__":
    main()
    