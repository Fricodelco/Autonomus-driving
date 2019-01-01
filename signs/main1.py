import helpers
import cv2
import random
import numpy as np
import os
import eval1



def load_data():
   

    IMAGE_DIR_TRAINING = "data/training/"
    IMAGE_DIR_VALIDATION = "data/val/"
    IMAGE_DIR_TEST = "data/test/"

    IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
    VALIDATION_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_VALIDATION)

    return IMAGE_LIST, TEST_IMAGE_LIST, VALIDATION_IMAGE_LIST



def standardize_input(image):
 
    standard_im = np.copy(image)

    standard_im = cv2.resize(standard_im, (64, 64))
    return standard_im



def one_hot_encode(label):
    
    one_hot_encoded = []
    if label == "none":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "pedistrain":
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no_drive":
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "stop":
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "way_out":
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "no_entry":
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "road_works":
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "parking":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "a_unevenness":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]
    return one_hot_encoded



def standardize(image_list):
   

    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        
        standardized_im = standardize_input(image)

        
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list



def get_misclassified_images(test_images):
   
    misclassified_images_labels = []
    
    for image in test_images:
    
        im = image[0]
        true_label = image[1]
        
        assert (len(true_label) == 8), "8e"


        predicted_label = eval1.predict_label(im)
        assert (len(predicted_label) == 8), "9e" 

       
        if (predicted_label != true_label):
        
            misclassified_images_labels.append((im, predicted_label, true_label))

   
    return misclassified_images_labels


def main():
   
    IMAGE_LIST, TEST_IMAGE_LIST, VALIDATION_IMAGE_LIST = load_data()
   
    STANDARDIZED_VAL_LIST = standardize(VALIDATION_IMAGE_LIST)
    random.shuffle(STANDARDIZED_VAL_LIST)

  
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_VAL_LIST)
   
    total = len(STANDARDIZED_VAL_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = (100*num_correct) / total
    print('Tochnost ' + str(accuracy))
    print("Non detected " + str(len(MISCLASSIFIED)) + ' from ' + str(total))
    while(1):
        if cv2.waitKey(1)==ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
