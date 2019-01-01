import numpy as np
import cv2
resY=64
resX=40
def one_hot_encode(label):

    one_hot_encoded = [0,0,0]
    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]

    return one_hot_encoded


def standardize_input(image):
    standard_im = image
   
    return standard_im

def middle(img):
    summ=0
    width=img.shape[0]
    height=img.shape[1]
    for i in range(width):
        for j in range(height):
            try:
                img.shape[2]
                summ+=img.item(i,j,2)
            except:
                summ+=img[i][j]
    
    return summ/(height*width)

def predict_label(image):
    image = cv2.resize(image, (resX, resY))
    temp=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #print(mid)
    array_alpha = np.array([1.0])
    array_beta = np.array([-0.8*middle(temp)]) 
    #cv2.add(image, array_beta, image)                    
    #cv2.multiply(image, array_alpha, image)  
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #image=cv2.medianBlur(image,3)
    
    image=cv.GaussianBlur(image,(3,3),0)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #print(image[60][7])
    
    mask=cv2.threshold(image,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=mask[1]
    #img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    if middle(img[10:25,15:25]) > 100:
        predicted_label = "red"
    elif middle(img[25:40,15:25]) > 100:
        predicted_label = "yellow"
    elif middle(img[40:55,15:25]) > 100:
        predicted_label = "green"    
    else:
	predicted_label = "none"
	cv2.imshow("img",img)
	
    
    encoded_label = one_hot_encode(predicted_label)

    return encoded_label
