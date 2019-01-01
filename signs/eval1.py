import helpers
import cv2
import random
import numpy as np

def get_standatr_signs():
   
    a_unevenness = cv2.imread("data/standards/a_unevenness.jpg",0)
    a_unevenness = cv2.resize(a_unevenness, (64, 64))
    a_unevenness=cv2.GaussianBlur(a_unevenness,(3,3),0)
    #a_unevenness = cv2.inRange(a_unevenness, (89, 91, 60), (255, 255, 255))
    mask=cv2.threshold(a_unevenness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    a_unevenness=mask[1]

    no_drive = cv2.imread("data/standards/no_drive.png",0)
    no_drive = cv2.resize(no_drive, (64, 64))
    no_drive=cv2.GaussianBlur(no_drive,(3,3),0)
    #no_drive = cv2.inRange(no_drive, (89, 91, 149), (255, 255, 255))
    mask=cv2.threshold(no_drive,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    no_drive=mask[1]

    no_entry = cv2.imread("data/standards/no_entry.jpg",0)
    no_entry = cv2.resize(no_entry, (64, 64))
    no_entry=cv2.GaussianBlur(no_entry,(3,3),0)
    #no_entry = cv2.inRange(no_entry, (89, 91, 149), (255, 255, 255))
    mask=cv2.threshold(no_entry,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    no_entry=mask[1]
    
    parking = cv2.imread("data/standards/parking.jpg",0)
    parking = cv2.resize(parking, (64, 64))
    parking=cv2.GaussianBlur(parking,(3,3),0)
    #parking = cv2.inRange(parking, (89, 91, 141), (255, 255, 255))
    mask=cv2.threshold(parking,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    parking=mask[1]

    pedistrain = cv2.imread("data/standards/pedistrain.jpg",0)
    pedistrain = cv2.resize(pedistrain, (64, 64))
    pedistrain=cv2.GaussianBlur(pedistrain,(3,3),0)
    #pedistrain = cv2.inRange(pedistrain, (89, 91, 149), (255, 255, 255))
    mask=cv2.threshold(pedistrain,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pedistrain=mask[1]

    road_works = cv2.imread("data/standards/road_works.jpg",0)
    road_works = cv2.resize(road_works, (64, 64))
    road_works=cv2.GaussianBlur(road_works,(3,3),0)
    #road_works = cv2.inRange(road_works, (89, 91, 166), (255, 255, 255))
    mask=cv2.threshold(road_works,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    road_works=mask[1]

    stop = cv2.imread("data/standards/stop.jpg",0)
    stop = cv2.resize(stop, (64, 64))
    stop=cv2.GaussianBlur(stop,(3,3),0)
    #stop = cv2.inRange(stop, (89, 91, 149), (255, 255, 255))
    mask=cv2.threshold(stop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    stop=mask[1]

    way_out = cv2.imread("data/standards/way_out.jpg",0)
    #way_out = cv2.inRange(way_out, (89, 91, 149), (255, 255, 255))
    way_out=cv2.GaussianBlur(way_out,(3,3),0)
    way_out = cv2.resize(way_out, (64, 64))
    mask=cv2.threshold(way_out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    way_out=mask[1]
    
    cv2.imshow("a_unevenness",a_unevenness)
    cv2.imshow("no_drive",no_drive)
    cv2.imshow("no_entry",no_entry)
    cv2.imshow("parking",parking)
    cv2.imshow("pedistrain",pedistrain)
    cv2.imshow("road_works",road_works)
    cv2.imshow("stop",stop)
    cv2.imshow("way_out",way_out)
    standart_signs = {
        "a_unevenness": a_unevenness,
        "no_drive": no_drive,
        "no_entry": no_entry,
        "parking": parking,
        "pedistrain": pedistrain,
        "road_works": road_works,
        "stop": stop,
        "way_out": way_out
    }
    return standart_signs

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

def predict_label(image):

    standart_signs = get_standatr_signs()
    a_unevenness = standart_signs["a_unevenness"]
    no_drive = standart_signs["no_drive"]
    no_entry = standart_signs["no_entry"]
    parking = standart_signs["parking"]
    pedistrain = standart_signs["pedistrain"]
    road_works = standart_signs["road_works"]
    stop = standart_signs["stop"]
    way_out = standart_signs["way_out"]

    predicted_label = [0, 0, 0, 0, 0, 0, 0, 1]
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.medianBlur(image,3)
    mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image=mask[1]
    image=cv2.medianBlur(image,3)
    #image=cv2.erode(image,(5,5),iterations=3)
    a_unevenness_val = 0
    no_drive_val = 0
    no_entry_val = 0
    none_val = 0
    parking_val = 0
    pedistrain_val = 0
    road_works_val = 0
    stop_val = 0
    way_out_val = 0

    cv2.imshow("rgb_image", image)
    which_one=np.array([0,0,0,0,0,0,0,0,0])
    for i in range(64):
        for j in range(64):
            if image[i][j] == a_unevenness[i][j]:
                which_one[0] += 1
            if image[i][j] == no_drive[i][j]:
                which_one[1]+= 1
            if image[i][j] == no_entry[i][j]:
                which_one[2] += 1
            if image[i][j] == parking[i][j]:
                which_one[3] += 1
            if image[i][j] == pedistrain[i][j]:
                which_one[4] += 1
            if image[i][j] == road_works[i][j]:
                which_one[5] += 1
            if image[i][j] == stop[i][j]:
                which_one[6] += 1
            if image[i][j] == way_out[i][j]:
                which_one[7] += 1
    #print(a_unevenness_val,no_drive_val,no_entry_val,parking_val,pedistrain_val,road_works_val,stop_val,way_out_val)
    maxx=which_one[0]
    count=0
    for i in range(8):
            if which_one[i]>maxx:
                maxx=which_one[i]
                count=i
    if maxx>2900:            
        if count == 0:
            predicted_label = one_hot_encode("a_unevenness")
        elif count == 1:
            predicted_label = one_hot_encode("no_drive")
        elif count == 2:
            predicted_label = one_hot_encode("no_entry")
        elif count == 3:
            predicted_label = one_hot_encode("parking")
        elif count == 4:
            predicted_label = one_hot_encode("pedistrain")
        elif count == 5:
            predicted_label = one_hot_encode("road_works")
        elif count == 6:
            predicted_label = one_hot_encode("stop")
        elif count == 7:
            predicted_label = one_hot_encode("way_out")
    else:
        predicted_label = one_hot_encode("none")
    return predicted_label
