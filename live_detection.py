from tensorflow import keras
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import time

MODEL_PATH = 'Model/traffic_sign_model.h5'

loaded_model = keras.models.load_model(MODEL_PATH)

def returnRedness(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return v

def returnBlueness(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return u

def threshold(img, T=150):
    _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return img 

def findContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def findBiggestContour(contours):
    m = 0
    c = [cv2.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

def boundaryBox(img, contours):
    x, y, w, h = cv2.boundingRect(contours)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    sign = img[y:(y + h), x:(x + w)]
    return img, sign


# Constants for resizing
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3

def preprocess_image_for_prediction(image, img_height=30, img_width=30):
    try:
        # Ensure the image has the correct shape and color channels
        image = cv2.resize(image, (img_height, img_width))
        image_fromarray = Image.fromarray(image, 'RGB')
        resized_image = np.array(image_fromarray)

        # Normalize the image to the range [0, 1]
        resized_image = resized_image / 255.0

        # Expand the dimensions to match the model's input shape
        preprocessed_image = np.expand_dims(resized_image, axis=0)

        return preprocessed_image

    except Exception as e:
        print("Error during image preprocessing:", str(e))
        return None
    
def predict(image):
    predictions = loaded_model.predict(preprocess_image_for_prediction(image))
    return np.argmax(predictions)


# Label Overview
label_to_text = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


#desired_fps = 5  # Adjust this value to your desired FPS
#delay_time = 1.0 / desired_fps

cap = cv2.VideoCapture(0)

while True:

    # start_time = time.time()

    _, frame = cap.read()
    redness = returnRedness(frame)
    blueness = returnBlueness(frame)
    
    red_thresh = threshold(redness)    
    blue_thresh = threshold(blueness)

    try:
        red_contours = findContour(red_thresh)
        blue_contours = findContour(blue_thresh)     
        
        if red_contours:
            red_big = findBiggestContour(red_contours)
            if cv2.contourArea(red_big) > 3000:
                print(cv2.contourArea(red_big))
                img, sign = boundaryBox(frame, red_big)
                label = label_to_text[predict(sign)]
                print("Now, I see:", label_to_text[predict(sign)])
        if blue_contours:
            blue_big = findBiggestContour(blue_contours)
            if cv2.contourArea(blue_big) > 3000:
                img, sign = boundaryBox(frame, blue_big)
                label = label_to_text[predict(sign)]
                print(f"Now, I see: {label}")
        else:
            cv2.imshow('frame', frame)
    
    except:
        cv2.imshow('frame', frame)
    
    # elapsed_time = time.time() - start_time
    # remaining_time = max(0, delay_time-elapsed_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
