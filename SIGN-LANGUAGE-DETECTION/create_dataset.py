import mediapipe as mp
import os
import cv2
import matplotlib.pyplot as plt
import pickle

# FOR HANDS DETECTION BY MEDIAPIPE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)



DATA_DIR = './data'

data = []  # image data
labels = [] # category for each one of our image


# Iterating in our data directory

for dir_ in os.listdir(DATA_DIR):             # FULL OUTER DIRECTORY "data"
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # SUDIRECTORY "0", "1",etc
       data_aux = []
       img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))   #IMG IN OUR SUDIRECTORY "0.jpg","1.jpg",etc
       img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # CONVERTING THE IMAGES FROM BGR TO RGB BEACAUSE MATPLOTLIB imshow() process only on the rgb format 

       results = hands.process(img_rgb)
       if results.multi_hand_landmarks:
           for hand_landmarks in results.multi_hand_landmarks:
               for i in range(len(hand_landmarks.landmark)):
                  x =  hand_landmarks.landmark[i].x
                  y =  hand_landmarks.landmark[i].y  
                  data_aux.append(x)  # APPENDING THE X CORDINATES IN THIS 
                  data_aux.append(y)  # APPENDING THE Y CORDINATES IN THIS 

           data.append(data_aux)
           labels.append(dir_) # category "0", "1", etc...  

f = open('data.pickle','wb') #  To save data  
pickle.dump({'data':data,'labels':labels}, f)
f.close

    # MY  LANDMARKS DRAWINGS
    # mp_drawing.draw_landmarks(
    # img_rgb,
    # hand_landmarks,
    # mp_hands.HAND_CONNECTIONS,
    # mp_drawing_styles.get_default_hand_landmarks_style(),
    # mp_drawing_styles.get_default_hand_connections_style())
           
       
           

#        plt.figure() 
#        plt.imshow(img_rgb) 
# plt.show()

'''
DATA_DIR = './data': This sets a variable DATA_DIR to the string './data', which is likely the directory where the image files are stored.
for dir_ in os.listdir(DATA_DIR):: This loops through each file or directory in the DATA_DIR directory and sets the variable dir_ to the name of the current file or directory.
for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:: This loops through each image file in the current directory (dir_). The os.path.join() function is used to concatenate the DATA_DIR directory with the dir_ directory to create the full path to the image file. The [:1] slice limits the loop to only the first image file in the directory.
img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)): This reads the current image file using the OpenCV library's imread() function and sets the variable img to the resulting image data.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB): This converts the image data from OpenCV's default BGR color format to RGB format, which is the format expected by matplotlib's imshow() function.
plt.figure(): This creates a new figure for the image to be displayed on.
plt.imshow(img_rgb): This displays the image using the matplotlib library's imshow() function.
plt.show(): This displays the current figure. Since this is inside the inner loop, a new figure will be created and displayed for each image file.

cv2.imread() is a function from the OpenCV (Open Source Computer Vision) library used to read an image from a file.

os.path.join() is a function from the built-in os module that joins one or more path components (in this case, DATA_DIR, dir_, and img_path) together into a single path.

os.listdir() is a function from the os module that lists the contents of a directory. It returns a list of filenames in the directory.

Therefore, os.path.join(DATA_DIR, dir_, img_path) concatenates the DATA_DIR directory path, the dir_ subdirectory path, and the img_path filename together to create the full path to the image file.

cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) reads the image data from the file at the specified path using the cv2.imread() function and sets it to the variable img. This variable now holds a NumPy array representing the image data, which can be processed or displayed further as needed.
'''


