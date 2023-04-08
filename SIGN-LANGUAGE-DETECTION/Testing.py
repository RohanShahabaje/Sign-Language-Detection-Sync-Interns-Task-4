import cv2
import mediapipe as mp
import pickle
import numpy as np
capture = cv2.VideoCapture(0)

# IMPORTING MY MODEL
my_model_dict = pickle.load(open('./my_model.p','rb'))
model = my_model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A',1:"B",2:"L",3:"OKAY",4:"P",5:"I LOVE YOU"}        
while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = capture.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             # CONVERTING THE IMAGES FROM BGR TO RGB BEACAUSE MATPLOTLIB imshow() process only on the rgb format 
    H, W, _ = frame.shape
    results = hands.process(frame_rgb) 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        for hand_landmarks in results.multi_hand_landmarks:
               for i in range(len(hand_landmarks.landmark)):
                  x =  hand_landmarks.landmark[i].x
                  y =  hand_landmarks.landmark[i].y  
                  data_aux.append(x)  # APPENDING THE X CORDINATES IN THIS 
                  data_aux.append(y)  # APPENDING THE Y CORDINATES IN THIS 

                  x_.append(x)
                  y_.append(y)
        
        x1 =int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
 
        x2 =int(max(x_) * W) - 10
        y2 = int(max(y_) * H)  - 10

        prediction = model.predict([np.asarray(data_aux)])  

        predicted_char = labels_dict[int(prediction[0])]
        print(predicted_char)
   
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
        cv2.putText(frame, predicted_char, (x1, y1  - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
    
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()