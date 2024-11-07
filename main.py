from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from twilio.rest import Client
import time


def send_alert(message):
    client = Client('AC557a60f93e8c5e41ef36efb48ec649fb', 'c7cec26c7b44218f58ccc0de12fd77d5')

    message = client.messages.create(
        body=message,
        from_='+14179003949',
        to='+918129233661'
    )


# Set the dimensions for input images
H = 224
W = 224


# Load the trained model
Vehicle_model = tf.keras.models.load_model("Animal_gecwmodel.h5")

# Set up the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.243.118:4747/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Preprocess frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (W, H))
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_frame

    # Perform inference with the loaded model
    predicted_vehicle = Vehicle_model.predict(data)
    print(predicted_vehicle)
    value1 = predicted_vehicle[0,0]
    value2 = predicted_vehicle[0,1]
    value3 = predicted_vehicle[0,2]  
    print(value1,value2,value3)
    
    if value1>0.80:
        animal_type="Elephent"
        send_alert("Urgent: Elephant spotted in the village! Please exercise caution and stay indoors until further notice.")
        time.sleep(20)
    elif value3>0.80:
        animal_type="Tiger"
        send_alert("Urgent: Tiger spotted in the village! Please exercise caution and stay indoors until further notice.")
        time.sleep(20)
    else:
        animal_type="None"
    print(animal_type)

    # print(type(predicted_vehicle))

    # index = np.argmax(predicted_vehicle)
    # if index == 0:
    #     animal_type = "Elephant"
    #     send_alert("Urgent: Elephant spotted in the village! Please exercise caution and stay indoors until further notice.")
    #     time.sleep(20)
    #     break
    # elif index == 1:
    #     animal_type = "None"
    # elif index ==2:
    #     animal_type = "Tiger"
    #     send_alert("Urgent: Tiger spotted in the village! Please exercise caution and stay indoors until further notice.")
    #     time.sleep(20)
    #     break
    # print(animal_type)


    # Draw bounding box or label around the detected object
    cv2.putText(frame, animal_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)
    delay = 50  # Example delay of 50 milliseconds (20 frames per second)
    cv2.waitKey(delay)

    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
