
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing  import image

model = load_model('treino_ml/converted_keras (1)/keras_model.h5')

cap = cv2.VideoCapture(0)
class_labels=['SP' , 'BH']

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (244,244))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor/=255.

    prediction = model.predict(img_tensor)

    class_index = np.argmax(prediction[0])

    class_label = class_labels[class_index]
    print(class_label)

    cv2.putText(frame, class_label, (10,30), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255,255,255),3)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 