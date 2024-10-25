import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from datetime import datetime, time





cap = cv.VideoCapture(1)
# WHILE LOOP

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        final_name_list = final_name.tolist()
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)



        # Loop through the cells in the worksheet and update them with the face recognition results
        for row in range(2, 69):
            cell = sheet.cell(row=row, column=2)
            image = cell.value
            image = cv2.imread(image)

            # Predict the person in the image
            person = predict_person(image)

            # Update the cell with the face recognition result
            cell.value = person

        # Save the workbook
        workbook.save(file_name)
        print(final_name)
        print(type(final_name))
        print(final_name_list)
        print(type(final_name_list))

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') ==27:
        break



cap.release()
cv.destroyAllWindows
