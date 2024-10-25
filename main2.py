# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from datetime import datetime, time
from Excel_Data import workbook
import openpyxl

today_date = datetime.now().strftime("%d-%m-%Y")
file_name = f"{today_date}.xlsx"
workbook = openpyxl.load_workbook(file_name)
sheet = workbook.active

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_14classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("logistic_regression_160x160.pkl", 'rb'))

cap = cv.VideoCapture(0)

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        final_name_list = final_name.tolist()
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)

        now = datetime.now().time()
        face_locations, face_names = ypred, final_name

        for face_loc, name in zip(face_locations, face_names):
            if name not in detected_faces:
                detected_faces[name] = False

            if not detected_faces[name]:
                detected_faces[name] = True
                now = datetime.now().time()
                if time(8, 45, 0) <= now <= time(9, 45, 0):
                    if final_name == "vaibhav":
                        sheet["D54"] = "P"
                    if final_name == "Dev":
                        sheet["D19"] = "P"
                    if final_name == "Shourya":
                        sheet["D49"] = "P"
                    if final_name == "Manan":
                        sheet["D39"] = "P"
                if time(14, 45, 0) <= now <= time(15, 55, 0):
                    if final_name == "vaibhav":
                        sheet["E54"] = "P"
                    if final_name == "Dev":
                        sheet["E19"] = "P"
                    if final_name == "Shourya":
                        sheet["E49"] = "P"
                    if final_name == "Manan":
                        sheet["E39"] = "P"
        print(final_name)
        print(type(final_name))
        print(final_name_list)
        print(type(final_name_list))
    cv.imshow("Face Recognition:", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

file_name = f"{today_date}.xlsx"
workbook.save(file_name)


cap.release()
cv.destroyAllWindows