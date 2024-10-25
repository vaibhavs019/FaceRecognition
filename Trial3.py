import openpyxl
from datetime import datetime
import cv2
import numpy as np
import os
from keras_facenet import FaceNet
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

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_5classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

capt = cv2.VideoCapture(1)

while True:
    isTrue, frame = capt.read()
    # isTrue, frame2 = capt2.read()

    cv2.imshow("video",frame)
    # cv2.imshow("webcam", frame2)

# Function to predict the person in the image
def predict_person(image):
    image = cv2.resize(image, (160, 160))
    image = np.expand_dims(image, axis=0)
    ypred = facenet.embeddings(image)
    face_name = encoder.inverse_transform(ypred)[0]
    return face_name

# Loop through the cells in the worksheet and update them with the face recognition results
for row in range(2, 69):
    cell = sheet.cell(row=row, column=2)
    image = cell.value
    image = cv2.imread(frame)

    # Predict the person in the image
    person = predict_person(frame)

    # Update the cell with the face recognition result
    cell.value = person
    if cv2.waitKey(20) & 0xFF==("q"):
        break
capt.release()
cv2.destroyAllWindows()

cv2.waitKey(0)

# Save the workbook
workbook.save(file_name)


