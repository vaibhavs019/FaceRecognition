import cv2

# how to open camera using opencv
capt = cv2.VideoCapture(0)
capt2 = cv2.VideoCapture(1)
# capt2 = cv2.VideoCapture(1)

while True:
    isTrue, frame = capt.read()
    isTrue, frame2 = capt2.read()
    # isTrue, frame2 = capt2.read()

    cv2.imshow("video",frame)
    cv2.imshow("video2", frame2)
    # cv2.imshow("webcam", frame2)

    key = cv2.waitKey(1)
    if key == 27:
        break
capt.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
