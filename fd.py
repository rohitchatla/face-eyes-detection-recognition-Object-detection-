#face & eyes detection/recognition(Object detection)

import numpy as np
import cv2

#haarcascades
# cascades : https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eyes.xml')

capt = cv2.VideoCapture(0)  # 0 for pc,1 for external

while 1:
    ret, fd = capt.read()
    gray = cv2.cvtColor(fd, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(fd, (x, y), (x + w, y + h), (128,128,0), 2) # olive:128,128,0
        r_gray = gray[y:y + h, x:x + w]
        r_color = fd[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(r_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(r_color, (ex, ey), (ex + ew, ey + eh), (147, 112, 219), 2) # mediumpurple:(147,112,219)

    cv2.imshow('Detection', fd)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
capt.release()
cv2.destroyAllWindows()
#use-cases::

# customised haarcascades can be made so any object probably can be detected
# which can be more efficient
# can be used to detect bikes,car number plates by sensing with speed device
# also can be used to make cool social media stickers
#in gaming(game develoment) for AI & ML
# so on.....