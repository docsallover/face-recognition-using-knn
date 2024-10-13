import cv2
import numpy as np
import os
import pickle

face_data = []
i = 0

cam = cv2.VideoCapture(0)

facec = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input('Enter your name --> ')

ret = True

while(ret):
    ret, frame = cam.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = facec.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in face_coordinates:
            faces = frame[y:y+h, x:x+w, :]
            resized_faces = cv2.resize(faces, (50, 50))

            if i % 10 == 0 and len(face_data) < 10:
                face_data.append(resized_faces)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        i += 1

        cv2.imshow('frames', frame)

        if cv2.waitKey(1) == 27 or len(face_data) >= 10:
            break
    else:
        print('error')
        break


cv2.destroyAllWindows()
cam.release()


face_data = np.asarray(face_data)
face_data = face_data.reshape(10, -1)

if 'names.pkl' not in os.listdir('data/'):
    names = [name]*10
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    names = names + [name]*10
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)


if 'faces.pkl' not in os.listdir('data/'):
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open('data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)

    faces = np.append(faces, face_data, axis=0)
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)
