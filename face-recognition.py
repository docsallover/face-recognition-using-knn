import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)

with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

facec = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

print('Shape of Faces matrix --> ', faces.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces,labels)

# Define KNN functions


# def distance(x1, x2):
#     d = np.sqrt(((x1 - x2) ** 2).sum())
#     return d


# def knn(xt, X_train=faces, y_train=labels, k=5):
#     vals = []

#     for ix in range(len(labels)):
#         d = distance(X_train[ix], xt)
#         vals.append([d, y_train[ix]])

#     sorted_labels = sorted(vals, key=lambda z: z[0])
#     neighbours = np.asarray(sorted_labels)[:k, -1]

#     freq = np.unique(neighbours, return_counts=True)

#     # freq[0] is list of names and freq[1] is list of counts
#     return freq[0][freq[1].argmax()]

cam = cv2.VideoCapture(0)

while True:
    ret, fr = cam.read()
    if ret == True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        face_coordinates = facec.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_coordinates:
            fc = fr[y:y + h, x:x + w, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
            text = knn.predict(r)
            cv2.putText(fr, text[0], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)

        cv2.imshow('face recog', fr)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
