import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    faces_file = 'data/faces.pkl'
    names_file = 'data/names.pkl'

    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)

    with open(names_file, 'rb') as f:
        names = pickle.load(f)

    return faces, names

def train_knn(faces, labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    return knn

if __name__ == "__main__":
    faces, labels = load_data()
    knn = train_knn(faces, labels)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)

    while True:
        ret, fr = cam.read()
        if ret:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            face_coordinates = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face_coordinates:
                face_region = fr[y:y + h, x:x + w, :]
                resized_face = cv2.resize(face_region, (50, 50)).flatten().reshape(1,-1)
                text = knn.predict(resized_face)
                cv2.putText(fr, text[0], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)

            cv2.imshow('face recog', fr)
            if cv2.waitKey(1) == 27:
                break
        else:
            print("Error capturing frame")
            break

    cv2.destroyAllWindows()
    cam.release()
