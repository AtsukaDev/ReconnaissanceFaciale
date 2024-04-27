import cv2
import urllib3 as urllib
import numpy as np

# On accède à la webcam
cam = cv2.VideoCapture(0)

# Paramètres pour définir la fenêtre
nom_fenetre = "video_cam"

largeur_image = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur_image = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow(nom_fenetre, cv2.WND_PROP_FULLSCREEN)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Tout le temps
while True:
    # On prend une photo
    ret, image = cam.read()
    if ret:

        # On affiche la fenêtre avec notre image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.imshow(nom_fenetre, image)

        # On ajoute un temps d'attente
        # et on arrête la boucle si on appuie sur la touche 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
