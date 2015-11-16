import cv2
import sys
import Image
from PIL import Image as Image2
def crop(img):
    sub_face = img

    faces = face_cascade.detectMultiScale(img, 1.1, 5)
    for (x,y,w,h) in faces:
        sub_face = img[y:y+h, x:x+w]

    return sub_face
#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
def capture_face(qtd):
    video_capture = cv2.VideoCapture(0)
    faces_captured = []
    while len(faces_captured) != qtd:
    # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        for (x, y, w, h) in faces:
    	       img = Image2.fromarray(frame)
    	       #img.crop((x,y,x+w,y+h)).save('teste.png')
               faces_captured.append(img.crop((x,y,x+w,y+h)))
    return faces_captured
# When everything is done, release the capture
