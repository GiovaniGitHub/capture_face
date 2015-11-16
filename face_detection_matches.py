import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from capture_face import *
from utils import *
from getMatches import *
mypath = 'pictures/'
qtd = 3

list_imgs = [PIL2array(i) for i in capture_face(qtd)]
dict_faces = {}
for i,x in enumerate(list_imgs):
    dict_faces.update({i:face_detectDB(x,mypath)})

'''
for i in onlyfiles:
	for j in onlyfiles:
		sift = cv2.SIFT()
		img1 = cv2.imread(mypath+i,0)          # queryImage
		img2 = cv2.imread(mypath+j,0) # trainImage
		#V,S,img1_me = pca(img1)
		#V,S,img2_me = pca(img2)
		print i," ",j," ",len(getMatches(img1,img2))
<<<<<<< HEAD
=======

>>>>>>> a08189dba778dfd2372b17d7f6aa43c6b557a5ab
'''
