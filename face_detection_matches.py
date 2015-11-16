import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
mypath = '.../pictures/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

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
