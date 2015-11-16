import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

def getMatches(img1,img2):
	sift = cv2.SIFT()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
  	return good

def face_detectDB(img1,path):
    onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
    out = (onlyfiles[0],getMatches(img1,cv2.imread(join(path,onlyfiles[0]),0)))
    for i in onlyfiles[1:]:
        atual = getMatches(img1,cv2.imread(join(path,i),0) )
        if len(out[1]) < len(atual):
            out = (i,atual)
    return out
