import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
mypath = '.../pictures/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def pca(X):
  # Principal Component Analysis
  # input: X, matrix with training data as flattened arrays in rows
  # return: projection matrix (with important dimensions first),
  # variance and mean

  #get dimensions
  num_data,dim = X.shape

  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      #print 'PCA - compact trick used'
      M = np.dot(X,X.T) #covariance matrix
      e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
      tmp = np.dot(X.T,EV).T #this is the compact trick
      V = tmp[::-1] #reverse since last eigenvectors are the ones we want
      S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
  else:
      #print 'PCA - SVD used'
      U,S,V = np.linalg.svd(X)
      V = V[:num_data] #only makes sense to return the first num_data

  #return the projection matrix, the variance and the mean
  return V,S,mean_X

def getMatches(img1,img2):
	sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
  	return good

for i in onlyfiles:
	for j in onlyfiles:
		sift = cv2.SIFT()
		img1 = cv2.imread(mypath+i,0)          # queryImage
		img2 = cv2.imread(mypath+j,0) # trainImage
		V,S,img1_me = pca(img1)
		V,S,img2_me = pca(img2)
		print i," ",j," ",len(getMatches(img1,img2))
	
