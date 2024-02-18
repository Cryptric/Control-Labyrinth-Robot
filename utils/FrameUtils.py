import pickle

import cv2

with open("./camera_params.pkl", "rb") as f:
	params = pickle.load(f)

mtx = params['mtx']
dist = params['dist']
newcameramtx = params['newcameramtx']


def remove_distortion(frame):
	return cv2.undistort(frame, mtx, dist, None, newcameramtx)
