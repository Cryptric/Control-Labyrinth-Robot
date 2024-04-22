import pickle

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from CueNetV2 import device
from Params import *

with open("./camera_params.pkl", "rb") as f:
	params = pickle.load(f)

mtx = params['mtx']
dist = params['dist']
newcameramtx = params['newcameramtx']


def remove_distortion(frame):
	return cv2.undistort(frame, mtx, dist, None, newcameramtx)


def find_board_corners(frame):
	frame, _ = process_frame(frame)
	frame = frame.astype(np.uint8)

	mask = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
	mask[CORNER_MASK_MIN_Y:CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X:CORNER_MASK_MAX_X] = 1

	dst = cv2.cornerHarris(frame, 2, 5, 0.04)
	dst = cv2.dilate(dst, None)

	dst[mask == 0] = 0
	corners = np.transpose(np.nonzero(dst > 0.1 * dst.max()))

	corner_br_idx = np.argmax(corners[:, 0] * corners[:, 1])
	corner_bl_idx = np.argmax(corners[:, 0] * (IMG_SIZE_X - corners[:, 1]))
	corner_tl_idx = np.argmax((IMG_SIZE_Y - corners[:, 0]) * (IMG_SIZE_X - corners[:, 1]))
	corner_tr_idx = np.argmax((IMG_SIZE_Y - corners[:, 0]) * corners[:, 1])

	corner_br = np.flip(corners[corner_br_idx])
	corner_bl = np.flip(corners[corner_bl_idx])
	corner_tl = np.flip(corners[corner_tl_idx])
	corner_tr = np.flip(corners[corner_tr_idx])

	return corner_br, corner_bl, corner_tl, corner_tr


def calc_px2mm(ps):
	assert len(ps) == 4
	A = np.empty((0, 8), float)
	corners = np.array([[0, 0], [310, 0], [310, 265], [0, 265]])
	b = corners.reshape(8, )
	for i in range(4):
		u = ps[i][0]
		v = ps[i][1]
		x = corners[i][0]
		y = corners[i][1]
		r1 = np.array([[u, v, 1, 0, 0, 0, -u * x, -v * x]])
		r2 = np.array([[0, 0, 0, u, v, 1, -u * y, -v * y]])
		A = np.append(A, r1, axis=0)
		A = np.append(A, r2, axis=0)
	hs = np.linalg.solve(A, b)
	homo = np.array([[hs[0], hs[1], hs[2]], [hs[3], hs[4], hs[5]], [hs[6], hs[7], 1]])
	return homo


def mapping_px2mm(homo, uv):
	vec = np.array([[uv[0]], [uv[1]], [1]])
	xy = np.matmul(homo, vec)
	result = xy[0][0] / xy[2][0], xy[1][0] / xy[2][0]
	return result


def mapping_mm2px(homo, xy):
	x = xy[0]
	y = xy[1]
	vec = np.array([[homo[0][2]-x], [homo[1][2]-y]])
	m = np.array([[homo[2][0]*x-homo[0][0], homo[2][1]*x-homo[0][1]], [homo[2][0]*y-homo[1][0], homo[2][1]*y-homo[1][1]]])
	rm = np.linalg.inv(m)
	uv = np.matmul(rm, vec)
	return [uv[0][0], uv[1][0]]


def process_frame(frame):
	frame = remove_distortion(frame)
	# values for 2000 exposure
	frame = np.clip(frame.astype(np.uint16) * 3.7 + 110, 0, 255)
	return frame, torch.squeeze(TF.to_tensor(frame[PROCESSING_Y:PROCESSING_Y + PROCESSING_SIZE_HEIGHT, PROCESSING_X:PROCESSING_X + PROCESSING_SIZE_WIDTH].astype("float32") / 255).to(device))


def calc_angle(a, b, c):
	ba = a - b
	bc = c - b
	ba_n = ba / np.linalg.norm(ba)
	bc_n = bc / np.linalg.norm(bc)
	return np.arccos(np.clip(np.dot(ba_n, bc_n), -1.0, 1.0))


def check_corner_points(corner_br, corner_bl, corner_tl, corner_tr):
	angle_br = calc_angle(corner_bl, corner_br, corner_tr)
	angle_bl = calc_angle(corner_tl, corner_bl, corner_br)
	angle_tl = calc_angle(corner_tr, corner_tl, corner_bl)
	angle_tr = calc_angle(corner_br, corner_tr, corner_tl)
	return abs(math.pi / 2 - angle_br) < CORNER_ANGLE_MAX_DEVIATION and abs(math.pi / 2 - angle_bl) < CORNER_ANGLE_MAX_DEVIATION and abs(math.pi / 2 - angle_tl) < CORNER_ANGLE_MAX_DEVIATION and abs(math.pi / 2 - angle_tr) < CORNER_ANGLE_MAX_DEVIATION
