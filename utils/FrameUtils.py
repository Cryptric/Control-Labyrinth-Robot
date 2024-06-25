import pickle

import cv2
import numpy
import numpy as np

from Params import *

with open("./camera_params.pkl", "rb") as f:
	params = pickle.load(f)

mtx = params['mtx']
dist = params['dist']
newcameramtx = params['newcameramtx']


def calc_distance(x, y):
	return np.linalg.norm(x - y)


def remove_distortion(frame):
	return cv2.undistort(frame, mtx, dist, None, newcameramtx)


def process_frame(frame, gain=1, bias=0, equalize=True):
	frame = remove_distortion(frame)
	if equalize:
		frame = cv2.equalizeHist(frame)
	frame = np.clip(frame.astype(np.uint16) * gain + bias, 0, 255).astype(np.uint8)
	return frame


def find_corners(frame, block_size=2, ksize=5, k=0.02, threshold=0.0005):
	frame = process_frame(frame)
	frame = frame.astype(np.uint8)

	mask = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
	mask[CORNER_MASK_MIN_Y:CORNER_MASK_MAX_Y, CORNER_MASK_MIN_X:CORNER_MASK_MAX_X] = 1

	dst = cv2.cornerHarris(frame, block_size, ksize, k)
	dst = cv2.dilate(dst, None)

	dst[mask == 0] = 0
	corners = np.transpose(np.nonzero(dst > threshold * dst.max()))
	return corners


def find_board_corners(frame):
	corners = find_corners(frame)

	corner_br_idx = np.argmax(corners[:, 0] * corners[:, 1])
	corner_bl_idx = np.argmax(corners[:, 0] * (IMG_SIZE_X - corners[:, 1]))
	corner_tl_idx = np.argmax((IMG_SIZE_Y - corners[:, 0]) * (IMG_SIZE_X - corners[:, 1]))
	corner_tr_idx = np.argmax((IMG_SIZE_Y - corners[:, 0]) * corners[:, 1])

	corner_br = np.flip(corners[corner_br_idx])
	corner_bl = np.flip(corners[corner_bl_idx])
	corner_tl = np.flip(corners[corner_tl_idx])
	corner_tr = np.flip(corners[corner_tr_idx])

	return corner_br, corner_bl, corner_tl, corner_tr


def calc_mm2px_mat(coordinate_transform_mat, corner_bl, corner_br, corner_tr, corner_tl):
	coord_points = [apply_transform(coordinate_transform_mat, corner_bl), apply_transform(coordinate_transform_mat, corner_br), apply_transform(coordinate_transform_mat, corner_tr), apply_transform(coordinate_transform_mat, corner_tl)]
	target_points = [CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL]
	mm2px_mat = calc_transform_mat(coord_points, np.array(target_points))
	return mm2px_mat


def calc_transform_mat(ps, target_matrix=None):
	assert len(ps) == 4
	A = np.empty((0, 8), float)
	corners = target_matrix if target_matrix is not None else np.array([[0, 0], [BOARD_LENGTH_X, 0], [BOARD_LENGTH_X, BOARD_LENGTH_Y], [0, BOARD_LENGTH_Y]])
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


def apply_transform(homo, uv):
	vec = np.array([[uv[0]], [uv[1]], [1]])
	xy = np.matmul(homo, vec)
	result = xy[0][0] / xy[2][0], xy[1][0] / xy[2][0]
	return result


def apply_inverse_transform(homo, xy):
	x = xy[0]
	y = xy[1]
	vec = np.array([[homo[0][2]-x], [homo[1][2]-y]])
	m = np.array([[homo[2][0]*x-homo[0][0], homo[2][1]*x-homo[0][1]], [homo[2][0]*y-homo[1][0], homo[2][1]*y-homo[1][1]]])
	rm = np.linalg.inv(m)
	uv = np.matmul(rm, vec)
	return [uv[0][0], uv[1][0]]


def sequence_apply_transform(homo, xs, ys):
	ones = np.ones_like(xs)
	vecs = np.vstack([xs, ys, ones])
	coords = homo @ vecs
	res = coords[0:2] / coords[2]  # divide every column with the last row entry
	return res


def sequence_apply_inverse_transform(homo, xs, ys):
	res = np.zeros((xs.shape[0], 2))
	for i in range(xs.shape[0]):
		res[i] = apply_inverse_transform(homo, [xs[i], ys[i]])
	return res.T


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
	angle_check = abs(np.pi / 2 - angle_br) < CORNER_ANGLE_MAX_DEVIATION and abs(np.pi / 2 - angle_bl) < CORNER_ANGLE_MAX_DEVIATION and abs(np.pi / 2 - angle_tl) < CORNER_ANGLE_MAX_DEVIATION and abs(np.pi / 2 - angle_tr) < CORNER_ANGLE_MAX_DEVIATION
	distance_x = abs(calc_distance(corner_bl, corner_br) + calc_distance(corner_tl, corner_tr)) / 2
	distance_y = abs(calc_distance(corner_bl, corner_tl) + calc_distance(corner_br, corner_tr)) / 2
	length_check = distance_x * 1.5 < BOARD_LENGTH_X < distance_x * 1.8 and distance_y * 1.5 < BOARD_LENGTH_Y < distance_y * 1.8
	length_check = True # TODO
	return angle_check and length_check


params = cv2.SimpleBlobDetector_Params()
params.minArea = 1
params.minThreshold = 30
params.maxThreshold = 130
params.minInertiaRatio = 0.3
params.maxInertiaRatio = 3
params.minDistBetweenBlobs = 3
detector = cv2.SimpleBlobDetector_create(params)
def detect_focal(frame, scale=5):
	match_area = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
	key_points = detector.detect(match_area)
	if len(key_points) == 0:
		print("Warning no angle keypoint detected")
		return np.array([0, 0])
	if len(key_points) > 1:
		print(f"Warning multiple angle keypoint detected ({len(key_points)})")
	key_point = key_points[0]
	for i in range(len(key_points)):
		if abs(25 - key_points[i].size) < abs(25 - key_point.size):
			key_point = key_points[i]
	# if len(key_points) > 1 or len(key_points) == 0:
	# 	print(f"WARNING invalid key points length: {len(key_points)}: 1. ({key_points[0].pt[0] / scale}, {key_points[0].pt[1] / scale}), 2. ({key_points[1].pt[0] / scale}, {key_points[1].pt[1] / scale})")
	x = key_point.pt[0] / scale
	y = key_point.pt[1] / scale
	return np.array([x, y])


def detect_focal_x(frame):
	match_area = frame[X_FOCAL_AREA_Y_MIN:X_FOCAL_AREA_Y_MAX, X_FOCAL_AREA_X_MIN:X_FOCAL_AREA_X_MAX]
	[x, y] = detect_focal(match_area)
	if x == 0 and y == 0:
		return np.array([0, 0])
	return np.array([x + X_FOCAL_AREA_X_MIN, y + X_FOCAL_AREA_Y_MIN])


def detect_focal_y(frame):
	match_area = frame[Y_FOCAL_AREA_Y_MIN:Y_FOCAL_AREA_Y_MAX, Y_FOCAL_AREA_X_MIN:Y_FOCAL_AREA_X_MAX]
	[x, y] = detect_focal(match_area)
	if x == 0 and y == 0:
		return np.array([0, 0])
	return np.array([x + Y_FOCAL_AREA_X_MIN, y + Y_FOCAL_AREA_Y_MIN])



def detect_focal2(frame, scale=5):
	match_area = frame[X_FOCAL_AREA_Y_MIN:X_FOCAL_AREA_Y_MAX, X_FOCAL_AREA_X_MIN:X_FOCAL_AREA_X_MAX]
	match_area = cv2.resize(match_area, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
	p_size = 4 * scale
	p_size2 = p_size // 2
	pattern = np.ones((p_size, p_size), dtype=np.uint8)
	pattern[p_size2 - 1:p_size2 + 1, p_size2 - 1:p_size2 + 1] = 0
	res = cv2.matchTemplate(match_area, pattern, cv2.TM_CCOEFF_NORMED)
	pos = np.unravel_index(res.argmax(), res.shape)
	return numpy.array([pos[1] / scale + X_FOCAL_AREA_X_MIN, pos[0] / scale + X_FOCAL_AREA_Y_MIN])
