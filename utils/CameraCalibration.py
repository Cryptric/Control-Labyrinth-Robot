import pickle

import numpy as np
import cv2
from pyaer import libcaer
from pyaer.davis import DAVIS

from Params import *

IMG_WIDTH = 346
IMG_HEIGHT = 260


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def capture_calibration_frame(rows=15, cols=15):
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("davis346_config.json")

	print("Press 'c' to select a frame for calibration or 'q' to quit")
	corner_list = [None] * 10
	while True:
		try:
			data = device.get_event()
			if data is not None:
				(_, _, _, _, _, frames, _, _) = data
				if frames.shape[0] != 0:
					img = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2RGB)
					ret, corners = cv2.findChessboardCorners(frames[0], (rows, cols), None)
					if ret:
						corners2 = cv2.cornerSubPix(frames[0], corners, (11, 11), (-1, -1), CRITERIA)
						cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
						cv2.imshow("calibration frame", img)
						corner_list.append(corners2)
						corner_list.pop(0)

						if not corner_list[0] is None:
							mtx, dist, newcameramtx = calc_calibration_parameters(frames[0], corner_list, rows=rows, cols=cols)
							# remove distortion
							dst = cv2.undistort(frames[0], mtx, dist, None, newcameramtx)
							dst = cv2.rectangle(dst, (PROCESSING_X, PROCESSING_Y), (PROCESSING_X + PROCESSING_SIZE_WIDTH, PROCESSING_Y + PROCESSING_SIZE_HEIGHT), (255, 0, 0), 1)
							cv2.imshow("preview", dst)
						action = cv2.waitKey(100)
						if action == ord('q'):
							print("Stopping calibration procedure")
							device.shutdown()
							exit(0)
						elif action == ord('c'):
							device.shutdown()
							return frames[0], corner_list
		except KeyboardInterrupt:
			device.shutdown()
			exit(-1)


def calc_calibration_parameters(frame, corners2, rows=15, cols=15):
	objp = np.zeros((rows * cols, 3), np.float32)
	objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * 25
	_, mtx, dist, _, _ = cv2.calibrateCamera([objp for _ in range(len(corners2))], corners2, frame.shape, None, None)
	newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMG_WIDTH, IMG_HEIGHT), 1, (IMG_WIDTH, IMG_HEIGHT))
	return mtx, dist, newcameramtx


def store_parameters(mtx, dist, newcameramtx):
	params = {
		'mtx': mtx,
		'dist': dist,
		'newcameramtx': newcameramtx,
	}
	with open('camera_params.pkl', 'wb') as f:
		pickle.dump(params, f)


def main():
	rows = 10
	cols = 14
	frame, corners2 = capture_calibration_frame(rows=rows, cols=cols)
	mtx, dist, newcameramtx = calc_calibration_parameters(frame, corners2, rows=rows, cols=cols)
	store_parameters(mtx, dist, newcameramtx)


if __name__ == '__main__':
	main()
