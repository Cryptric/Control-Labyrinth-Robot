import pickle

import numpy as np
import cv2
from pyaer import libcaer
from pyaer.davis import DAVIS


IMG_WIDTH = 346
IMG_HEIGHT = 260


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def capture_calibration_frame(rows=15, cols=15):
	device = DAVIS(noise_filter=True)
	device.start_data_stream()
	device.set_bias_from_json("./../davis346_config.json")

	print("Press 'c' to select a frame for calibration or 'q' to quit")
	while True:
		try:
			data = device.get_event()
			if data is not None:
				(_, _, _, _, _, frames, _, _) = data
				if frames.shape[0] != 0:
					img = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2RGB)
					ret, corners = cv2.findChessboardCorners(frames[0], (15, 15), None)
					if ret:
						corners2 = cv2.cornerSubPix(frames[0], corners, (11, 11), (-1, -1), CRITERIA)
						cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
						cv2.imshow("calibration frame", img)
						action = cv2.waitKey(1)
						if action == ord('q'):
							print("Stopping calibration procedure")
							device.shutdown()
							exit(0)
						elif action == ord('c'):
							device.shutdown()
							return frames[0], corners2
		except KeyboardInterrupt:
			device.shutdown()
			exit(-1)


def calc_calibration_parameters(frame, corners2, rows=15, cols=15):
	objp = np.zeros((rows * cols, 3), np.float32)
	objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * 25
	_, mtx, dist, _, _ = cv2.calibrateCamera([objp], [corners2], frame.shape, None, None)
	newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMG_WIDTH, IMG_HEIGHT), 1, (IMG_WIDTH, IMG_HEIGHT))

	params = {
		'mtx': mtx,
		'dist': dist,
		'newcameramtx': newcameramtx,
	}
	with open('./../camera_params.pkl', 'wb') as f:
		pickle.dump(params, f)

	# remove distortion
	dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
	cv2.imshow("result", dst)
	cv2.waitKey(5000)


def main():
	frame, corners2 = capture_calibration_frame()
	calc_calibration_parameters(frame, corners2)


if __name__ == '__main__':
	main()
