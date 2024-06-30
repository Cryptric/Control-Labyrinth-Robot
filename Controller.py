import pickle
import time
from datetime import datetime
from multiprocessing import Pipe, Process, Event, Queue
from multiprocessing.connection import Connection
from typing import List

import numpy as np
import serial

import Davis346Reader
import Plotter
from HighLevelController.NearestPointController import NearestPointController
from HighLevelController.NextNearestPointController import NextNearestPointController
from HighLevelController.PathFindingNearestPointController import PathFindingNearestPointController
from MPC import MPC
from Params import *
from utils.ControlUtils import *
from utils.FrameUtils import *
from utils.store import store
import matplotlib

w_circ = gen_circ()

filter_x = SignalFilter()
filter_y = SignalFilter()

# tuple, of (system state, reference trajectory, predicted trajectory)
recorded_data_x = {"state": [], "delay_compensated_state": [], "target_trajectory": [], "predicted_state": [], "mpc_signal": [], "signal_multiplier": [], "disturbance_compensation": [], "board_angle": []}
recorded_data_y = {"state": [], "delay_compensated_state": [], "target_trajectory": [], "predicted_state": [], "mpc_signal": [], "signal_multiplier": [], "disturbance_compensation": [], "board_angle": []}
prev_angles = np.array([0, 0], dtype=np.float64)


def update(consumer_conn: Connection, frame_buffer, cue_net, mpc_x, mpc_y, path_controller, target_pos_x, target_pos_y, coordinate_transform_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, orig_focal_pos, arduino, termination_event: Event, plot_queue: Queue):
	global w_circ
	global recorded_data_x
	global recorded_data_y
	if consumer_conn.poll():
		with Timer("control loop"):
			try:
				frame, t = consumer_conn.recv()
				# frame = process_frame(frame, equalize=False)
				# torch_frame = frame2torch(frame)
				# heatmap = cue_net.calc_position_heatmap(torch_frame, frame_buffer.pop(0), frame_buffer[0])[0]
				# heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))

				ball_pos = find_center4(frame)
				angle_x, angle_y = calc_board_angle(frame, orig_focal_pos, prev_angles[0], prev_angles[1])
				prev_angles[:] = angle_x, angle_y
				pos = calc_corrected_pos(ball_pos, angle_x, angle_y)

				x_mm, y_mm = apply_transform(coordinate_transform_mat, pos)
				speed_x, speed_y = calc_speed(x_mm, prev_pos_x, dt), calc_speed(y_mm, prev_pos_y, dt)

				xk_x = np.array([x_mm, speed_x])
				xk_y = np.array([y_mm, speed_y])

				target_trajectory = path_controller.get_reference_trajectory(x_mm, y_mm)

				# target_trajectory = w_circ[0:N]

				signal_x_rad, disturbance_compensation_x, xkp_x, deactivate_multiplier_x = mpc_x.get_control_signal(target_trajectory[:, 0], xk_x)
				signal_y_rad, disturbance_compensation_y, xkp_y, deactivate_multiplier_y = mpc_y.get_control_signal(target_trajectory[:, 1], xk_y)

				signal_multiplier = path_controller.get_signal_multiplier(deactivate_multiplier_x or deactivate_multiplier_y)

				predicted_state_x = mpc_x.get_predicted_state(xk_x, signal_x_rad)
				predicted_state_y = mpc_y.get_predicted_state(xk_y, signal_y_rad)

				x_path_diff_norm = np.linalg.norm(target_trajectory[:, 0] - predicted_state_x)
				y_path_diff_norm = np.linalg.norm(target_trajectory[:, 1] - predicted_state_y)
				signal_multiplier_x = 1
				signal_multiplier_y = 1
				if signal_multiplier != 1:
					l = signal_multiplier / np.sqrt(x_path_diff_norm**2 + y_path_diff_norm**2)
					signal_multiplier_x = max(1, l * x_path_diff_norm)
					signal_multiplier_y = max(1, l * y_path_diff_norm)

				# print(f"disturbance correction: x={disturbance_compensation_x:.4f}, y={disturbance_compensation_y:.4f}")

				signal_x_deg = filter_x((signal_x_rad[0] + disturbance_compensation_x) * 180 / np.pi * signal_multiplier_x)
				signal_y_deg = filter_y((signal_y_rad[0] + disturbance_compensation_y) * 180 / np.pi * signal_multiplier_y)
				send_control_signal(arduino, signal_x_deg, signal_y_deg)


				plot_queue.put_nowait((frame, None, [x_mm, y_mm], target_trajectory, [predicted_state_x, predicted_state_y], [signal_x_deg, signal_y_deg], [speed_x, speed_y], t))
				w_circ = np.roll(w_circ, -1, axis=0)

				# recording
				recorded_data_x["state"].append(xk_x)
				recorded_data_x["delay_compensated_state"].append(xkp_x)
				recorded_data_x["target_trajectory"].append(target_trajectory[0:N, 0])
				recorded_data_x["predicted_state"].append(predicted_state_x)
				recorded_data_x["mpc_signal"].append(signal_x_rad)
				recorded_data_x["signal_multiplier"].append(signal_multiplier_x)
				recorded_data_x["disturbance_compensation"].append(disturbance_compensation_x)
				recorded_data_x["board_angle"].append(angle_x)

				recorded_data_y["state"].append(xk_y)
				recorded_data_y["delay_compensated_state"].append(xkp_y)
				recorded_data_y["target_trajectory"].append(target_trajectory[0:N, 1])
				recorded_data_y["predicted_state"].append(predicted_state_y)
				recorded_data_y["mpc_signal"].append(signal_y_rad)
				recorded_data_y["signal_multiplier"].append(signal_multiplier_y)
				recorded_data_y["disturbance_compensation"].append(disturbance_compensation_y)
				recorded_data_y["board_angle"].append(angle_y)

				return x_mm, y_mm, signal_x_deg, signal_y_deg
			except EOFError:
				print("Producer exited")
				print("Shutting down")
				termination_event.set()
	return prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y


def onclick(event, px2mm_mat):
	board_coordinates = apply_transform(px2mm_mat, [event.xdata, event.ydata])
	print("Board coordinates: {}, {}".format(board_coordinates[0], board_coordinates[1]))


def main():
	start_time = datetime.now()

	# cue_net = CueNetV2.load_cue_net_v2()
	# cue_net.warmup()
	frame_buffer = []

	arduino = serial.Serial('/dev/ttyUSB1', 115200, timeout=5)

	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()


	orig_focal_x_pos = np.array([0, 0], dtype=np.float64)
	orig_focal_y_pos = np.array([0, 0], dtype=np.float64)
	calibration_frame = None
	for i in range(50):
		calibration_frame, _ = consumer_conn.recv()
	orig_focal_x_pos[:] = detect_focal_x(calibration_frame)
	orig_focal_y_pos[:] = detect_focal_y(calibration_frame)
	orig_focal_pos = np.array([orig_focal_x_pos[0], orig_focal_y_pos[1]])

	coordinate_transform_mat, mm2px_mat = get_transform_matrices()

	frame = None
	for i in range(10):
		frame, _ = consumer_conn.recv()
		# tensor_frame = frame2torch(process_frame(frame))
		# frame_buffer.append(tensor_frame)

	target_pos_queue = Queue()
	target_pos_x = 0
	target_pos_y = 0

	ball_pos = find_center4(frame)
	[prev_pos_x, prev_pos_y] = apply_transform(coordinate_transform_mat, calc_corrected_pos(ball_pos, 0, 0))

	prev_signal_x = 0
	prev_signal_y = 0

	plot_queue = Queue()
	plot_process = Process(target=Plotter.plot, args=(plot_queue, termination_event, target_pos_queue, ["signal x", "signal y"], CORNER_BR, CORNER_BL, CORNER_TL, CORNER_TR))
	plot_process.start()

	mpc_x = MPC(K_x)
	mpc_y = MPC(K_y)

	path_controller = NearestPointController(gen_path_simple_labyrinth())

	# clear pipe
	while consumer_conn.poll():
		consumer_conn.recv()

	runtime_start = time.time()
	while not termination_event.is_set():
		prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y = update(consumer_conn, frame_buffer, None, mpc_x, mpc_y, path_controller, target_pos_x, target_pos_y, coordinate_transform_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, orig_focal_pos, arduino, termination_event, plot_queue)

	# make sure pipe isn't full, such that producer can exit
	while p.is_alive():
		try:
			if consumer_conn.poll():
				consumer_conn.recv()
		except EOFError:
			break
	consumer_conn.close()

	print_timers()

	p.join()
	plot_process.join()
	plot_queue.close()

	with open("recorded_x.pkl", 'wb') as f:
		pickle.dump(recorded_data_x, f, pickle.HIGHEST_PROTOCOL)

	with open("recorded_y.pkl", 'wb') as f:
		pickle.dump(recorded_data_y, f, pickle.HIGHEST_PROTOCOL)

	follow_mse = calc_following_mse(recorded_data_x) + calc_following_mse(recorded_data_y)
	store(recorded_data_x, recorded_data_y, start_time, time.time() - runtime_start, follow_mse)
	print(f"Following MSE: {follow_mse}")
	print(f"Control signal smoothness: {calc_control_signal_smoothness_measure(recorded_data_x) + calc_control_signal_smoothness_measure(recorded_data_y)}")
	print(f"Camera process: {p.is_alive()}")
	print(f"Plot process: {plot_process.is_alive()}")
	exit(0)


if __name__ == "__main__":
	main()
