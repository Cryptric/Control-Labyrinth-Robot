import pickle
import time
from datetime import datetime
from multiprocessing import Pipe, Process, Event, Queue
from multiprocessing.connection import Connection
from typing import List

import numpy as np
import serial
from torch import Tensor

import CueNetV2
import Davis346Reader
import Plotter
from HighLevelController.NextNearestPointController import NextNearestPointController
from MPC import MPC
from Params import *
from utils.ControlUtils import find_center, send_control_signal, calc_speed, gen_reference_path, gen_circ, Timer, \
	print_timers, calc_following_mse, gen_bernoulli_lemniscate, gen_star, calc_control_signal_smoothness_measure, gen_path
from utils.FrameUtils import find_board_corners, calc_px2mm, mapping_px2mm, process_frame, check_corner_points, \
	mapping_mm2px
from utils.store import store

w_circ = gen_circ()

# tuple, of (system state, reference trajectory, predicted trajectory)
recorded_data_x = []
recorded_data_y = []


def update(consumer_conn: Connection, frame_buffer: List[Tensor], cue_net: CueNetV2, mpc_x, mpc_y, path_controller, target_pos_x, target_pos_y, px2mm_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, arduino, termination_event: Event, plot_queue: Queue):
	global w_circ
	global recorded_data_x
	global recorded_data_y
	if consumer_conn.poll():
		with Timer("control loop"):
			try:
				frame, t = consumer_conn.recv()
				frame, new_frame = process_frame(frame)
				heatmap = cue_net.calc_position_heatmap(new_frame, frame_buffer.pop(0), frame_buffer[0])
				heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))
				x, y = find_center(heatmap)

				frame_buffer.append(new_frame)

				x_mm, y_mm = mapping_px2mm(px2mm_mat, [x, y])
				speed_x, speed_y = calc_speed(x_mm, prev_pos_x, dt), calc_speed(y_mm, prev_pos_y, dt)

				xk_x = np.array([x_mm, speed_x])
				xk_y = np.array([y_mm, speed_y])

				# w_x = gen_reference_path(x_mm, target_pos_x)
				# w_y = gen_reference_path(y_mm, target_pos_y)
				# print(w_circ)
				# signal_x_rad = mpc_x.get_control_signal(w_x, xk_x)[0]
				# signal_y_rad = mpc_y.get_control_signal(w_y, xk_y)[0]

				reference_trajectory = path_controller.get_reference_trajectory(x_mm, y_mm)

				# signal_x_rad = mpc_x.get_control_signal(w_circ[0:N, 0], xk_x)
				# signal_y_rad = mpc_y.get_control_signal(w_circ[0:N, 1], xk_y)

				signal_x_rad = mpc_x.get_control_signal(reference_trajectory[:, 0], xk_x)
				signal_y_rad = mpc_y.get_control_signal(reference_trajectory[:, 1], xk_y)

				signal_multiplier = path_controller.get_signal_multiplier()

				signal_x_deg = signal_x_rad[0] * 180 / math.pi * signal_multiplier
				signal_y_deg = signal_y_rad[0] * 180 / math.pi * signal_multiplier
				send_control_signal(arduino, signal_x_deg, signal_y_deg)

				ref_trajectory = np.array([mapping_mm2px(px2mm_mat, reference_trajectory[i]) for i in range(N)])
				predicted_state_x = mpc_x.get_predicted_state(xk_x, signal_x_rad)
				predicted_state_y = mpc_y.get_predicted_state(xk_y, signal_y_rad)

				pred_trajectory = np.array([mapping_mm2px(px2mm_mat, (predicted_state_x[i], predicted_state_y[i])) for i in range(N)])

				plot_queue.put_nowait((frame, heatmap, [x, y], [ref_trajectory[:, 0], ref_trajectory[:, 1]], [pred_trajectory[:, 0], pred_trajectory[:, 1]], [signal_x_deg, signal_y_deg], [speed_x, speed_y], t))
				w_circ = np.roll(w_circ, -1, axis=0)

				# recording
				recorded_data_x.append((xk_x, w_circ[0:N, 0], predicted_state_x, signal_x_rad))
				recorded_data_y.append((xk_y, w_circ[0:N, 1], predicted_state_y, signal_y_rad))

				return x_mm, y_mm, signal_x_deg, signal_y_deg
			except EOFError:
				print("Producer exited")
				print("Shutting down")
				termination_event.set()
	return prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y


def onclick(event, px2mm_mat):
	board_coordinates = mapping_px2mm(px2mm_mat, [event.xdata, event.ydata])
	print("Board coordinates: {}, {}".format(board_coordinates[0], board_coordinates[1]))


def main():
	start_time = datetime.now()

	cue_net = CueNetV2.load_cue_net_v2()
	cue_net.warmup()
	frame_buffer = []

	arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)

	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=Davis346Reader.run, args=(producer_conn, termination_event))
	p.start()
	producer_conn.close()

	# Find board corners for calibration
	while True:
		frame, _ = consumer_conn.recv()
		corner_br, corner_bl, corner_tl, corner_tr = find_board_corners(frame)
		if check_corner_points(corner_br, corner_bl, corner_tl, corner_tr):
			px2mm_mat = calc_px2mm([corner_bl, corner_br, corner_tr, corner_tl])
			print(px2mm_mat)
			break

	for i in range(3):
		frame, _ = consumer_conn.recv()
		_, tensor_frame = process_frame(frame)
		frame_buffer.append(tensor_frame)

	target_pos_queue = Queue()
	target_pos_x = 0
	target_pos_y = 0

	prev_pos_x = 0
	prev_pos_y = 0

	prev_signal_x = 0
	prev_signal_y = 0

	plot_queue = Queue()
	plot_process = Process(target=Plotter.plot, args=(plot_queue, termination_event, target_pos_queue, ["signal x", "signal y"], corner_br, corner_bl, corner_tl, corner_tr))
	plot_process.start()

	mpc_x = MPC(K_x)
	mpc_y = MPC(K_y)

	path_controller = NextNearestPointController()

	# clear pipe
	while consumer_conn.poll():
		consumer_conn.recv()

	runtime_start = time.time()
	while not termination_event.is_set():
		prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y = update(consumer_conn, frame_buffer, cue_net, mpc_x, mpc_y, path_controller, target_pos_x, target_pos_y, px2mm_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, arduino, termination_event, plot_queue)
		if not target_pos_queue.empty():
			x, y = target_pos_queue.get()
			target_pos = mapping_px2mm(px2mm_mat, [x, y])
			target_pos_x, target_pos_y = target_pos[0], target_pos[1]

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


if __name__ == "__main__":
	main()
