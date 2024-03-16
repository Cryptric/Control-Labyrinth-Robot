import time
from multiprocessing import Pipe, Process, Event, Queue
from multiprocessing.connection import Connection
from typing import List

import cv2
import matplotlib
import numpy as np
import serial
from torch import Tensor

import CueNetV2
import Davis346Reader
import Plotter
from MPC import MPC
from Params import *
from utils.ControlUtils import find_center, send_control_signal, calc_speed, gen_reference_path, gen_circ
from utils.FrameUtils import find_board_corners, calc_px2mm, mapping_px2mm, process_frame, check_corner_points

matplotlib.use('TkAgg')

w_circ = gen_circ()


def update(consumer_conn: Connection, frame_buffer: List[Tensor], cue_net: CueNetV2, mpc_x, mpc_y, target_pos_x, target_pos_y, px2mm_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, arduino, termination_event: Event, plot_queue: Queue):
	global w_circ
	if consumer_conn.poll():
		try:
			frame, t = consumer_conn.recv()
			frame, new_frame = process_frame(frame)
			heatmap = cue_net.calc_position_heatmap(new_frame, frame_buffer.pop(0), frame_buffer[0])
			heatmap = np.pad(heatmap[0], ((Y_EDGE // 2, Y_EDGE // 2), (X_EDGE // 2, X_EDGE // 2)))
			x, y = find_center(heatmap)

			frame_buffer.append(new_frame)

			x_mm, y_mm = mapping_px2mm(px2mm_mat, [x, y])
			dt = time.time() - t
			speed_x, speed_y = calc_speed(x_mm, prev_pos_x, dt), calc_speed(y_mm, prev_pos_y, dt)

			xk_x = np.array([x_mm, speed_x])
			xk_y = np.array([y_mm, speed_y])

			# print("position: {}, {}".format(x_mm, y_mm))
			# print("target: {}, {}".format(target_pos_x, target_pos_y))
			# print("velocity: {}, {}".format(speed_x, speed_y))
			w_x = gen_reference_path(x_mm, target_pos_x)
			w_y = gen_reference_path(y_mm, target_pos_y)
			w = np.stack((w_x, w_y), axis=1)
			# print(w_circ)
			# signal_x_rad = mpc_x.get_control_signal(w_x, xk_x)[0]
			# signal_y_rad = mpc_y.get_control_signal(w_y, xk_y)[0]
			signal_x_rad = mpc_x.get_control_signal(w_circ[0:N, 0], xk_x)[0]
			signal_y_rad = mpc_y.get_control_signal(w_circ[0:N, 1], xk_y)[0]

			signal_x_deg = signal_x_rad * 180 / math.pi
			signal_y_deg = signal_y_rad * 180 / math.pi
			send_control_signal(arduino, X_CONTROL_SIGNAL_HORIZONTAL + signal_x_deg, Y_CONTROL_SIGNAL_HORIZONTAL + signal_y_deg)

			plot_queue.put_nowait((frame, heatmap, [x, y], [signal_x_deg, signal_y_deg], t))
			w_circ = np.roll(w_circ, -1, axis=0)
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
	cue_net = CueNetV2.load_cue_net_v2()
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

	# clear pipe
	while consumer_conn.poll():
		consumer_conn.recv()

	while not termination_event.is_set():
		prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y = update(consumer_conn, frame_buffer, cue_net, mpc_x, mpc_y, target_pos_x, target_pos_y, px2mm_mat, prev_pos_x, prev_pos_y, prev_signal_x, prev_signal_y, arduino, termination_event, plot_queue)
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
	p.join()
	plot_process.join()
	plot_queue.close()


if __name__ == "__main__":
	main()
