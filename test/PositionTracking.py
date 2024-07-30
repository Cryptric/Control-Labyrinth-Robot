from multiprocessing import Pipe, Event, Process
from multiprocessing.connection import Connection

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import Davis346Reader
from utils.ControlUtils import *
from utils.FrameUtils import *

matplotlib.use("qtAgg")


def track_process(consumer_conn: Connection, termination_event: Event):

	orig_focal_x_pos = np.array([0, 0], dtype=np.float64)
	orig_focal_y_pos = np.array([0, 0], dtype=np.float64)
	calibration_frame = None
	for i in range(50):
		calibration_frame, _ = consumer_conn.recv()
	# calibration_frame = remove_distortion(calibration_frame)
	orig_focal_x_pos[:] = detect_focal_x(calibration_frame)
	orig_focal_y_pos[:] = detect_focal_y(calibration_frame)

	corner_bl = calc_corrected_pos(CORNER_BL, 0, 0)
	corner_br = calc_corrected_pos(CORNER_BR, 0, 0)
	corner_tr = calc_corrected_pos(CORNER_TR, 0, 0)
	corner_tl = calc_corrected_pos(CORNER_TL, 0, 0)
	coordinate_transform_mat = calc_transform_mat([corner_bl, corner_br, corner_tr, corner_tl])
	print(coordinate_transform_mat)

	coord_points = [apply_transform(coordinate_transform_mat, corner_bl), apply_transform(coordinate_transform_mat, corner_br), apply_transform(coordinate_transform_mat, corner_tr), apply_transform(coordinate_transform_mat, corner_tl)]
	target_points = [CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL]
	mm2px_mat = calc_transform_mat(coord_points, np.array(target_points))
	print("mm to px transform matrix")
	print(mm2px_mat)

	prev_angles = np.array([0, 0])

	fig, ax = plt.subplots(ncols=2)
	ax[0].set_xlim(0, IMG_SIZE_X - 1)
	ax[1].set_xlim(0, IMG_SIZE_X - 1)
	ax[0].set_ylim(IMG_SIZE_Y - 1, 0)
	ax[1].set_ylim(IMG_SIZE_Y - 1, 0)
	img_plt = ax[0].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap='gray', vmin=0, vmax=255)
	img_plt_corrected = ax[1].imshow(np.zeros((IMG_SIZE_Y, IMG_SIZE_X)), cmap='gray', vmin=0, vmax=255)

	pos_patch = plt.Circle((0, 0), 6, fill=False, lw=5, color='purple', zorder=101, alpha=0.5, label="ball position")
	pos_plt = ax[0].add_patch(pos_patch)

	focal_plt, = ax[0].plot([orig_focal_x_pos[0], orig_focal_y_pos[0]], [orig_focal_x_pos[1], orig_focal_y_pos[1]], 'bx', label="marker position")
	orig_focal_plt, = ax[0].plot([orig_focal_x_pos[0], orig_focal_y_pos[0]], [orig_focal_x_pos[1], orig_focal_y_pos[1]], 'rx', label="neutral marker position")

	fixed_ball_patch = plt.Circle((0, 0), 6, fill=False, lw=5, color='red', zorder=101, alpha=0.5, label="frozen ball position")
	fixed_ball_plt = ax[0].add_patch(fixed_ball_patch)

	corrected_patch = plt.Circle((0, 0), 6, fill=False, lw=5, color='green', zorder=101, alpha=0.5, label="corrected ball position")
	corrected_patch_plt = ax[0].add_patch(corrected_patch)

	focal_area_x_patch = Rectangle((X_FOCAL_AREA_X_MIN, X_FOCAL_AREA_Y_MIN), X_FOCAL_AREA_X_MAX - X_FOCAL_AREA_X_MIN, X_FOCAL_AREA_Y_MAX - X_FOCAL_AREA_Y_MIN, alpha=0.5, label="X marker search area")
	focal_area_x_plt = ax[0].add_patch(focal_area_x_patch)
	focal_area_y_patch = Rectangle((Y_FOCAL_AREA_X_MIN, Y_FOCAL_AREA_Y_MIN), Y_FOCAL_AREA_X_MAX - Y_FOCAL_AREA_X_MIN, Y_FOCAL_AREA_Y_MAX - Y_FOCAL_AREA_Y_MIN, alpha=0.5, label="Y marker search area")
	focal_area_y_plt = ax[0].add_patch(focal_area_y_patch)

	corner_points_plt = ax[0].scatter([P_CORNER_BR[0], P_CORNER_BL[0], P_CORNER_TL[0], P_CORNER_TR[0]], [P_CORNER_BR[1], P_CORNER_BL[1], P_CORNER_TL[1], P_CORNER_TR[1]], label="detected board corners")
	# corner_points_plt = ax[0].scatter([corner_br[0], corner_bl[0], corner_tl[0], corner_tr[0]], [corner_br[1], corner_bl[1], corner_tl[1], corner_tr[1]], label="detected board corners")
	corner_points_plt1 = ax[1].scatter([CORNER_BR[0], CORNER_BL[0], CORNER_TL[0], CORNER_TR[0]], [CORNER_BR[1], CORNER_BL[1], CORNER_TL[1], CORNER_TR[1]], label="detected board corners")

	ball_pos_plt1 = ax[1].add_patch(plt.Circle((0, 0), 6, fill=False, lw=5, color='green', zorder=101, alpha=0.5, label="corrected ball position"))
	ball_detected_pos_plt1 = ax[1].add_patch(plt.Circle((0, 0), 6, fill=False, lw=5, color='orange', zorder=101, alpha=0.5, label="Detected ball position"))


	def update(_):
		if consumer_conn.poll():
			frame, _ = consumer_conn.recv()
			img_plt.set_array(frame)
			img_plt_corrected.set_array(remove_distortion(frame))

			focal_x_pos = detect_focal_x(frame)
			focal_y_pos = detect_focal_y(frame)
			focal_plt.set_xdata([focal_x_pos[0], focal_y_pos[0]])
			focal_plt.set_ydata([focal_x_pos[1], focal_y_pos[1]])
			try:
				orig_focal_pos = np.array([orig_focal_x_pos[0], orig_focal_y_pos[1]])
				focal_pos = np.array([focal_x_pos[0], focal_y_pos[1]])
				focal_displacement_px = orig_focal_pos - focal_pos
				focal_displacement_px = focal_displacement_px * (-1)
				focal_displacement_px[1] *= -1  # x marker is on negative side (center coordinates), while y marker is on positive side
				focal_displacement_mm = focal_displacement_px * PIXEL_SIZE

				ball_pos_px = find_center4(frame)
				x_mm, y_mm, angle_x, angle_y = detect_ball_pos_mm(frame, orig_focal_pos, coordinate_transform_mat, prev_angles)

				# print(f"Board angle x: {angle_x/np.pi * 180:.2f}°, measured displacement: {focal_displacement_mm[0]}")
				# print(f"Board angle y: {angle_y/np.pi * 180:.2f}°, measured displacement: {focal_displacement_mm[1]}")

				# print(f"Board ball pos: {apply_transform(coordinate_transform_mat, corrected_ball_pos)}, Corrected ball pos: {corrected_ball_pos}, measured pos: {ball_pos}")

				pos_plt.set_center(ball_pos_px)
				corrected_patch_plt.set_center(apply_transform(mm2px_mat, [x_mm, y_mm]))

				ball_pos_plt1.set_center(apply_transform(mm2px_mat, [x_mm, y_mm]))

				# pos = find_center4(frame)
				# pos_obscura = distortion_correct_point(pos[0], pos[1])
				# pos_corrected_df = calc_corrected_pos(pos_obscura, angle_x, angle_y)
				# x_mm, y_mm = apply_transform(coordinate_transform_mat, pos_corrected_df)
				# [x_px, y_px] = apply_transform(mm2px_mat, [x_mm, y_mm])
				# ball_detected_pos_plt1.set_center([x_px, y_px])

			except Exception as e:
				print(f"Error during evaluation: {e}")

		return img_plt, pos_plt, focal_plt, focal_area_x_plt,  focal_area_y_plt, orig_focal_plt, fixed_ball_plt, corrected_patch_plt, corner_points_plt, corner_points_plt1, img_plt_corrected, ball_pos_plt1, ball_detected_pos_plt1

	def onclick(_):
		fixed_ball_plt.set_center(corrected_patch_plt.get_center())

	fig.canvas.mpl_connect('button_press_event', onclick)
	fig.legend()
	anim = FuncAnimation(fig, update, cache_frame_data=False, interval=0, blit=True)
	plt.show()

	termination_event.set()

	t = time.time()
	while time.time() - t < 1:
		if consumer_conn.poll():
			consumer_conn.recv()


def main():
	consumer_conn, producer_conn = Pipe(False)
	termination_event = Event()
	p = Process(target=track_process, args=(consumer_conn, termination_event))
	p.start()

	Davis346Reader.run(producer_conn, termination_event)

	p.join()


if __name__ == '__main__':
	main()
