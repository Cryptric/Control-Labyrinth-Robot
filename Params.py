import numpy as np

IMG_SIZE_X = 346
IMG_SIZE_Y = 260

PIXEL_SIZE = 18.5 / 1000

PROCESSING_SIZE_WIDTH = 240
PROCESSING_SIZE_HEIGHT = 180

X_EDGE = IMG_SIZE_X - PROCESSING_SIZE_WIDTH
Y_EDGE = IMG_SIZE_Y - PROCESSING_SIZE_HEIGHT

PROCESSING_X = X_EDGE // 2
PROCESSING_Y = Y_EDGE // 2


X_CONTROL_SIGNAL_HORIZONTAL = 83
Y_CONTROL_SIGNAL_HORIZONTAL = 87

BOARD_LENGTH_X = 275
BOARD_LENGTH_Y = 230

CORNER_MASK_MIN_X, CORNER_MASK_MAX_X = 32, 320
CORNER_MASK_MIN_Y, CORNER_MASK_MAX_Y = 5, 250
CORNER_ANGLE_MAX_DEVIATION = 0.1
LENGTH_MAX_DEVIATION = 0.1

# Platform corners image coordinates from distortion free image
CORNER_BL = (42, 238)
CORNER_BR = (306, 236)
CORNER_TR = (305, 18)
CORNER_TL = (41, 17)

# Perspective corners image coordinates from unaltered camera image
P_CORNER_BL = (35, 242)
P_CORNER_BR = (312, 243)
P_CORNER_TR = (311, 9)
P_CORNER_TL = (35, 11)

ZERO_POSITION = np.array([41, 237])


X_FOCAL_AREA_X_MIN, X_FOCAL_AREA_X_MAX = 10, 40
X_FOCAL_AREA_Y_MIN, X_FOCAL_AREA_Y_MAX = 120, 135

Y_FOCAL_AREA_X_MIN, Y_FOCAL_AREA_X_MAX = 160, 180
Y_FOCAL_AREA_Y_MIN, Y_FOCAL_AREA_Y_MAX = 246, 260

rx = -140
ry = -110
d = 410
s = 8.2

subpixel_neighborhood_size_2 = 10

K_x = -0.18203246753246752
K_y = -0.1468831168831169

SERVO_MIN_PULSE_WIDTH = 544
SERVO_MAX_PULSE_WIDTH = 2400

g = 9.8 * 1000

dt = 1 / 50

STEPS_DEAD_TIME = 2

# control horizon in number of iterations, controller runs at 50 Hz by default
N = 50

Q = 1
R = 5 * 1000

# integral gain for disturbance compensator
DISTURBANCE_APPROXIMATION_INTEGRAL = 1

DISTURBANCE_INTEGRAL_CLIP = 0.1


MAX_ITER = 5

U_min = -0.08
U_max = 0.08
du_max = 0.04

RECORD_FRAMES = True
