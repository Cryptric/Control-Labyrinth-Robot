import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from MPC import MPC
from MPC_LSQ import MPC_LSQ
import Params


def main():
    with open("recorded_x.pkl", "rb") as f:
        recoded_data_x = pickle.load(f)

    with open("recorded_y.pkl", "rb") as f:
        recoded_data_y = pickle.load(f)

    # Example data (replace this with your own data)

    # Create initial plot
    fig, ax = plt.subplots(nrows=3)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Data Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    start_index = 261

    # Add a slider
    slider_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, 'Index', 0, len(recoded_data_x) - 1, valinit=start_index, valstep=1)

    slider_ax2 = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider2 = Slider(slider_ax2, 'Scale', 0.0, 200000.0, valinit=1.0)

    # Function to update the plot based on slider value
    def update(val):
        index = int(slider.val)

        draw(index)
        fig.canvas.draw_idle()

    def plot_mpc(xk_x, xk_y, ref_x, ref_y):
        mpc_x = MPC(Params.K_x)
        mpc_y = MPC(Params.K_y)

        t = time.time()
        signal_x_rad = mpc_x.get_control_signal(ref_x, xk_x)
        signal_y_rad = mpc_y.get_control_signal(ref_y, xk_y)
        dt = time.time() - t
        print(f"QUA: {dt * 1000}ms")
        print(signal_y_rad)

        predicted_state_x = mpc_x.get_predicted_state(xk_x, signal_x_rad)
        predicted_state_y = mpc_y.get_predicted_state(xk_y, signal_y_rad)

        ax[1].clear()
        ax[1].set_title('Recalculated State')
        ax[1].grid()

        ax[1].plot(xk_x[0], xk_y[0], 'ro')
        ax[1].quiver(xk_x[0], xk_y[0], xk_x[1] / 10, xk_y[1] / 10, angles='xy', scale_units='xy', scale=1, color='blue')

        ax[1].plot(ref_x, ref_y, marker=".", label="ref trajectory")
        ax[1].plot(predicted_state_x, predicted_state_y, marker=".", label="prod trajectory")

    def plot_mpc_lsq(xk_x, xk_y, ref_x, ref_y):
        mpc_x = MPC_LSQ(Params.K_x)
        mpc_y = MPC_LSQ(Params.K_y)

        t = time.time()
        signal_x_rad = mpc_x.get_control_signal(ref_x, xk_x)
        signal_y_rad = mpc_y.get_control_signal(ref_y, xk_y)
        dt = time.time() - t
        print(f"LSQ: {dt * 1000}ms")

        predicted_state_x = mpc_x.get_predicted_state(xk_x, signal_x_rad)
        predicted_state_y = mpc_y.get_predicted_state(xk_y, signal_y_rad)

        ax[2].clear()
        ax[2].set_title('Recalculated State')
        ax[2].grid()

        ax[2].plot(xk_x[0], xk_y[0], 'ro')
        ax[2].quiver(xk_x[0], xk_y[0], xk_x[1] / 10, xk_y[1] / 10, angles='xy', scale_units='xy', scale=1, color='blue')

        ax[2].plot(ref_x, ref_y, marker=".", label="ref trajectory")
        ax[2].plot(predicted_state_x, predicted_state_y, marker=".", label="prod trajectory")

    def draw(index):
        state_x, ref_x, pred_x = recoded_data_x[index]
        state_y, ref_y, pred_y = recoded_data_y[index]

        ax[0].clear()
        ax[0].set_title("Recorded data")
        ax[0].grid()

        ax[0].plot(state_x[0], state_y[0], 'ro')
        ax[0].quiver(state_x[0], state_y[0], state_x[1] / 10, state_y[1] / 10, angles='xy', scale_units='xy', scale=1, color='blue')

        ax[0].plot(ref_x, ref_y, marker=".", label="ref trajectory")
        ax[0].plot(pred_x, pred_y, marker=".", label="prod trajectory")

        plot_mpc(state_x, state_y, ref_x, ref_y)
        plot_mpc_lsq(state_x, state_y, ref_x, ref_y)


    # Register the update function with the slider
    slider.on_changed(update)
    slider2.on_changed(update)

    draw(start_index)

    plt.show()


def y_state_update(pos, speed):
    pos = pos + Params.dt * speed
    speed = speed + Params.dt * Params.K_y * Params.g * (-0.07) * 5/7
    return pos, speed


if __name__ == '__main__':
    main()
