import matplotlib.pyplot as plt
import time
import numpy as np


def visualize(simulation, frame_length=0.2):
    """
    Shows a simulation of touchscreen movement

    simulation: A list of numpy arrays, the output of the create_touch function from the simulator
    frame_length: How long each frame is displayed
    """
    plt.ion()  # turn on interactive mode, non-blocking `show`
    test = plt.imshow(simulation[0], cmap="CMRmap_r", interpolation="nearest", vmax=4)
    for frame in simulation:
        test.set_data(frame)
        plt.pause(frame_length)
        time.sleep(frame_length)
    plt.close()


def visualize_student(
    student_hmm, noisy_sim, actual_sim, frame_length=0.2, actual=True
):
    """
    Shows a simulation of the touchscreen movement side-by-side with the distribution calculated from
    the student's HMM from part 2.

    student_hmm: touchscreenHMM() object written by the student
    noisy_sim: List of np arrays containing the noisy data over n timepoints
    actual_sim: List of np arrays containing the real finger data over n timepoints
    frame_length: How long each frame is displayed
    actual: Whether to display the actual finger location or just the noisy data
    """
    frames_to_draw = noisy_sim
    if actual:
        combined_frames = []
        for i in range(len(noisy_sim)):
            frame = noisy_sim[i] + (actual_sim[i] * 2)
            combined_frames.append(frame)
        frames_to_draw = combined_frames

    plt.ion()  # turn on interactive mode, non-blocking `show`
    plt.subplot(1, 2, 1)
    test_real = plt.imshow(
        frames_to_draw[0], cmap="CMRmap_r", interpolation="nearest", vmax=4
    )
    plt.subplot(1, 2, 2)
    test_student = plt.imshow(
        frames_to_draw[0], cmap="gist_heat_r", interpolation="nearest", vmax=1.0
    )

    for i, frame in enumerate(frames_to_draw):
        test_real.set_data(frame)
        test_student.set_data(student_hmm.filter_noisy_data(noisy_sim[i]))
        # print(student_hmm.filter_noisy_data(noisy_sim[i]))
        plt.pause(frame_length)
        time.sleep(frame_length)

    plt.close()
