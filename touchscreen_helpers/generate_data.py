from typing import List, Tuple

import numpy as np

from .simulator import touchscreenSimulator


def create_simulations(
    size: int = 20, frames: int = 100
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates a simulator and selects all frames. This can be useful when you are trying to better understand the
    behaviour of the touchscreen simulator

    Input:
    - size:   The size of the screen
    - frames: The number of frames to generate over

    Output:
    - A list of tuples, with each index representing a timestep. Each tuple
      looks like (noisy_frame, actual_frame), where the frames are 2D NumPy
      arrays filled with 0s, and a single 1 denoting a touch location.
    """
    sim = touchscreenSimulator(width=size, height=size, frames=frames)
    sim.run_simulation()

    sim_data = []
    for _ in range(frames):
        sim_data.append(sim.get_frame(actual_position=True))
    return sim_data
