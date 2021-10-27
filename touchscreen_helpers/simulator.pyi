from typing import List, Tuple, Union

import numpy as np

from ..touchscreen import touchscreenHMM

class touchscreenSimulator:
    """
    Simulates a touchscreen and finger movements for the hidden markov model. It was chosen to use
    a numpy array to show the state over a plain coordinate of the touch to show how complex problems
    can be broken down into simple state spaces.
    """

    width: int
    height: int
    frames: int
    timestamp: int

    def __init__(self, width: int = 20, height: int = 20, frames: int = 100) -> None:
        """
        Creates a new touchscreen simulator
        Input:
        - width:  The width of the simulated screen
        - height: The height of the simulated screen
        """
        ...
    def run_simulation(self) -> None: ...
    def convert_coordinate_to_screen(self, x: int, y: int) -> np.ndarray: ...
    def load_simulation(self, data: List[Tuple[int, int, int, int]]) -> None: ...
    def visualize_simulation(
        self, noisy: bool = True, actual: bool = True, frame_length: float = 0.5
    ) -> None:
        """
        Visualizes the simulations, capable of both at the same time
        Input:
        - noisy:  If True, it will display the Noisy data
        - actual: If True, it will display the Actual data
        - frame_length: How long each frame is
        """
        ...
    def visualize_results(
        self,
        student_hmm: touchscreenHMM,
        actual: bool = True,
        frame_length: float = 0.5,
    ) -> None:
        """
        Function to visualize student's distribution over simulator data.
        Requires noisy data (because student HMM requires noisy data input);
        Actual data optional
        """
        ...
    def get_frame(
        self, actual_position: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Gets the next frame
        Input:
        - actual_position: A boolean if true, aslo returns the actual position
                           of the finger. Only to be set to True for testing
        Output:
        - The noisy frame or a tuple with the noisy frame and the actual frame.
        """
        ...
