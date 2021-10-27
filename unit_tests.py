import unittest

import numpy as np

from hmm import HMM
from hmm_runner import suppliedModel
from touchscreen import touchscreenHMM


class IOTest(unittest.TestCase):
    """
    Tests IO for hmm and touchscreen implementations. Contains basic test cases.

    Each test function instantiates a hmm and checks that all returned arrays/frames are probability distributions and sum to 1.
    """

    def _is_close(self, a, b, rel_tol=1e-07, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def _check_distribution(self, model):
        supplied_model = suppliedModel()
        model_instance = model(
            sensor_model=supplied_model.sensor_model,
            transition_model=supplied_model.transition_model,
            num_states=supplied_model.num_states,
        )
        model_instance.tell("A")
        self.assertTrue(
            self._is_close(np.sum(model_instance.ask(1)), 1),
            "HMM did not produce a probability distribution for timestep 1",
        )
        self.assertTrue(
            self._is_close(np.sum(model_instance.ask(4)), 1),
            "HMM did not produce a probability distribution for timestep 4",
        )

    def _check_filtered_frame(self, model):
        sample_frame = np.zeros((20, 20))
        sample_frame[0][0] = 1.0
        model_instance = model()
        filtered_frame = model_instance.filter_noisy_data(sample_frame) 
        self.assertTrue(
            self._is_close(np.sum(filtered_frame), 1),
            "Filtered frame is not a probability distribution",
        )

    def test_hmm(self):
        self._check_distribution(HMM)

    def test_touchscreen(self):
        self._check_filtered_frame(touchscreenHMM)


if __name__ == "__main__":
    unittest.main()
