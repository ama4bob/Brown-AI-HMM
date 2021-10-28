import numpy as np
from touchscreen_helpers.generate_data import create_simulations
from typing import Callable, List

# Implement part 2 here!

class HMM:

    # You may add instance variables, but you may not change the
    # initializer; HMMs will be initialized with the given __init__
    # function when grading.
    sensor_model = None
    transition_model = None
    probabilities = None

    def __init__(
        self,
        sensor_model,
        transition_model,
        width: int,
        height: int,
    ):
        # Initialize your HMM here!
        self.sensor_model = sensor_model
        self.transition_model = transition_model
        self.num_states = width * height
        self.probabilities = np.ones((self.num_states)) / self.num_states #in each cell, divide each by num_states
        self.mapToState = {w*height + h: (w, h) for w in range(width) for h in range(height)}

    def tell(self, observation):
        future_probabilities = np.zeros((self.num_states)) #num_state number of zeroes

        for fs in range(0, self.num_states): #From 0 to num_states
          for cs in range(0, self.num_states): #from 0 to num_states
            future_state = self.mapToState[fs]
            current_state = self.mapToState[cs]
            #In each future_state, it equals to the curren_state * the transition model value
            # print(f"current prob {self.probabilities[cs]}")
            # print(f"transition prob {self.transition_model(current_state, future_state)}")
            # print(f"future state {future_state}  current state{current_state}\n\n\n")
            future_probabilities[fs] += self.probabilities[cs] * self.transition_model(current_state, future_state)
          #In each future state, times the value by the sensor model value
          future_probabilities[fs] *= self.sensor_model(observation, future_state)
        
        #Derive alpha value by dividing by the total sum of the future probabilities s.t. the values equal to 1
        self.probabilities = future_probabilities / np.sum(future_probabilities)
        return self.probabilities


class touchscreenHMM:

    # You may add instance variables, but you may not create a
    # custom initializer; touchscreenHMMs will be initialized
    # with no arguments.

    def __init__(self, width=20, height=20):
        """
        Feel free to initialize things in here!
        """
        self.width = width
        self.height = height
        self.generate_models()
        self.hmm = HMM(self._sensor_model, self._transition_model, self.width, self.height)

    # NOTE: _sensor_model and _transition_model are private helper functions,
    # which means that they will only be called by you. This also means you are
    # free to change the parameters of these functions as you see fit. You may
    # even delete them! They are only here to point you in the right direction.

    def arr_to_pos(self, arr):
        return tuple(map(lambda x: x[0], np.where(arr)))

    def generate_models(self):
        #simulations = tuple(map(lambda arr: tuple(map(lambda x: x[0], np.where(arr))), create_simulations(1)[0]))
        simulations = tuple(map(lambda simulation: tuple(map(lambda arr: self.arr_to_pos(arr), simulation)), create_simulations(50, 100000)))
        #print(f"simulations\n\n\n\n{simulations}")
        tr_map = {}
        sn_map = {}
        prev_st = (-1, -1)

        for sim in simulations:
            # initialize transitions
            st = sim[1]
            prev_st_tr = tr_map.setdefault(prev_st, {})
            prev_st_tr[st] = prev_st_tr.get(st, 0) + 1  # What is this doing?
            prev_st = st

            # Initialize observations/sensors
            obs = sim[0]
            st_sn = sn_map.setdefault(st, {})
            st_sn[obs] = st_sn.get(obs, 0) + 1
        
        for prev_st in tr_map:
            count = sum(tr_map[prev_st].values())
            for st in tr_map[prev_st]:
                tr_map[prev_st][st] = tr_map[prev_st][st]/count 

        for obs in sn_map:
            count = sum(sn_map[obs].values())
            for st in sn_map[obs]:
                sn_map[obs][st] = sn_map[obs][st]/count 

        self.sn_model = sn_map
        self.tr_model = tr_map

        #print(f"SN MAP \n\n\n\n\n\n {sn_map}")
        #print(f"TR MAP \n\n\n\n\n\n {tr_map}")
           

        return

    def _sensor_model(self, observation, state) -> float:
        """
        This is the sensor model to get the probability of getting an observation from a state.

        Input:
        - observation: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.
        - state:       A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.

        Output:
        - The probability of observing that observation from that given state, a number.
        """
        # Write your code here!
        # normalize the probabilities
        #print("\n\n\n sensor model: \n\n\n" + str(self.sn_model.get(observation, {}).get(state, 0)))
        return self.sn_model.get(observation, {}).get(state, 0)

    def _transition_model(self, old_state, new_state) -> float:
        """
        This will be your transition model to go from the old state to the new state.

        Input:
        - old_state: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.
        - new_state: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.

        Output:
        - The probability of transitioning from the old state to the new state, a number.
        """
        #print("\n\n\n transition model: \n\n\n" + str(self.tr_model.get(old_state,{}).get(new_state, 0)))
        return self.tr_model.get(old_state,{}).get(new_state, 0)
        

    def filter_noisy_data(self, frame: np.ndarray) -> np.ndarray:
        """
        This is the function we will be calling during grading, passing in a noisy simualation. It should return the
        distribution where you think the actual position of the finger is in the same format that it is passed in as.

        DO NOT CHANGE THE FUNCTION PARAMETERS

        Input:
        - frame: A noisy frame to run your HMM on. This is a 2D NumPy array
                 filled with 0s, and a single 1 denoting a touch location.

        Output:
        - A 2D NumPy array with the probabilities of the actual finger location.
        """
        # Write your code here!
        return self.hmm.tell(self.arr_to_pos(frame))




if __name__ == "__main__":
    hmm = touchscreenHMM()
    # TODO: Use create_simulations to perform data analysis on several simulations.
    #
    # Here are some questions you may want to find answers for by performing
    # statistics on the simulations:
    #   - How often is the finger position (actual frame) the same as the
    #     observed finger position (noisy frame)?
    #   - If the observation is different frame the actual frame, how far away
    #     is it? Are certain distances more likely than others?
    #   - How likely does the finger continue in the same direction it was going?
    #
    # Come up with your own questions and try to answer them by analyzing the
    # generated simulations. Try to think about which statistics will inform
    # your sensor model, and which ones will inform your transition model!

    pass
