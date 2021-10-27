from typing import Callable, List

import numpy as np

# Implement your HMM for Part 1 here!


class HMM:

    # You may add instance variables, but you may not change the
    # initializer; HMMs will be initialized with the given __init__
    # function when grading.
    sensor_model = None
    transition_model = None
    probabilities = None

    def __init__(
        self,
        sensor_model: Callable[[str, int], float],
        transition_model: Callable[[int, int], float],
        num_states: int,
    ):
        """
        Inputs:
        - sensor_model: the sensor model of the HMM.
          This is a function that takes in an observation E
          (represented as a string 'A', 'B', ...) and a state S
          (reprensented as a natural number 0, 1, ...) and
          outputs the probability of observing E in state S.

        - transition_model: the transition model of the HMM.
          This is a function that takes in two states, s and s',
          and outputs the probability of transitioning from
          state s to state s'.

        - num_states: this is the number of hidden states in the HMM, an integer
        """
        # Initialize your HMM here!
        self.sensor_model = sensor_model
        self.transition_model = transition_model
        self.probabilities = np.ones((num_states)) / num_states #in each cell, divide each by num_states
        self.time = 0
        self.num_states = num_states

    def tell(self, observation):
        """
        Takes in an observation and records it.
        You will need to keep track of the current timestep and increment
        it for each observation.

        Input:
        - observation: The observation at the current timestep, a string

        Output:
        - None
        """
        future_probabilities = np.zeros((self.num_states)) #num_state number of zeroes

        for future_state in range(0, self.num_states): #From 0 to num_states
          for current_state in range(0, self.num_states): #from 0 to num_states
            #In each future_state, it equals to the curren_state * the transition model value
            future_probabilities[future_state] += self.probabilities[current_state] * self.transition_model(current_state, future_state)
          #In each future state, times the value by the sensor model value
          future_probabilities[future_state] *= self.sensor_model(observation, future_state)
        
        #Derive alpha value by dividing by the total sum of the future probabilities s.t. the values equal to 1
        self.probabilities = future_probabilities / np.sum(future_probabilities)
        self.time += 1

    def ask(self, time: int) -> List[float]:
        """
        Takes in a timestep that is greater than or equal to
        the current timestep and outputs a probability distribution
        (represented as a list) over states for that timestep.
        The index of the probability is the state it corresponds to.

        Input:
        - time: the timestep to get the observation distribution for, an integer

        Output:
        - a probability distribution over the hidden state for the given timestep, a list of numbers
        """
        future_probabilities = self.probabilities
        # From inputted time to the last time
        for remaining_time in range(self.time, time):
          current_probabilities = future_probabilities
          future_probabilities = np.zeros((self.num_states))
          
          # Same as tell
          for future_state in range(0, self.num_states):
            for current_state in range(0, self.num_states):
              future_probabilities[future_state] += self.probabilities[current_state] * self.transition_model(current_state, future_state)
          future_probabilities /= np.sum(future_probabilities)
        
        return future_probabilities
        
