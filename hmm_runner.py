import math
import string

from hmm import HMM


class suppliedModel:
    """
    A class containing the sensor_model and transition_model
    """

    def __init__(self):
        self.num_states = 4
        self.observations = list(string.ascii_uppercase)

    def char_to_int(self, letter):
        """
        converts a character to its corresponding place in the alphabet.
        For example, 'A' Is converted to 0.
        """
        letter = letter.upper()
        if letter not in self.observations:
            raise ValueError(f"The letter must be one of: {self.observations}")
        return self.observations.index(letter)

    def sensor_model(self, observation, state):
        """
        Takes in a letter observation and a state to compute P(observation | state)
        """
        observation_index = self.char_to_int(observation)
        y = (observation_index - 2 * state) / 3
        return math.exp(-(y ** 2) / 2) / (math.sqrt(2 * math.pi))
        # Equivalent to:
        # return scipy.stats.norm.pdf(observation_index, loc=2 * state, scale=3)

    def transition_model(self, old_state, new_state):
        """
        Takes in two states to calculate P(new_state | old_state)
        """
        M = [
            [0.5, 0.2, 0.1, 0.2],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.5, 0.2],
            [0.2, 0.1, 0.2, 0.5],
        ]
        return M[old_state][new_state]


class simpleModel:
    """
    A class containing the sensor_model and transition_model
    Only Accepts observations 'A', 'B', and 'C' for simplicity
    """

    def __init__(self):
        self.num_states = 4
        self.observations = ["A", "B", "C"]

    def char_to_int(self, letter):
        """
        converts a character to its corresponding place in the alphabet.
        For example, 'A' Is converted to 0.
        """
        letter = letter.upper()
        if letter not in self.observations:
            raise ValueError(f"The letter must be one of: {self.observations}")
        return self.observations.index(letter)

    def sensor_model(self, observation, state):
        """
        Takes in a letter observation and a state to compute P(observation | state)
        Only Accepts observations 'A', 'B', and 'C' for simplicity
        """
        observation_index = self.char_to_int(observation)
        M = [
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.9, 0.0],
        ]
        return M[state][observation_index]

    def transition_model(self, old_state, new_state):
        """
        Takes in two states to calculate P(new_state | old_state)
        """
        M = [
            [0.5, 0.2, 0.1, 0.2],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.5, 0.2],
            [0.2, 0.1, 0.2, 0.5],
        ]
        return M[old_state][new_state]


if __name__ == "__main__":
    """
    Runs A REPL function where you can add observations as letters, and it
    returns the ask() value of the current timepoint. You can also change the
    model to the mismatchedObservations model to hand simulate more easily!
    """
    import argparse
    import readline as _

    parser = argparse.ArgumentParser(description="Run the generic HMM REPL")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="use the simpleModel, which is easier to hand-simulate",
    )
    args = parser.parse_args()

    model = simpleModel() if args.simple else suppliedModel()
    hmm = HMM(
        model.sensor_model,
        model.transition_model,
        model.num_states,
    )

    print(f"Model: {type(model).__name__}")
    t = 0
    while t < 10:
        print(f"[t={t}] {hmm.ask(t)}")
        observation = input(f"[t={t}] Enter an observation: ")
        hmm.tell(observation)
        t += 1
    print(f"[t={t}] {hmm.ask(t)}")
