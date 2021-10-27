import math
import numpy as np
from math import sqrt
from scipy.stats import norm


class touchscreenEvaluator:
    def __init__(self):
        self.past_distributions = {}

    def calc_score(self, actual_frame, estimated_frame):
        """
        Calculates the accuracy of a frame distribution. It works by looking at the estimated_frame and applying a normal
        distribution map over the actual position, rewarding points for higher distributions around the actual point.
        :param actual_frame:
        :param estimated_frame:
        :return: The score of the corresponding board
        """
        n = len(actual_frame) * len(actual_frame[0])
        actual = np.reshape(actual_frame, n)
        estimated = np.reshape(estimated_frame, n)
        total = sum(estimated)
        if any(estimated < -0.001) or total > 1.001 or total < 0.999:
            print("Estimated frame is not a probability distribution!")
            return 0

        return (1 + 2 * np.dot(actual, estimated) - np.dot(estimated, estimated)) / 2.0

    def calc_consistency_score(self, actual_frame, estimated_frame):
        actual_loc = np.asarray(np.where(actual_frame == 1)).T.tolist()[0]
        if estimated_frame[actual_loc[0]][actual_loc[1]] > (1 / 400):
            return True
        else:
            return False

    def grid_distance(self, x1, y1, x2, y2):
        """
        Calculates the grid distance between two coordinates (diagonal moves count as 1)
        Inputs:
           x1, y1 - int coordinates of first point
           x2, y2 - int coordinates of second point
        Outputs:
           int representing the grid distance
        """
        return max(abs(x1 - x2), abs(y1 - y2))

    def calc_distribution_score(self, noisy_frame, actual_frame, student_frame):
        """
        Scores the accuracy of a student's frame distributtion comparing it to the acutal finger location frame.
        Rewards a higher amount of points when the further the noisy loc differs from the acutal loc.
        Grid locations closer to the real finger location will be rewarded more points.
        Specifics of the distribution scaling:
           - Points within a range [0, n/4] radius of real finger location will receive range [0.5, 0]^3 weight
           - Points within range (n/4, n] radius of real finger location will receive range (0, -1.5)^3 weight
           Thus it is possible to lose points if distribution falls outside of n/4 radius

        Inputs:
           noisy_frame: Noisy frame generated from simulation
           actual_frame: Acutal finger location frame from simulation
           student_frame: The student's distribution frame created by filter_noisy_data in touchscreenHMM

        Outputs:
           A score represented as a float for the frame (at timepoint t)
        """
        score = 0
        n = len(actual_frame)
        actual_loc = np.asarray(np.where(actual_frame == 1)).T.tolist()[0]
        noisy_loc = np.asarray(np.where(noisy_frame == 1)).T.tolist()[0]

        multipler = (
            self.grid_distance(noisy_loc[0], noisy_loc[1], actual_loc[0], actual_loc[1])
            / (2 * n)
        ) + 1

        for i in range(len(actual_frame)):
            for j in range(len(actual_frame[0])):
                if student_frame[i][j] == 0:
                    continue
                dist = self.grid_distance(actual_loc[0], actual_loc[1], i, j)
                score += math.pow(0.5 - 2 * (dist / n), 3) * student_frame[i][j]
        return multipler * score

    def evaluate_consistency(self):
        return self.missed

    def evaluate_student_hmm(self, touchscreenHMM, simulation):
        """
        ~ New version of evaluate_touchscreen_hmm function ~

        Calculates the accuracy of the students project.
        :param touchscreenHMM: an instance of a student's touchscreenHMM
               simulation: an instance of the touchscreen simulator
        :return: A dict of scores (keys: total, momentum, distribution, density)
        """
        pass

    def scale_score(self, score):
        """
        Simple scaling of scores from (0.2, 0.65) to (0, 100)
        """
        return (score - 0.2) * 223 if score > 0.2 else 0

    def evaluate_touchscreen_hmm(self, touchscreenHMM, simulation):
        """
        Calculates the accuracy of the students project.
        :param touchscreenHMM: An instance of a student's touchscreenHMM
        :return: The 'percentage accuracy' from the calc_score function over the whole screen, and the 'percentage
        accuracy' of just inputting the noisy location.
        """
        print("Evaluating student touchscreenHMM.")
        score = 0
        noisy_score = 0
        actual = 0
        frame = simulation.get_frame(actual_position=True)
        i = 0
        self.missed = 0
        self.noisy_missed = 0
        while frame:
            i += 1
            student_frame = touchscreenHMM.filter_noisy_data(frame[0])
            score += self.calc_score(frame[1], student_frame)
            actual += self.calc_score(frame[1], frame[1])
            noisy_score += self.calc_score(frame[1], frame[0])

            if not self.calc_consistency_score(frame[1], student_frame):
                self.missed += 1
            if not self.calc_consistency_score(frame[1], frame[0]):
                self.noisy_missed += 1

            frame = simulation.get_frame(actual_position=True)
        acc_score = round(self.scale_score(score / actual), 3)
        noisy_score = round(self.scale_score(noisy_score / actual), 3)
        rubric1 = (acc_score - noisy_score) / (100 - noisy_score)
        rubric1 = min(sqrt(max(rubric1, 0)) * 35, 35)

        rubric2 = (self.noisy_missed - self.missed) / self.noisy_missed
        rubric2 = min(max(rubric2 * 15, 0), 15)
        return {
            "accuracy_score": acc_score,
            "noisy_score": noisy_score,
            "rubric_1": str(round(rubric1, 1)) + "/35",
            "missed_frames": self.missed,
            "noisy_frames": self.noisy_missed,
            "rubric_2": str(round(rubric2, 1)) + "/15",
        }
