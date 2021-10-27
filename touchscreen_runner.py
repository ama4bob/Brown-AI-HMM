import argparse
import json

import numpy as np

from touchscreen import touchscreenHMM
from touchscreen_helpers.simulation_evaluator import touchscreenEvaluator
from touchscreen_helpers.simulator import touchscreenSimulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the touchscreen simulator")
    parser.add_argument(
        "--width", type=int, default=20, help="the width of the touchscreen"
    )
    parser.add_argument(
        "--height", type=int, default=20, help="the height of the touchscreen"
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="the number of frames to simulate"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show visualization of touchscreen simulation",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="score the student's HMM"
    )
    parser.add_argument("--save_file", type=str, help="save the simulation to a file")
    parser.add_argument("--load_file", type=str, help="load the simulation from a file")
    parser.add_argument(
        "--frame_length",
        type=float,
        default=0.1,
        help="the number of seconds each frame is shown for",
    )

    args = parser.parse_args()
    save_file = args.save_file
    load_file = args.load_file

    if save_file and load_file:
        parser.error("both --save_file and --load_file were given")
    elif load_file and not args.visualize and not args.evaluate:
        parser.error(
            "nothing to do for loaded simulation: --visualize and --evaluate were false"
        )

    simulator = touchscreenSimulator(args.width, args.height, args.frames)

    if load_file:
        print(f"Loading saved simulation from {load_file}.")
        with open(load_file, "r") as f:
            first_line = [int(x) for x in f.readline().strip().split(" ")]
            width, height, frames = first_line
            data = np.loadtxt(f, dtype="int", skiprows=0)
        simulator.load_simulation(data)
    else:
        print("Running simulation.")
        simulator.run_simulation()

    if save_file:
        print(f"Saving simulation to {save_file}.")
        with open(save_file, "w") as f:
            f.write(
                "%d %d %d\n" % (simulator.width, simulator.height, simulator.frames)
            )
            for (noisy, actual) in iter(
                lambda: simulator.get_frame(actual_position=True), None
            ):
                noisy_loc = np.asarray(np.where(noisy == 1)).T.tolist()[0]
                actual_loc = np.asarray(np.where(actual == 1)).T.tolist()[0]
                to_print = noisy_loc + actual_loc
                np.savetxt(f, to_print, fmt="%i", newline=" ")
                f.write("\n")
        simulator.timestamp = 0

    if args.evaluate:
        student_hmm = touchscreenHMM(args.width, args.height)
        if args.visualize:
            simulator.visualize_results(student_hmm, args.frame_length)
        evaluator = touchscreenEvaluator()
        score = evaluator.evaluate_touchscreen_hmm(student_hmm, simulator)
        print(f"Score: {json.dumps(score, indent=4, sort_keys=True)}")
    elif args.visualize:
        simulator.visualize_simulation(args.frame_length)
