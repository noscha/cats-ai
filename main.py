import argparse

from cats_ai.evaluation import experiment
from cats_ai.prompts import ACCIDENT_ANALYSIS, ACCIDENT_DETECTION, ACCIDENT_PREDICTION

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=int, choices=[0, 1, 2])
    args = parser.parse_args()

    if args.prompt == 0:
        experiment(ACCIDENT_PREDICTION, True)
    elif args.prompt == 1:
        experiment(ACCIDENT_DETECTION, False)
    elif args.prompt == 2:
        experiment(ACCIDENT_ANALYSIS, False)
