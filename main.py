import argparse

from cats_ai.evaluation import experiment_accident, experiment_source
from cats_ai.prompts import ACCIDENT_ANALYSIS, ACCIDENT_DETECTION, ACCIDENT_PREDICTION

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    match args.prompt:
        case 0:
            experiment_accident(ACCIDENT_PREDICTION, True)
        case 1:
            experiment_accident(ACCIDENT_DETECTION, False)
        case 2:
            experiment_accident(ACCIDENT_ANALYSIS, False)
        case 3:
            experiment_source()
