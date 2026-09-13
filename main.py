import argparse

from cats_ai.evaluation import experiment_accident, experiment_source
from cats_ai.prompts import ACCIDENT_ANALYSIS, ACCIDENT_DETECTION, ACCIDENT_PREDICTION

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--prompt", type=int, choices=[0, 1, 2, 3], required=True)

    parser.add_argument(
        "-m", "--modelid", type=str, choices=["2B", "4B", "8B"], required=True
    )

    args = parser.parse_args()
    model_id = args.modelid

    match args.prompt:
        case 0:
            experiment_accident(
                ACCIDENT_PREDICTION, True, "ACCIDENT_PREDICTION", model_id
            )
        case 1:
            experiment_accident(
                ACCIDENT_DETECTION, False, "ACCIDENT_DETECTION", model_id
            )
        case 2:
            experiment_accident(ACCIDENT_ANALYSIS, False, "ACCIDENT_ANALYSIS", model_id)
        case 3:
            experiment_source(model_id)
