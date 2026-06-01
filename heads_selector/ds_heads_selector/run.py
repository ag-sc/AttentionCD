import argparse


from .tfqa_mc import run_head_selection as run_head_selection_tfqa_mc
from .triviaqa_natural_questions import (
    run_head_selection as run_head_selection_triviaqa,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="data/natural_questions")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_path = args.data_path

    if "natural_questions" in dataset_name or "triviaqa" in dataset_name:
        run_head_selection_triviaqa(
            model_name=model_name, dataset_name=dataset_name, data_path=data_path
        )
    elif "tfqa_mc" in dataset_name:
        run_head_selection_tfqa_mc(model_name=model_name, data_path=data_path)
    else:
        raise Exception(
            "Invalid dataset name. Supported datasets are natural_questions, triviaqa and tfqa_mc"
        )
