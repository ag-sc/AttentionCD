import argparse

from heads_selector.jsd_heads_selector.selector import run_jsd_heads_selector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name

    run_jsd_heads_selector(model_name=model_name, dataset_name=dataset_name)
