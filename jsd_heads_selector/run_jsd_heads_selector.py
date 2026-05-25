import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama_modified import (
    LlamaForCausalLM as ModifiedLlamaForCausalLM,
)

import gc
import torch
import json

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch.nn.functional as F

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

import os
import numpy as np


def load_json(file_path):
    # input file is in json format

    with open(file_path, "r") as f:
        list_data = json.load(f)

    return list_data


def load_tfqa_csv(file_path):

    list_data = []
    with open(file_path, "r") as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {
                "question": df["Question"][idx],
                "answer_best": df["Best Answer"][idx],
                "answer_true": df["Correct Answers"][idx],
                "answer_false": df["Incorrect Answers"][idx],
            }
            list_data.append(data)

    return list_data


TRIVIAQA_VAL_FOLD_0_PATH = "data/triviaqa/triviaqa_val_fold_0_2k"
TRIVIAQA_VAL_FOLD_1_PATH = "data/triviaqa/triviaqa_val_fold_1_2k"
NQ_VAL_FOLD_0_PATH = "data/natural_questions/natural_questions_val_fold_0_1k"
NQ_VAL_FOLD_1_PATH = "data/natural_questions/natural_questions_val_fold_1_1k"


def load_csv(dataset_name):
    # input file is in csv format, can be loaded by pandas
    # required columns: [prompt] only

    if dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext")["validation"]
    elif dataset_name == "natural_questions":
        dataset = load_dataset("nq_open")["validation"]
    elif "triviaqa_val_fold_0" == dataset_name:
        print("WARNING: Using fold 0 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_0_PATH)
    elif "triviaqa_val_fold_1" == dataset_name:
        print("WARNING: Using fold 1 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_1_PATH)
    elif "natural_questions_val_fold_0" == dataset_name:
        print("WARNING: Using fold 0 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(NQ_VAL_FOLD_0_PATH)
    elif "natural_questions_val_fold_1" == dataset_name:
        print("WARNING: Using fold 1 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(NQ_VAL_FOLD_1_PATH)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    list_data = list(dataset["question"])

    if "triviaqa" in dataset_name:
        labels = []
        for x in dataset["answer"]:
            labels.append([x["normalized_value"]])
    else:
        labels = list(dataset["answer"])

    return list_data, labels


def get_all_texts(list_data_dict, labels):
    all_texts = []

    for prompt, answers in zip(list_data_dict, labels):

        for answer in answers:
            prompt = f"Q: {prompt}\n"
            answer = f"A: {answer}"
            all_texts.append(
                {"prompt": prompt, "answer": answer, "text": prompt + answer}
            )

    return all_texts


def get_all_texts_tfqa(list_data_dict):
    from benchmark.tfqa_mc import format_best, split_multi_answer

    all_texts = []
    for idx, sample in enumerate(list_data_dict):
        # reference answers
        ref_best = format_best(sample["answer_best"])
        ref_true = split_multi_answer(sample["answer_true"])
        ref_false = split_multi_answer(sample["answer_false"])

        for temp_ans in ref_true:
            # prompt, answer = build_prompt_and_answer(sample["question"], temp_ans)
            prompt = f"Q: {sample['question']}\n"
            answer = f"A: {temp_ans}"
            all_texts.append(
                {"prompt": prompt, "answer": answer, "text": prompt + answer}
            )

    return all_texts


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"], return_tensors="pt", padding=True, truncation=True
    )
    prompt_tokens = tokenizer(examples["prompt"], truncation=True)
    model_inputs["prompt_len"] = [len(ids) for ids in prompt_tokens["input_ids"]]

    return model_inputs


def get_log_probs(model, tokenizer, dataloader):
    with torch.no_grad():
        all_log_probs = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            prompt_lens = batch["prompt_len"]

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # (seq_len, vocab_size)
            outputs = outputs.log_softmax(dim=-1)

            relevant_outputs = [
                # outputs[i, prompt_len - 1 : -1, :].detach().cpu().float().numpy()
                outputs[i, prompt_len - 1 : -1, :]
                for i, prompt_len in enumerate(batch["prompt_len"])
            ]

            all_log_probs.extend(relevant_outputs)

    return all_log_probs


def get_js_divs(
    final_logits: torch.FloatTensor,
    premature_layers_logits: torch.FloatTensor,
) -> torch.FloatTensor:

    # 2. Calculate the softmax values for mature_layer and all premature_layers
    # shape: (batch_size, vocab_size)
    softmax_mature_layer = F.softmax(final_logits, dim=-1)
    # print("softmax_mature_layer", softmax_mature_layer.shape)

    # shape: (num_premature_layers, batch_size, vocab_size)
    softmax_premature_layers = F.softmax(premature_layers_logits, dim=-1)
    # print("softmax_premature_layers", softmax_premature_layers.shape)

    # 3. Calculate the average distribution
    # shape: (num_premature_layers, batch_size, vocab_size)
    avg_dist = 0.5 * (softmax_mature_layer + softmax_premature_layers)

    # 4. Calculate log-softmax for the KL divergence
    # shape: (batch_size, vocab_size)
    log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1)

    # shape: (num_premature_layers, batch_size, vocab_size)
    log_softmax_premature_layers = F.log_softmax(premature_layers_logits, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    # shape: (num_premature_layers, batch_size)
    kl1 = F.kl_div(log_softmax_mature_layer, avg_dist, reduction="none").mean(-1)
    # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1)

    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
    js_divs = js_divs.mean(-1)

    return js_divs.detach().cpu().float().numpy()


def get_heads_dict_for_threshold(
    k: int,
    df: pd.DataFrame,
    model_tag: str,
    model,
    heads_dict_dir: str,
):
    """
    if k = 1, calculates 99th percentile --> top 1%
    finds the threshold such that only 1% of values are higher than the threshold

    """
    threshold = df["mean_js_div"].quantile((100 - k) / 100)
    top_k_percent = df[df["mean_js_div"] >= threshold].sort_values(
        by="mean_js_div", ascending=False
    )

    print(f"Threshold for top {k}%: {threshold:.2e}")

    total_num_heads = model.config.num_hidden_layers * model.config.num_attention_heads
    percentage_of_heads = (len(top_k_percent) / (total_num_heads)) * 100
    print(f"Total num of heads: {total_num_heads}")
    print(f"Number of divergent heads: {len(top_k_percent)}")
    print(f"Percentage of divergent heads: {round(percentage_of_heads, 2)} %")

    heads_dict = {idx: [] for idx in range(model.config.num_hidden_layers)}
    for _, row in top_k_percent.iterrows():
        layer_idx = row["layer_idx"]
        head_idx = int(row["head_idx"].item())

        heads_dict[int(layer_idx)].append(head_idx)

    heads_dict_path = f"{heads_dict_dir}/heads_dict_js_divs_top_{k}_{model_tag}.json"
    os.makedirs(os.path.dirname(heads_dict_path), exist_ok=True)

    with open(heads_dict_path, "w+") as f:
        json.dump(heads_dict, f, indent=4)

    print(f"Saved to: {heads_dict_path}")


def get_heads_dict_for_threshold_per_layer(
    k: int, matrix, heads_dict_dir: str, model_tag: str
):
    top_threshold_percentile = 100 - k

    divergent_heads = {}
    for layer, x in enumerate(matrix):
        keep_percentile = np.percentile(x, top_threshold_percentile)
        keep_indices = np.where(matrix[layer] >= keep_percentile)[0].tolist()
        if keep_indices:
            divergent_heads[layer] = keep_indices
        else:
            divergent_heads[layer] = []

    heads_dict_path = f"{heads_dict_dir}/heads_dict_js_divs_top_{k}_{model_tag}.json"
    os.makedirs(os.path.dirname(heads_dict_path), exist_ok=True)

    with open(heads_dict_path, "w+") as f:
        json.dump(divergent_heads, f, indent=4)

    print(f"Saved to: {heads_dict_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name

    print("model", model_name)
    print("dataset", dataset_name)

    model_tag = model_name.split("/")[-1].strip()

    model_args = {}
    model_args["torch_dtype"] = torch.bfloat16
    model_args["attn_implementation"] = "eager"

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, **model_args
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    assistant_model = ModifiedLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, **model_args
    )
    assistant_model.eval()
    assistant_model.to(model.device)

    if dataset_name.startswith("tfqa"):
        if "val_fold" in dataset_name:
            # tfqa_val_fold_0 or tfqa_val_fold_1
            print("Fetching TFQA data from json files.")
            list_data_dict = load_json(f"data/tfqa/{dataset_name}.json")
        else:
            print("Fetching whole TFQA data")
            fp = os.path.join("data/tfqa/TruthfulQA.csv")
            list_data_dict = load_tfqa_csv(fp)

        all_texts = get_all_texts_tfqa(list_data_dict)
    else:
        print(f"Fetching data from {dataset_name}")
        list_data_dict, labels = load_csv(dataset_name=dataset_name)
        all_texts = get_all_texts(list_data_dict, labels)

    print("# samples", len(list_data_dict))

    print("all_texts")
    print(all_texts[0])

    ds = Dataset.from_list(all_texts)
    tokenized_dataset = ds.map(
        tokenize_function, batched=True, remove_columns=ds.column_names
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_size = 32
    dataloader = DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    base_all_log_probs = get_log_probs(
        model=model, tokenizer=tokenizer, dataloader=dataloader
    )

    print("base log probs shape", base_all_log_probs[0].shape)

    all_results = []

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    print(f"Iterating through {num_heads} heads in {num_layers} layers.")

    for layer_idx in tqdm(range(0, num_layers)):
        for head_idx in range(0, num_heads):
            print(
                f"Running: layer: {layer_idx}, head: {head_idx}", end="\r", flush=True
            )

            result_dict = {}
            os.environ["LAYERS_TO_ABLATE"] = f"{layer_idx}"
            os.environ["HEADS_TO_ABLATE"] = f"{head_idx}"

            all_log_probs = get_log_probs(
                model=assistant_model, tokenizer=tokenizer, dataloader=dataloader
            )

            js_divs = []
            for base_probs, log_probs in zip(base_all_log_probs, all_log_probs):
                base_probs = torch.Tensor(base_probs)
                log_probs = torch.Tensor(log_probs)
                js_divs.append(get_js_divs(base_probs, log_probs))

            mean_js_div = np.mean(js_divs).item()

            result_dict["mean_js_div"] = mean_js_div
            result_dict["layer_idx"] = layer_idx
            result_dict["head_idx"] = head_idx

            all_results.append(result_dict)

            del all_log_probs
            gc.collect()
            torch.cuda.empty_cache()

    head_divs_results_dir = f"head_divs_results_{dataset_name}"
    os.makedirs(head_divs_results_dir, exist_ok=True)

    path = f"{head_divs_results_dir}/head_divs_results_{dataset_name}_{model_tag}.json"
    with open(path, "w+") as f:
        json.dump(all_results, f)

    df = pd.DataFrame(all_results)
    pivot_df = df.pivot(index="layer_idx", columns="head_idx", values="mean_js_div")
    matrix = pivot_df.to_numpy()

    heads_dict_path = ""
    for k in [1, 5, 10, 15, 20, 25, 50, 75, 100]:
        get_heads_dict_for_threshold(
            k=k,
            df=df,
            model_tag=model_tag,
            model=model,
            heads_dict_dir=f"{head_divs_results_dir}/js_divs",
        )
        print("-----\n\n")

    for k in [1, 5, 10, 15, 20, 25, 50, 75, 100]:
        get_heads_dict_for_threshold_per_layer(
            k=k,
            matrix=matrix,
            model_tag=model_tag,
            heads_dict_dir=f"{head_divs_results_dir}/js_divs_per_layer",
        )
        print("-----\n\n")
