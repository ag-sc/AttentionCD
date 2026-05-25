# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import ssl
import urllib.request
from datasets import Dataset, load_dataset

from transformers import AutoConfig
import torch
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_substrings=None, tail_length=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_substrings = stop_substrings if stop_substrings else []
        self.tail_length = tail_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        seq = input_ids[0]
        seq_len = seq.shape[0]

        if seq_len == 0:
            return False

        start_idx = max(0, seq_len - self.tail_length)
        tail_tokens = seq[start_idx:].tolist()
        decoded_tail = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)

        for stop_str in self.stop_substrings:
            if stop_str in decoded_tail:
                return True

        return False


def set_stop_words(model, tokenizer, stop_words):
    model.stop_words = stop_words
    model.stopping_criteria = StoppingCriteriaList()
    model.stopping_criteria.append(
        LLamaQaStoppingCriteria(tokenizer, stop_words, tail_length=3)
    )


TRIVIAQA_VAL_FOLD_0_PATH = "data/triviaqa/triviaqa_val_fold_0_2k"
TRIVIAQA_VAL_FOLD_1_PATH = "data/triviaqa/triviaqa_val_fold_1_2k"
NQ_VAL_FOLD_0_PATH = "data/natural_questions/natural_questions_val_fold_0_1k"
NQ_VAL_FOLD_1_PATH = "data/natural_questions/natural_questions_val_fold_1_1k"


def load_csv(dataset_name, debug):
    # input file is in csv format, can be loaded by pandas
    # required columns: [prompt] only

    if dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext")["validation"]
    elif dataset_name == "natural_questions":
        dataset = load_dataset("nq_open")["validation"]
    elif "triviaqa_val_fold_0" == dataset_name:
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_0_PATH)
    elif "triviaqa_val_fold_1" == dataset_name:
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_1_PATH)
    elif "natural_questions_val_fold_0" == dataset_name:
        dataset = Dataset.load_from_disk(NQ_VAL_FOLD_0_PATH)
    elif "natural_questions_val_fold_1" == dataset_name:
        dataset = Dataset.load_from_disk(NQ_VAL_FOLD_1_PATH)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    list_data = list(dataset["question"])
    labels = list(dataset["answer"])

    if debug:
        limit = 100
        print(f"WARNING! Debug with {limit} samples only.", flush=True)
        list_data = list_data[:limit]
        labels = labels[:limit]
    return list_data, labels


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def build_prompt(question_text, prompt_style="zero_shot"):
    # this prompt is designed for trivia QA
    if prompt_style == "zero_shot":
        question_text_prompt = "Answer the following question concisely.\n"
        question_text_prompt += f"Q:{question_text}\nA:"
    elif prompt_style == "few_shot":
        # question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt = f"Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n"
        # question_text_prompt += f'Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\n\n'
        question_text_prompt += (
            f"Q: Where in England was Dame Judi Dench born?\nA: York\n\n"
        )
        question_text_prompt += f"Q: {question_text}\nA: "
    elif prompt_style == "zero_shot_w_instru":
        raise NotImplementedError("zero_shot_w_instru Not implemented yet.")
    return question_text_prompt


def parse_python_exp(value):
    # This evaluates the string as Python code
    result = eval(value)
    return list(result) if isinstance(result, (range, list, tuple)) else [result]


def run(
    dataset_name, dataset, labels, model, tokenizer, generate_kwargs, stop_word_list
):
    device = model.device
    prompt_style = "few_shot"

    result_dict = {
        "qid_list": [],
        "answers": {},
        "model_completion": {},
        "questions": {},
        "logit_scores": {},
    }

    for i, question in enumerate(dataset):
        answer = labels[i]

        prompt = build_prompt(question, prompt_style)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        outputs = model.generate(input_ids, **generate_kwargs)

        sequences, scores = outputs.sequences, outputs.scores

        gen_sequences = sequences[:, input_ids.shape[-1] :][0, :]

        model_completion = tokenizer.decode(gen_sequences, skip_special_tokens=True)

        logit_scores = 0

        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        if "Q:" in model_completion:
            model_completion = model_completion.split("Q:")[0].strip()
        model_completion = model_completion.strip()

        result_dict["qid_list"].append(i)
        result_dict["answers"][i] = answer
        result_dict["model_completion"][i] = model_completion
        result_dict["questions"][i] = question
        result_dict["logit_scores"][i] = logit_scores

    # evaluation
    do_rating = True
    if do_rating:
        from benchmark.triviaqa_util import evaluate_triviaqa
        from benchmark.triviaqa_util import evaluate_nq

        ground_truth = result_dict["answers"]
        predicted_answers = result_dict["model_completion"]
        qid_list = result_dict["qid_list"]
        if dataset_name in [
            "triviaqa",
            "triviaqa_val_fold_0",
            "triviaqa_val_fold_1",
        ]:
            eval_metrics = evaluate_triviaqa(
                ground_truth, predicted_answers, qid_list=qid_list, mute=False
            )
        elif dataset_name in [
            "natural_questions",
            "natural_questions_val_fold_0",
            "natural_questions_val_fold_1",
        ]:
            eval_metrics = evaluate_nq(
                ground_truth, predicted_answers, qid_list=qid_list, mute=False
            )
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

        if "error_id" in eval_metrics:
            error_id_list = eval_metrics["error_id"]
            del eval_metrics["error_id"]
            eval_metrics["num_error"] = len(error_id_list)

            error_samples = {}
            for id in error_id_list:
                question = result_dict["questions"][id]
                answer = (
                    result_dict["answers"][id]["normalized_aliases"]
                    if dataset_name == "triviaqa"
                    else result_dict["answers"][id]
                )
                prediction = result_dict["model_completion"][id]
                error_sample = {
                    "Q": question,
                    "model_prediction": prediction,
                    "A": answer,
                    "correct": 0,
                }
                error_samples[id] = error_sample

            correct_samples = {}
            for id in qid_list:
                if id not in error_id_list:
                    question = result_dict["questions"][id]
                    answer = (
                        result_dict["answers"][id]["normalized_aliases"]
                        if dataset_name == "triviaqa"
                        else result_dict["answers"][id]
                    )
                    prediction = result_dict["model_completion"][id]
                    correct_sample = {
                        "Q": question,
                        "model_prediction": prediction,
                        "A": answer,
                        "correct": 1,
                    }
                    correct_samples[id] = correct_sample

        exact_match_acc = eval_metrics["exact_match"]
        f1 = eval_metrics["f1"]

    return eval_metrics


def prepare(
    model_name,
    head_indices,
    layer_indices,
    cd_temperature,
    max_new_tokens,
    output_hidden_states=False,
    output_logits=False,
):
    if head_indices is None and layer_indices is None:
        print("Preparing baseline model and tokenizer...")
        return prepare_baseline(model_name, max_new_tokens)

    if head_indices is None:
        raise Exception("head_indices should be provided")
    if layer_indices is None:
        raise Exception("layer_indices should be provided")

    device = "cuda"
    from utils.cd_utils import monkeypatch_cd

    monkeypatch_cd()

    from transformers.models.llama_modified import (
        LlamaForCausalLM as ModifiedLlamaForCausalLM,
    )

    assistant_model = ModifiedLlamaForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=torch.bfloat16
    )

    assistant_model = assistant_model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    stop_word_list = ["Q:", "\n"]
    set_stop_words(assistant_model, tokenizer, stop_word_list)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        assistant_model=assistant_model,
        assistant_model_two=None,
        cd_temperature=cd_temperature,
        layer_indices=layer_indices,
        head_indices=None,
        repetition_penalty=1.2,
        return_dict_in_generate=True,
        stopping_criteria=assistant_model.stopping_criteria,
        output_hidden_states=output_hidden_states,
        output_logits=output_logits,
    )

    return assistant_model, tokenizer, generate_kwargs, stop_word_list


def prepare_baseline(model_name, max_new_tokens):
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    stop_word_list = ["Q:", "\n"]
    set_stop_words(model, tokenizer, stop_word_list)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        assistant_model=None,
        assistant_model_two=None,
        cd_temperature=None,
        layer_indices=None,
        head_indices=None,
        repetition_penalty=1.2,
        return_dict_in_generate=True,
        stopping_criteria=model.stopping_criteria,
        output_hidden_states=False,
        output_logits=False,
    )

    return model, tokenizer, generate_kwargs, stop_word_list


def evaluate(
    model_name: str,
    dataset_name: str,
    data_path: str,
    head_indices: str,
    layer_indices: str,
    cd_temperature: float,
    max_new_tokens: int = 256,
):
    output_hidden_states = False
    output_logits = False

    # load dataset
    list_data_dict, labels = load_csv(dataset_name, debug=False)

    assistant_model, tokenizer, generate_kwargs, stop_word_list = prepare(
        model_name=model_name,
        head_indices=head_indices,
        layer_indices=layer_indices,
        cd_temperature=cd_temperature,
        max_new_tokens=max_new_tokens,
        output_hidden_states=output_hidden_states,
        output_logits=output_logits,
    )

    if "fold_0" in dataset_name:
        try:
            permute_idx = np.load(os.path.join(data_path, "fold_0_val_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(data_path, "fold_0_val_idx_{}.npy"), permute_idx)
    elif "fold_1" in dataset_name:
        try:
            permute_idx = np.load(os.path.join(data_path, "fold_1_val_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(data_path, "fold_1_val_idx_{}.npy"), permute_idx)
    else:
        try:
            permute_idx = np.load(os.path.join(data_path, "val_test_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(data_path, "val_test_idx_{}.npy"), permute_idx)

    eval_metrics = run(
        dataset_name=dataset_name,
        dataset=list_data_dict,
        labels=labels,
        model=assistant_model,
        tokenizer=tokenizer,
        generate_kwargs=generate_kwargs,
        stop_word_list=stop_word_list,
    )

    eval_metrics_ = dict(
        exact_match=eval_metrics["exact_match"],
        f1=eval_metrics["f1"],
        score=eval_metrics["exact_match"] * eval_metrics["f1"],
    )

    return eval_metrics_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="natural_questions_val_fold_0"
    )
    parser.add_argument("--data-path", type=str, default="data/natural_questions")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_path = args.data_path

    cd_temperature = 1.0
    max_new_tokens = 256

    config = AutoConfig.from_pretrained(model_name)
    model_tag = model_name.split("/")[-1]

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    all_results = []

    baseline_eval_metrics = evaluate(
        model_name=model_name,
        dataset_name=dataset_name,
        data_path=data_path,
        head_indices=None,
        layer_indices=None,
        cd_temperature=None,
        max_new_tokens=max_new_tokens,
    )

    print("baseline_eval_metrics")
    print(baseline_eval_metrics)

    out_path = (
        f"head_divs_results_{dataset_name}/head_downstream_results_{model_tag}.json"
    )
    print("Output will be saved here:", out_path)

    for layer_idx in tqdm(range(0, num_layers)):
        for head_idx in range(0, num_heads):
            print(
                f"Running: layer: {layer_idx}, head: {head_idx}", end="\r", flush=True
            )

            result_dict = {}
            os.environ["LAYERS_TO_ABLATE"] = f"{layer_idx}"
            os.environ["HEADS_TO_ABLATE"] = f"{head_idx}"

            eval_metrics = evaluate(
                model_name=model_name,
                dataset_name=dataset_name,
                data_path=data_path,
                head_indices=head_idx,
                layer_indices=layer_idx,
                cd_temperature=cd_temperature,
                max_new_tokens=max_new_tokens,
            )

            result_dict["exact_match"] = eval_metrics["exact_match"]
            result_dict["f1"] = eval_metrics["f1"]
            result_dict["score"] = eval_metrics["score"]

            result_dict["exact_match_delta"] = (
                baseline_eval_metrics["exact_match"] - eval_metrics["exact_match"]
            )
            result_dict["f1_delta"] = baseline_eval_metrics["f1"] - eval_metrics["f1"]
            result_dict["score_delta"] = (
                baseline_eval_metrics["score"] - eval_metrics["score"]
            )

            result_dict["layer_idx"] = layer_idx
            result_dict["head_idx"] = head_idx

            all_results.append(result_dict)

    with open(out_path, "w+") as f:
        json.dump(all_results, f)

    print("Output saved here:", out_path)
