# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import random
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
import time

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
    elif "triviaqa_val_fold_0" in dataset_name:
        print("WARNING: Using fold 0 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_0_PATH)
    elif "triviaqa_val_fold_1" in dataset_name:
        print("WARNING: Using fold 1 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(TRIVIAQA_VAL_FOLD_1_PATH)
    elif "natural_questions_val_fold_0" in dataset_name:
        print("WARNING: Using fold 0 val split, for hyperparameter sweep.")
        dataset = Dataset.load_from_disk(NQ_VAL_FOLD_0_PATH)
    elif "natural_questions_val_fold_1" in dataset_name:
        print("WARNING: Using fold 1 val split, for hyperparameter sweep.")
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
    if value is None:
        return None
    # This evaluates the string as Python code
    result = eval(value)
    return list(result) if isinstance(result, (range, list, tuple)) else [result]


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    #############################################################################################
    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "prunecd", "dola", "actd", "end", "att_cd"],
    )

    # PruneCD
    parser.add_argument("--sleb_list", type=str, default=None)
    parser.add_argument("--cd_temperature", type=float, default=None)

    # AttCD
    parser.add_argument("--layer_indices", type=parse_python_exp, default=None)
    parser.add_argument("--head_indices", type=str, default=None)

    # DoLa
    parser.add_argument("--dola_layers", type=str, default=None)

    # Activation Decoding & END
    parser.add_argument("--info_layer", type=int, default=26)  # llama-2-7b-chat : 26
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    #############################################################################################
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "triviaqa",
            "natural_questions",
            "triviaqa_val_fold_0",
            "triviaqa_val_fold_1",
            "natural_questions_val_fold_0",
            "natural_questions_val_fold_1",
        ],
        default="triviaqa",
    )
    parser.add_argument("--data-path", type=str, default="../scripts/data/nq")

    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["zero_shot", "few_shot", "zero_shot_w_instru"],
        default="few_shot",
    )
    parser.add_argument("--debug", action="store_true")
    #############################################################################################

    args = parser.parse_args()
    model_name = args.model_name
    device = args.device

    if "val" in args.dataset_name:
        print(f"WARNING! Using train split for hyperparmeter sweep.")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    stop_word_list = ["Q:", "\n"]
    set_stop_words(model, tokenizer, stop_word_list)

    output_hidden_states = False
    output_logits = False

    mode = args.mode
    if mode == "greedy":
        print("MODE: naive decoding from the last layer", flush=True)
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            stopping_criteria=model.stopping_criteria,
            repetition_penalty=1.2,
        )

    elif mode == "prunecd":
        if "Llama" in model_name:
            from utils.prunecd_utils import monkeypatch_prunecd_llama_efficient

            monkeypatch_prunecd_llama_efficient(model, args.sleb_list)
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                cd_temperature=args.cd_temperature,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                stopping_criteria=model.stopping_criteria,
                output_hidden_states=output_hidden_states,
                output_logits=output_logits,
                layer_indices=None,
            )

        elif "mistral" in model_name:
            import torch

            assistant_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )

            with open(args.sleb_list, "r") as f:
                lines = f.readlines()
            skip_list = []
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    skip_list = list(map(int, line.split(",")))

            with torch.no_grad():
                kept_layers = [
                    layer
                    for i, layer in enumerate(assistant_model.model.layers)
                    if i not in skip_list
                ]
                assistant_model.model.layers = torch.nn.ModuleList(kept_layers)
                assistant_model.config.num_hidden_layers = len(
                    assistant_model.model.layers
                )

            print("=> New layer count:", len(assistant_model.model.layers))
            print(assistant_model)
            from utils.prunecd_utils import monkeypatch_cd

            monkeypatch_cd()
            assistant_model = assistant_model.to(model.device)
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                assistant_model=assistant_model,
                cd_temperature=args.cd_temperature,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                stopping_criteria=model.stopping_criteria,
                output_hidden_states=output_hidden_states,
                output_logits=output_logits,
                layer_indices=None,
            )

    elif mode == "att_cd":
        print(
            f"MODE: Attention CD with head_indices: {args.head_indices}, layer_indices: {args.layer_indices}, with cd_temp {args.cd_temperature}",
            flush=True,
        )
        if args.head_indices is None:
            raise Exception("head_indices should be provided")
        if args.layer_indices is None:
            raise Exception("layer_indices should be provided")

        from custom_models.modeling_llama import (
            LlamaForCausalLM as ModifiedLlamaForCausalLM,
        )

        assistant_model = ModifiedLlamaForCausalLM.from_pretrained(
            model_name, attn_implementation="eager", torch_dtype=torch.bfloat16
        )

        from utils.cd_utils import monkeypatch_cd

        monkeypatch_cd()
        assistant_model = assistant_model.to(model.device)
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            assistant_model=assistant_model,
            assistant_model_two=None,
            cd_temperature=args.cd_temperature,
            layer_indices=args.layer_indices,
            head_indices=args.head_indices,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            stopping_criteria=model.stopping_criteria,
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
        )

    elif mode == "dola":
        print(f"MODE: dola {args.dola_layers} decoding", flush=True)
        from utils.cd_utils import monkeypatch_cd

        monkeypatch_cd()
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            dola_layers=args.dola_layers,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            stopping_criteria=model.stopping_criteria,
        )

    elif mode == "actd":
        from utils.cd_utils import monkeypatch_cd

        monkeypatch_cd()
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            info_layer=args.info_layer,
            alpha=args.alpha,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            stopping_criteria=model.stopping_criteria,
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
        )

    elif mode == "end":
        from utils.cd_utils import monkeypatch_cd

        monkeypatch_cd()
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            alpha=args.alpha,
            relative_top=args.relative_top,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            stopping_criteria=model.stopping_criteria,
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
        )

    # load dataset
    list_data_dict, labels = load_csv(args.dataset_name, args.debug)

    result_dict = {
        "qid_list": [],
        "answers": {},
        "model_completion": {},
        "questions": {},
        "logit_scores": {},
    }

    print("Begin inference...\n")
    print("***Hyperparameters***:", args)
    print("\nSample prompt: \n", build_prompt(list_data_dict[0], args.prompt_style))
    print("*" * 20)
    print("\n\n")

    os.makedirs(args.data_path, exist_ok=True)

    if "fold_0" in args.dataset_name:
        try:
            permute_idx = np.load(os.path.join(args.data_path, "fold_0_val_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(args.data_path, "fold_0_val_idx_{}.npy"), permute_idx)
    elif "fold_1" in args.dataset_name:
        try:
            permute_idx = np.load(os.path.join(args.data_path, "fold_1_val_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(args.data_path, "fold_1_val_idx_{}.npy"), permute_idx)
    else:
        try:
            permute_idx = np.load(os.path.join(args.data_path, "val_test_idx_{}.npy"))
        except:
            permute_idx = np.random.permutation(len(list_data_dict))
            np.save(os.path.join(args.data_path, "val_test_idx_{}.npy"), permute_idx)

    dataset = list_data_dict

    all_logits = []

    for i, question in enumerate(tqdm(dataset, disable=args.disable_tqdm)):
        answer = labels[i]

        prompt = build_prompt(question, args.prompt_style)
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

    model_tag = (
        model_name.split("/")[-1]
        if model_name[-1] != "/"
        else model_name.split("/")[-2]
    )

    output_file = args.output_path
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Saving results to {args.output_path}")
    print("Begin evaluation...")

    # evaluation
    args.do_rating = True
    if args.do_rating:
        from triviaqa_util import evaluate_triviaqa
        from triviaqa_util import evaluate_nq
        import json

        ground_truth = result_dict["answers"]
        predicted_answers = result_dict["model_completion"]
        qid_list = result_dict["qid_list"]
        if args.dataset_name in [
            "triviaqa",
            "triviaqa_val_fold_0",
            "triviaqa_val_fold_1",
            "triviaqa_val_split",
        ]:
            eval_metrics = evaluate_triviaqa(
                ground_truth, predicted_answers, qid_list=qid_list, mute=False
            )
        elif args.dataset_name in [
            "natural_questions",
            "natural_questions_val_fold_0",
            "natural_questions_val_fold_1",
            "natural_questions_val_split",
        ]:
            eval_metrics = evaluate_nq(
                ground_truth, predicted_answers, qid_list=qid_list, mute=False
            )
        else:
            raise NotImplementedError(
                f"Dataset {args.dataset_name} not implemented yet."
            )

        if "error_id" in eval_metrics:
            error_id_list = eval_metrics["error_id"]
            del eval_metrics["error_id"]
            eval_metrics["num_error"] = len(error_id_list)

            error_samples = {}
            for id in error_id_list:
                question = result_dict["questions"][id]
                answer = (
                    result_dict["answers"][id]["normalized_aliases"]
                    if args.dataset_name == "triviaqa"
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
                        if args.dataset_name == "triviaqa"
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
            final_samples = {
                "error_samples": error_samples,
                "correct_samples": correct_samples,
            }
            with open(output_file.replace(".json", "_results.json"), "w") as f:
                json.dump(final_samples, f, indent=4)

        exact_match_acc = eval_metrics["exact_match"]
        f1 = eval_metrics["f1"]
        print(f"Exact Match:{exact_match_acc:.5f}, F1:{f1:.5f}")

        eval_metrics["model_name"] = model_name
        eval_metrics["dataset"] = "triviaqa"
        eval_metrics["mode"] = mode
        eval_metrics["parameters"] = vars(args)
        eval_metrics["sample_prompt"] = build_prompt(
            list_data_dict[0], args.prompt_style
        )
        with open(output_file.replace(".json", "_rating.json"), "w") as f:
            json.dump(eval_metrics, f, indent=4)
