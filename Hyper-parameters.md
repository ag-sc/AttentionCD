
## Method: **ActD**

[In-context sharpness as alerts: An inner representation perspective for hallucination mitigation](https://arxiv.org/abs/2403.01548) (Chen et al., 2024)

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | ActD | TfQA-MC | info_layer=16, alpha=0.6 |
| Llama-3.2-3B-Instruct | ActD | TfQA-MC | info_layer=28, alpha=0.8 |
| Llama-3.1-8B-Instruct | ActD | TfQA-MC | info_layer=32, alpha=0.8 |
| Llama-3.2-1B-Instruct | ActD | TfQA-Gen | info_layer=16, alpha=0.8 |
| Llama-3.2-3B-Instruct | ActD | TfQA-Gen | info_layer=22, alpha=0.8 |
| Llama-3.1-8B-Instruct | ActD | TfQA-Gen | info_layer=32, alpha=0.8 |
| Llama-3.2-1B-Instruct | ActD | NQ | info_layer=16, alpha=0.2 |
| Llama-3.2-3B-Instruct | ActD | NQ | info_layer=28, alpha=0.2 |
| Llama-3.1-8B-Instruct | ActD | NQ | info_layer=30, alpha=0.2 |
| Llama-3.2-1B-Instruct | ActD | TriviaQA | info_layer=16, alpha=0.6 |
| Llama-3.2-3B-Instruct | ActD | TriviaQA | info_layer=28, alpha=0.8 |
| Llama-3.1-8B-Instruct | ActD | TriviaQA | info_layer=32, alpha=0.8 |



## Method: **END**

[Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models](https://aclanthology.org/2025.findings-naacl.217/) (Wu et al., Findings 2025)

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | END | TfQA-MC | rel_top=0.001, alpha=0.5 |
| Llama-3.2-3B-Instruct | END | TfQA-MC | rel_top=0.001, alpha=0.25 |
| Llama-3.1-8B-Instruct | END | TfQA-MC | rel_top=0.001, alpha=0.25 |
| Llama-3.2-1B-Instruct | END | TfQA-Gen | rel_top=0.01, alpha=3.0 |
| Llama-3.2-3B-Instruct | END | TfQA-Gen | rel_top=0.01, alpha=1.0 |
| Llama-3.1-8B-Instruct | END | TfQA-Gen | rel_top=0.01, alpha=2.0 |
| Llama-3.2-1B-Instruct | END | NQ | rel_top=0.1, alpha=0.5 |
| Llama-3.2-3B-Instruct | END | NQ | rel_top=0.001, alpha=0.375 |
| Llama-3.1-8B-Instruct | END | NQ | rel_top=0.001, alpha=0.5 |
| Llama-3.2-1B-Instruct | END | TriviaQA | rel_top=0.001, alpha=0.375 |
| Llama-3.2-3B-Instruct | END | TriviaQA | rel_top=0.01, alpha=0.25 |
| Llama-3.1-8B-Instruct | END | TriviaQA | rel_top=0.01, alpha=0.25 |



## Method: **DoLa**
[DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883) (Chuang et al., 2024)

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | DoLa | TfQA-MC | dola_layers=low |
| Llama-3.2-3B-Instruct | DoLa | TfQA-MC | dola_layers=low |
| Llama-3.1-8B-Instruct | DoLa | TfQA-MC | dola_layers=low |
| Llama-3.2-1B-Instruct | DoLa | TfQA-Gen | dola_layers=high |
| Llama-3.2-3B-Instruct | DoLa | TfQA-Gen | dola_layers=low |
| Llama-3.1-8B-Instruct | DoLa | TfQA-Gen | dola_layers=low |
| Llama-3.2-1B-Instruct | DoLa | NQ | dola_layers=low |
| Llama-3.2-3B-Instruct | DoLa | NQ | dola_layers=low |
| Llama-3.1-8B-Instruct | DoLa | NQ | dola_layers=high |
| Llama-3.2-1B-Instruct | DoLa | TriviaQA | dola_layers=low |
| Llama-3.2-3B-Instruct | DoLa | TriviaQA | dola_layers=high |
| Llama-3.1-8B-Instruct | DoLa | TriviaQA | dola_layers=high |



## Method: **PruneCD**
[PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality](https://aclanthology.org/2025.emnlp-main.1651/) (Yu et al., EMNLP 2025)

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | END | TfQA-MC | k=6, alpha=1.0 |
| Llama-3.2-3B-Instruct | END | TfQA-MC | k=3, alpha=1.0 |
| Llama-3.1-8B-Instruct | END | TfQA-MC | k=6, alpha=1.0 |
| Llama-3.2-1B-Instruct | END | TfQA-Gen | k=3, alpha=1.50 |
| Llama-3.2-3B-Instruct | END | TfQA-Gen | k=3, alpha=1.50 |
| Llama-3.1-8B-Instruct | END | TfQA-Gen | k=4, alpha=1.50 |
| Llama-3.2-1B-Instruct | END | NQ | k=7, alpha=0.15 |
| Llama-3.2-3B-Instruct | END | NQ | k=4, alpha=0.10 |
| Llama-3.1-8B-Instruct | END | NQ | k=4, alpha=0.25 |
| Llama-3.2-1B-Instruct | END | TriviaQA | k=7, alpha=0.15 |
| Llama-3.2-3B-Instruct | END | TriviaQA | k=3, alpha=0.10 |
| Llama-3.1-8B-Instruct | END | TriviaQA | k=6, alpha=0.25 |


## Method: **DeCoRe**
[DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations](https://aclanthology.org/2025.findings-emnlp.531/) (Gema et al., Findings 2025)

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | END | TfQA-MC | k=50 |
| Llama-3.2-3B-Instruct | END | TfQA-MC | k=5 |
| Llama-3.1-8B-Instruct | END | TfQA-MC | k=10 |
| Llama-3.2-1B-Instruct | END | TfQA-Gen | k=100 |
| Llama-3.2-3B-Instruct | END | TfQA-Gen | k=20 |
| Llama-3.1-8B-Instruct | END | TfQA-Gen | k=25 |
| Llama-3.2-1B-Instruct | END | NQ | k=100 |
| Llama-3.2-3B-Instruct | END | NQ | k=100 |
| Llama-3.1-8B-Instruct | END | NQ | k=1 |
| Llama-3.2-1B-Instruct | END | TriviaQA | k=100 |
| Llama-3.2-3B-Instruct | END | TriviaQA | k=100 |
| Llama-3.1-8B-Instruct | END | TriviaQA | k=20 |



## Method: **AttentionCD-JSD**

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | END | TfQA-MC | k=10, layers=(), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | TfQA-MC | k=50, layers=(), alpha=0.1 |
| Llama-3.1-8B-Instruct | END | TfQA-MC | k=50, layers=(), alpha=0.1 |
| Llama-3.2-1B-Instruct | END | TfQA-Gen | k=10, layers=(0, 7), alpha=1.0 |
| Llama-3.2-3B-Instruct | END | TfQA-Gen | k=5, layers=(0, 13), alpha=2.0 |
| Llama-3.1-8B-Instruct | END | TfQA-Gen | k=5, layers=(0, 15), alpha=2.0 |
| Llama-3.2-1B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.1-8B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.2-1B-Instruct | END | TriviaQA | k=5, layers=(0, 7), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | TriviaQA | k=50, layers=(0, 13), alpha=0.1 |
| Llama-3.1-8B-Instruct | END | TriviaQA | k=75, layers=(15, 31), alpha=0.1 |


## Method: **AttentionCD-DS**

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | END | TfQA-MC | k=10, layers=(), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | TfQA-MC | k=50, layers=(), alpha=0.1 |
| Llama-3.1-8B-Instruct | END | TfQA-MC | k=50, layers=(), alpha=0.1 |
| Llama-3.2-1B-Instruct | END | TfQA-Gen | k=100, layers=(0, 15), alpha=1.5 |
| Llama-3.2-3B-Instruct | END | TfQA-Gen | k=25, layers=(13, 27), alpha=2.0 |
| Llama-3.1-8B-Instruct | END | TfQA-Gen | k=20, layers=(15, 31), alpha=2.0 |
| Llama-3.2-1B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.1-8B-Instruct | END | NQ | k=50, layers=(), alpha=0.1 |
| Llama-3.2-1B-Instruct | END | TriviaQA | k=25, layers=(7, 15), alpha=0.1 |
| Llama-3.2-3B-Instruct | END | TriviaQA | k=5, layers=(13, 27), alpha=0.5 |
| Llama-3.1-8B-Instruct | END | TriviaQA | k=1, layers=(15, 31), alpha=0.5 |