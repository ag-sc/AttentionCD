
## Method: **ActD**

[In-context sharpness as alerts: An inner representation perspective for hallucination mitigation](https://arxiv.org/abs/2403.01548) (Chen et al., 2024)

- **info_layer**: decoder layer used for calculating activations - {12, 14, 16} for Llama-3.2-1B-Instruct, {22, 24, 26, 28} for Llama-3.2-3B-Instruct and {26, 28, 30, 32} for Llama-3.1-8B-Instruct; 
- **alpha**: scaling factor for adjusting the next token probability distribution - {0.2, 0.4, 0.6, 0.8}.

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


- **rel_top**: threshold for applying plausibility constraint while contrastive decoding - {0.1, 0.01, 0.001}; 
- **alpha**: scaling factor for adjusting the next token probability distribution - for TruthfulQA-MC, Natural Questions, and TriviaQA {0.25, 0.375, 0.5} and for TruthfulQA-Gen {1.0, 2.0, 3.0}.

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
Ref: [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883) (Chuang et al., 2024)

- **dola_layers**: range of decoder layers to consider as candidate premature layers. We use either high layers - the upper half or low - the lower half of the decoder layer; 
- **alpha**: always set to 1.0


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


- **k**: number of layers pruned - [1, 4] for Llama-3.2-1B-Instruct, [3, 6] for Llama-3.2-3B-Instruct, and [3, 8] for Llama-3.1-8B-Instruct; 
- **alpha**: [1.0, 1.5] for TruthfulQA and [0.1, 0.3] for other datasets.


| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | PruneCD | TfQA-MC | k=4, alpha=1.0 |
| Llama-3.2-3B-Instruct | PruneCD | TfQA-MC | k=3, alpha=1.0 |
| Llama-3.1-8B-Instruct | PruneCD | TfQA-MC | k=6, alpha=1.0 |
| Llama-3.2-1B-Instruct | PruneCD | TfQA-Gen | k=3, alpha=1.50 |
| Llama-3.2-3B-Instruct | PruneCD | TfQA-Gen | k=3, alpha=1.50 |
| Llama-3.1-8B-Instruct | PruneCD | TfQA-Gen | k=4, alpha=1.50 |
| Llama-3.2-1B-Instruct | PruneCD | NQ | k=4, alpha=0.15 |
| Llama-3.2-3B-Instruct | PruneCD | NQ | k=4, alpha=0.10 |
| Llama-3.1-8B-Instruct | PruneCD | NQ | k=4, alpha=0.25 |
| Llama-3.2-1B-Instruct | PruneCD | TriviaQA | k=4, alpha=0.15 |
| Llama-3.2-3B-Instruct | PruneCD | TriviaQA | k=3, alpha=0.10 |
| Llama-3.1-8B-Instruct | PruneCD | TriviaQA | k=6, alpha=0.25 |


## Method: **DeCoRe**
[DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations](https://aclanthology.org/2025.findings-emnlp.531/) (Gema et al., Findings 2025)


- **k**: percentage of retrieval heads to mask globally from the model;  
- **alpha**: uses an entropy-guided dynamic CD temperature.

| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | DeCoRe | TfQA-MC | k=50 |
| Llama-3.2-3B-Instruct | DeCoRe | TfQA-MC | k=5 |
| Llama-3.1-8B-Instruct | DeCoRe | TfQA-MC | k=10 |
| Llama-3.2-1B-Instruct | DeCoRe | TfQA-Gen | k=100 |
| Llama-3.2-3B-Instruct | DeCoRe | TfQA-Gen | k=20 |
| Llama-3.1-8B-Instruct | DeCoRe | TfQA-Gen | k=25 |
| Llama-3.2-1B-Instruct | DeCoRe | NQ | k=100 |
| Llama-3.2-3B-Instruct | DeCoRe | NQ | k=100 |
| Llama-3.1-8B-Instruct | DeCoRe | NQ | k=1 |
| Llama-3.2-1B-Instruct | DeCoRe | TriviaQA | k=100 |
| Llama-3.2-3B-Instruct | DeCoRe | TriviaQA | k=100 |
| Llama-3.1-8B-Instruct | DeCoRe | TriviaQA | k=20 |



## Method: **AttentionCD-JSD**

- **k**: percentage of task-specific heads to mask from each layer of the model;
- **layers**: range of decoder layers where attention heads should be masked. We use either High layers - the upper half, Low - the lower half, or All decoder layers in the model; 
- **alpha**: uses an entropy-guided dynamic CD temperature for TruthfulQA-MC, {1.0, 1.5, 2.0} for TruthfulQA-Gen, a static {0.1} for Natural Questions and {0.1, 0.5} for TriviaQA.


| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | AttentionCD-JSD | TfQA-MC     | k=25, layers=(7, 15) |
| Llama-3.2-3B-Instruct | AttentionCD-JSD | TfQA-MC     | k=50, layers=(0, 13) |
| Llama-3.1-8B-Instruct | AttentionCD-JSD | TfQA-MC     | k=15, layers=(15, 31) |
| Llama-3.2-1B-Instruct | AttentionCD-JSD | TfQA-Gen    | k=10, layers=(0, 7), alpha=1.0 |
| Llama-3.2-3B-Instruct | AttentionCD-JSD | TfQA-Gen    | k=5, layers=(0, 13), alpha=2.0 |
| Llama-3.1-8B-Instruct | AttentionCD-JSD | TfQA-Gen    | k=5, layers=(0, 15), alpha=2.0 |
| Llama-3.2-1B-Instruct | AttentionCD-JSD | NQ          | k=10, layers=(0, 7), alpha=0.1 |
| Llama-3.2-3B-Instruct | AttentionCD-JSD | NQ          | k=25, layers=(0, 27), alpha=0.1 |
| Llama-3.1-8B-Instruct | AttentionCD-JSD | NQ          | k=1, layers=(0, 31), alpha=0.1 |
| Llama-3.2-1B-Instruct | AttentionCD-JSD | TriviaQA    | k=5, layers=(0, 7), alpha=0.1 |
| Llama-3.2-3B-Instruct | AttentionCD-JSD | TriviaQA    | k=50, layers=(0, 13), alpha=0.1 |
| Llama-3.1-8B-Instruct | AttentionCD-JSD | TriviaQA    | k=75, layers=(15, 31), alpha=0.1 |


## Method: **AttentionCD-DS**

- **k**: percentage of task-specific heads to mask from each layer of the model;
- **layers**: range of decoder layers where attention heads should be masked. We use either High layers - the upper half, Low - the lower half, or All decoder layers in the model; 
- **alpha**: uses an entropy-guided dynamic CD temperature for TruthfulQA-MC, {1.0, 1.5, 2.0} for TruthfulQA-Gen, a static {0.1} for Natural Questions and {0.1, 0.5} for TriviaQA.


| Model | Method | Dataset | Hyper-parameters |
| :--- | :--- | :--- | :--- |
| Llama-3.2-1B-Instruct | AttentionCD-DS | TfQA-MC  | k=10, layers=(0, 7) |
| Llama-3.2-3B-Instruct | AttentionCD-DS | TfQA-MC  | k=1, layers=(0, 13) |
| Llama-3.1-8B-Instruct | AttentionCD-DS | TfQA-MC  | k=5, layers=(15, 31) |
| Llama-3.2-1B-Instruct | AttentionCD-DS | TfQA-Gen | k=100, layers=(0, 15), alpha=1.5 |
| Llama-3.2-3B-Instruct | AttentionCD-DS | TfQA-Gen | k=25, layers=(13, 27), alpha=2.0 |
| Llama-3.1-8B-Instruct | AttentionCD-DS | TfQA-Gen | k=20, layers=(15, 31), alpha=2.0 |
| Llama-3.2-1B-Instruct | AttentionCD-DS | NQ       | k=10, layers=(0, 7), alpha=0.1 |
| Llama-3.2-3B-Instruct | AttentionCD-DS | NQ       | k=50, layers=(13, 27), alpha=0.1 |
| Llama-3.1-8B-Instruct | AttentionCD-DS | NQ       | k=25, layers=(0, 31), alpha=0.1 |
| Llama-3.2-1B-Instruct | AttentionCD-DS | TriviaQA | k=25, layers=(7, 15), alpha=0.1 |
| Llama-3.2-3B-Instruct | AttentionCD-DS | TriviaQA | k=5, layers=(13, 27), alpha=0.5 |
| Llama-3.1-8B-Instruct | AttentionCD-DS | TriviaQA | k=1, layers=(15, 31), alpha=0.5 |