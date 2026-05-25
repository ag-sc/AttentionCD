# AttentionCD: Reducing Factuality Hallucinations in Large Language Models using Task-specific Head-based Contrastive Decoding


<div align="center">
  <img src="figures/example_fig.png" alt="AttentionCD" width="350">
</div>

Despite the tremendous volume of data they are trained on, Large Language Models (LLMs) can generate plausibly sounding but non-factual responses. These factually hallucinated responses reduce users’ trust in such models and hinder their adoption in safety-critical applications. Among the many training-free hallucination mitigation strategies proposed, contrastive decoding strategies have received wide attention. Most of the work in this direction explores how contrasting probability distributions from different decoder layers can help in reducing factuality hallucinations. This work, in contrast, investigates whether masking of attention heads can provide reliable contrastive signals. 

AttentionCD is a decoding strategy that masks task-specific attention heads to yield better contrastive signals to reduce factuality hallucinations in LLMs. Our approach shows competitive performance with the existing layer-based contrastive decoding strategies and offers an average performance improvement of over +20\% over greedy decoding across different tasks and models.


> We extend the code-base from the authors of PruneCD: https://github.com/hoeng4/PruneCD/tree/main. Thank you for your open-source contribution!


## 1. Task-specific Head Discovery

### 1.1. Run JSD Heads Selection

### 1.2. Run DS Heads Selection

## 2. Hyper-parameter Sweep

## 3. Evaluation Runs