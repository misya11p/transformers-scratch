# Implementing Transformer Models from Scratch

Transformer系モデルをスクラッチ実装し、理解を深める。

## 実装済みモデル

### Vanilla Transformer

- [Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017, pp. 5998–6008. arXiv:1706.03762.](https://arxiv.org/abs/1706.03762)
- Implementation: [models/vanilla_transformer.py](models/vanilla_transformer.py)

### GPT-2

- [Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Implementation: [models/gpt2.py](models/gpt2.py)

### Vision Transformer

- [Dosovitskiy, Alexey, et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *arXiv preprint arXiv:2010.11929* (2021).](https://arxiv.org/abs/2010.11929)
- Implementation: [models/vision_transformer.py](models/vision_transformer.py)

### CLIP

- [Radford, Alec, et al. "Learning Transferable Visual Models from Natural Language Supervision." *arXiv* preprint arXiv:2103.00020, 2021.](https://arxiv.org/abs/2103.00020)
- Implementation: [models/clip.py](models/clip.py)

## プログラム実行

*下記学習プログラムはVanilla Transformer、GPT-2のみ対応。

### 準備

```
uv sync
uv run python train_tokenizer.py
```

### 学習

```
uv run torchrun --nproc_per_node=1 train.py -c gpt2
```
