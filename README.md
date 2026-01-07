# Implementing Transformer Models from Scratch

Transformer系モデルをスクラッチ実装し、理解を深める。

## 実装

- Vanilla Transformer[^1]: [models/vanilla_transformer.py](models/vanilla_transformer.py)
- GPT-2[^2]: [models/gpt2.py](models/gpt2.py)
- Vision Transformer[^3]: [models/vision_transformer.py](models/vision_transformer.py)
- CLIP[^4]: [models/clip.py](models/clip.py)
- 学習コード（言語モデル学習のみ）: [train.py](train.py)


[^1]: [Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017, pp. 5998–6008. arXiv:1706.03762.](https://arxiv.org/abs/1706.03762)
[^2]: [Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
[^3]: [Dosovitskiy, Alexey, et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *arXiv preprint arXiv:2010.11929* (2021).](https://arxiv.org/abs/2010.11929)
[^4]: [Radford, Alec, et al. "Learning Transferable Visual Models from Natural Language Supervision." *arXiv* preprint arXiv:2103.00020, 2021.](https://arxiv.org/abs/2103.00020)

## プログラム実行

```
uv sync
```

### トークナイザー学習

[学習済みのTokenizer](trained/tokenizer.json)は置いてあるのでやらなくてもいい。

```
uv run python train_tokenizer.py
```

### 学習

`config/`にモデルの設定をtoml形式で記述し、ファイル名を指定する。

```
uv run torchrun --nproc_per_node=1 train.py -c gpt2
```

### デモ

学習下モデルを使って実際に文章生成を試す。

[playground.ipynb](playground.ipynb)
