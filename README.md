# Deep Learning Scratch

深層学習モデルをスクラッチ実装していく。

## 準備

```
uv sync
uv run python prepare.py
```

## 学習

```
uv run torchrun --nproc_per_node=1 train.py
```

## テキスト生成

`demo.ipynb`
