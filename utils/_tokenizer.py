from transformers import PreTrainedTokenizerFast


def get_tokenizer(fpath_tokenizer="tokenizer.json"):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=fpath_tokenizer)
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    })
    return tokenizer
