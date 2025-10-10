from transformers import PreTrainedTokenizerFast


def get_tokenizer(fpath_tokenizer="tokenizer.json"):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=fpath_tokenizer)
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    })
    return tokenizer


def train_tokenizer(text, fpath_tokenizer="tokenizer.json", vocab_size=16000):
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.normalizers import NFKC
    from tokenizers.pre_tokenizers import UnicodeScripts
    from tokenizers.decoders import Metaspace
    from tokenizers.trainers import UnigramTrainer
    from tokenizers.processors import TemplateProcessing

    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = UnicodeScripts()
    tokenizer.decoder = Metaspace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="<unk>",
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.add_special_tokens()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.save(fpath_tokenizer)
