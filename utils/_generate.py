import torch
from ._tokenizer import get_tokenizer


class Generator:
    def __init__(self, model):
        self.model = model
        self.tokenizer = get_tokenizer()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, start_text, max_len=100, temperature=1.0, top_k=5):
        self.model.eval()
        token_ids = self.tokenizer.encode(start_text, add_special_tokens=False)
        token_ids.insert(0, self.tokenizer.bos_token_id)

        for _ in range(max_len):
            input_ids = torch.tensor(
                token_ids[-max_len:],
                device=self.device
            ).unsqueeze(0)

            outputs = self.model(input_ids)
            logits = outputs[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probabilities, 1)]

            token_ids.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(
            token_ids, skip_special_tokens=True
        )
        return generated_text
