import torch
import torch.nn as nn


class GridworldTransformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(context_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = (input_ids == self.pad_id)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        logits = self.output_projection(x)
        return logits
