from typing import Optional, Tuple

import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    LSTM-based caption decoder.

    - Takes image features from the encoder: (B, feat_dim)
    - Takes tokenized captions during training: (B, max_len)
    - Outputs vocabulary logits for each time step: (B, max_len, vocab_size)

    It also provides a `generate` method for greedy decoding at test time.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        feat_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # If not specified, assume encoder feature dim == embed_size
        self.feat_dim = feat_dim or embed_size

        # 1) Word embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx,
        )

        # 2) LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3) Map encoder feature -> initial hidden state
        #    (we'll use zeros for initial cell state c0)
        self.encoder2hidden = nn.Linear(self.feat_dim, hidden_size)

        # 4) Final projection from hidden state -> vocabulary logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    # ---------------------- helpers ----------------------

    def init_hidden(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from encoder features.

        features: (B, feat_dim)
        returns: h0, c0 each of shape (num_layers, B, hidden_size)
        """
        B = features.size(0)

        # Map feature to hidden dimension
        h0_single = torch.tanh(self.encoder2hidden(features))  # (B, hidden_size)

        # Repeat for each layer
        h0 = h0_single.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_size)
        c0 = torch.zeros_like(h0)  # initialize cell state to zeros

        return h0, c0

    # ---------------------- training forward ----------------------

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during TRAINING.

        features: (B, feat_dim)  - output of EncoderCNN
        captions: (B, max_len)   - token IDs, including <bos> and possibly <eos>

        Returns:
            logits: (B, max_len, vocab_size)

        NOTE: in the training loop (WP4), you'll typically shift captions
        so that:
          - input to the decoder is caption[:, :-1]
          - target for loss is caption[:, 1:]
        """
        # Initialize hidden state from encoder features
        h0, c0 = self.init_hidden(features)  # (num_layers, B, hidden_size)

        # Embed the input captions
        emb = self.embedding(captions)       # (B, max_len, embed_size)

        # Run LSTM
        outputs, _ = self.lstm(emb, (h0, c0))    # (B, max_len, hidden_size)

        # Project to vocabulary logits
        logits = self.fc(outputs)                # (B, max_len, vocab_size)

        return logits

    # ---------------------- inference (greedy) ----------------------

    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        max_len: int = 20,
    ) -> torch.Tensor:
        """
        Greedy decoding for INFERENCE.

        features: (B, feat_dim)
        returns:
            predicted_ids: (B, <= max_len)  - token IDs (no <bos>, may contain <eos>)
        """
        self.eval()

        device = features.device
        B = features.size(0)

        # Initialize hidden state
        hidden = self.init_hidden(features)

        # Start with <bos> token
        inputs = torch.full(
            (B, 1),
            fill_value=self.bos_idx,
            dtype=torch.long,
            device=device,
        )

        generated_tokens = []

        for _ in range(max_len):
            emb = self.embedding(inputs)           # (B, 1, embed_size)
            output, hidden = self.lstm(emb, hidden)  # output: (B, 1, hidden_size)
            logits = self.fc(output[:, -1, :])     # (B, vocab_size)
            next_token = logits.argmax(dim=-1)     # (B,)

            generated_tokens.append(next_token)

            inputs = next_token.unsqueeze(1)       # next step's input

            # If ALL sequences predicted <eos>, we can stop early
            if (next_token == self.eos_idx).all():
                break

        # Stack to (B, T)
        predicted_ids = torch.stack(generated_tokens, dim=1)
        return predicted_ids
