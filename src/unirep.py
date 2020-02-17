import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class UniRep(nn.Module):
    def __init__(self, num_tokens, padding_idx, embed_size, hidden_size, num_layers):
        super().__init__()

        # Define parameters
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(self.num_tokens, self.embed_size, padding_idx = padding_idx)

        self.lin = nn.Linear(self.hidden_size, self.num_tokens)

        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)

    def forward(self, xb, lengths):
        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Pack padded sequence
        packed_seq = pack_padded_sequence(xb, lengths, batch_first = True, enforce_sorted = False)
        packed_out, last_hidden = self.rnn(packed_seq)
        out = pad_packed_sequence(packed_out, batch_first = True)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        log_likelihoods = nn.functional.log_softmax(linear, dim = 2)

        # Calculate loss
        true = torch.zeros(trunc_xb.shape, dtype = torch.int64, device = device) + PADDING_VALUE
        true[:, :-1] = trunc_xb[:, 1:]

        # Flatten the sequence dimension to compare each timestep in cross entropy loss
        pred = pred.flatten(0, 1)
        true = true.flatten()
        loss = criterion(pred, true)
        return loss, {}

    def get_representations(self, xb, mask):
        with torch.no_grad():
            embedding = self.embed(xb)
            hidden = None
            if self.rnn_type == "mLSTM":
                out, _ = self.rnn(embedding, hidden, mask)
            else:
                out, _ = self.rnn(embedding, hidden)

            mask = mask.unsqueeze(-1)
            masked_out = out * mask
            representations = masked_out.sum(dim = 1) / mask.sum(dim = 1)
            return representations

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"UniRep summary:\n"
                f"  RNN type:    {type(self.rnn).__name__}\n"
                f"  Embed size:  {self.embed_size}\n"
                f"  Hidden size: {self.hidden_size}\n"
                f"  Layers:      {self.num_layers}\n"
                f"  Parameters:  {num_params:,}\n")
