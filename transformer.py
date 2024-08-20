import torch
import torch.nn as nn

#PARAMETERS
BLOCK_SIZE = 64 #context length
EMBED_DIM = 512
NUM_HEADS = 8
dropout_rate = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, batch_first=True)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if BLOCK_SIZE >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:BLOCK_SIZE, :BLOCK_SIZE]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask, need_weights=True)

        output = self.proj(attn_output)

        return output


class FeedFoward(nn.Module):
    """ Feed-Forward Network according to the specs of 'Attention Is All You Need """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            #Scale up the dimensionality of the inner layer by 4x
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            #Activation function
            nn.LeakyReLU(),
            #Scale the output back down to EMBED_DIM
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Layer(nn.Module):
    """ Transformer decoder layer designed according to 'Attention Is All You Need' """

    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiHead()
        self.feed_forward = FeedFoward()
        self.normalization = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        #Masked Multi-Head Attention, Addition and Layer Normalization
        x = x + self.multihead_attention(x)
        x = self.normalization(x)
        #Feed Forward, Addition, and Layer Normalization
        x = x + self.feed_forward(x)
        x = self.normalization(x)
        return x


class DecoderTransformer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(Layer(), Layer(), Layer(), Layer(), nn.LayerNorm(EMBED_DIM))
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            #Compare predictions to the targets to get the loss
            #View flattens the batch and block dimensions of the logits and targets tensors to fit Pytorch cross entropy's parameters
            x, y, z = logits.shape
            loss = nn.functional.cross_entropy(logits.view(x*y, z), targets.view(x*y))
        return logits, loss
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx