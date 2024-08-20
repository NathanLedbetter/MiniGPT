import torch
import time
import numpy
import torch.utils.data
import sentencepiece as spm
from transformer import DecoderTransformer

### MODEL PARAMETERS ###
BATCH_SIZE = 32
BLOCK_SIZE = 64 #context length
steps = 70000
eval_interval = 100
learning_rate = 3e-4
model_path = "MiniGPT_10k.pth"
vocab = '10k.model'
data_file = 'datasets/simplewiki_tokens_10k.npy'

timer = time.time()
start = timer

### INITIALIZE THE MODEL ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load(vocab)
vocab_size = sp.vocab_size()    
print("Vocab Size:", vocab_size)

model = DecoderTransformer(vocab_size)
model = torch.load(model_path)
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'million parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

### DATA LOADING ###
tokens = torch.tensor(numpy.load(data_file), dtype=torch.long)
print(len(tokens)/1e6, 'million tokens')
print(len(tokens)//(BATCH_SIZE* BLOCK_SIZE), 'batches per epoch')
# Train and test splits
n = int(0.9*len(tokens)) # first 90% will be train, rest val
train_data = tokens[:n]
val_data = tokens[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


#The loss can vary wildly depending on the batch, so we find the average of multiple batches to give a better indicator of the model's performance
@torch.inference_mode()
def get_loss(split):
    model.eval() #switch to eval mode
    losses = torch.zeros(10)
    for i in range(10):
            inputs,targets = get_batch(split)
            logits, loss = model.forward(inputs, targets)
            losses[i] = loss.item()
    avg_loss = losses.mean()
    model.train() #switch back to train mode
    return avg_loss.item()


### TRAINING ###
for iter in range(steps):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == steps - 1:
        step_time = time.time() - timer
        timer = time.time()
        print(f"step: {iter}    train loss:{get_loss('train'):.3f}    val loss:{get_loss('val'):.3f}    step time:{step_time:.2f}s")

    # sample a batch of data
    inputs, targets = get_batch('train')
    # evaluate the loss
    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Generate from the model starting from a random vocabulary token
context = torch.randint(vocab_size, (1, 1), dtype=torch.long, device=device)
print(sp.Decode(model.generate(context, max_new_tokens=100)[0].tolist()))

print(f"Time elapsed: {(time.time() - start):.2f}")

torch.save(model, model_path)
