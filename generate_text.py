import torch
import sentencepiece as spm

#Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Load the model and its associated sentencepiece vocabulary
model = torch.load("MiniGPT_10k.pth")
model.to(device)
sp = spm.SentencePieceProcessor()
sp.load('10k.model')

#Generate from the model starting from a random vocabulary token
print(sp.Decode(model.generate(torch.randint(10240, (1, 1), dtype=torch.long, device=device), max_new_tokens=200)[0].tolist()))