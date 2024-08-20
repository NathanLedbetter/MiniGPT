import sentencepiece as spm
import numpy
with open('datasets/simplewiki_ascii.txt', 'r', errors='ignore') as f:
    text = f.read()
sp = spm.SentencePieceProcessor()
sp.load('10k.model')
data = sp.Encode(text)
print(len(data))
numpy.save('datasets/simplewiki_tokens_10k', data)