import sentencepiece as spm


spm.SentencePieceTrainer.train('--input=datasets/simplewiki_ascii.txt --model_prefix=10k --vocab_size=10240 --max_sentence_length=5611')
sp = spm.SentencePieceProcessor()
sp.load('10k.model')

#print part of the vocab
for id in range(100):
    print(sp.id_to_piece(id*100))