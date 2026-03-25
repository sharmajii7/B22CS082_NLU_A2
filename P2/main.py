import torch
import torch.nn as nn
import torch.optim as optim
import random

# load names and strip whitespace
with open('TrainingNames.txt', 'r', encoding='utf-8') as tn:
    train = [l.strip().lower() for l in tn if l.strip()]

# build vocab including our special sequence tokens
chars = sorted(list(set(''.join(train)) | {'<SOS>', '<EOS>', '<PAD>'}))
charToIdx = {c: i for i, c in enumerate(chars)}
idxToChar = {i: c for i, c in enumerate(chars)}
vocabSz = len(chars)

# standard hyperparameters
embedSz = 64
hiddenSz = 128
epochs = 10 
lr = 0.005
lossFn = nn.CrossEntropyLoss()

# 1. VANILLA RNN
print("Vanilla RNN ->")
vEmb = nn.Embedding(vocabSz, embedSz)
vRnn = nn.RNN(embedSz, hiddenSz, batch_first=True)
vFc = nn.Linear(hiddenSz, vocabSz) # maps hidden state back to vocab size

vOpt = optim.Adam(list(vEmb.parameters()) + list(vRnn.parameters()) + list(vFc.parameters()), lr=lr)

vParams = list(vEmb.parameters()) + list(vRnn.parameters()) + list(vFc.parameters())
totParams = sum(p.numel() for p in vParams)
modelSzMb = (totParams * 4) / (1024 * 1024)
print(f"Vanilla RNN Parameters: {totParams:,}")
print(f"Vanilla RNN Size: {modelSzMb:.4f} MB")

for ep in range(epochs):
    random.shuffle(train)
    lossSum = 0
    for name in train:
        vOpt.zero_grad()
        
        # wrap name in SOS and EOS tokens
        seq = [charToIdx['<SOS>']] + [charToIdx[c] for c in name] + [charToIdx['<EOS>']]
        
        # target is just the input shifted by 1
        x = torch.tensor(seq[:-1]).unsqueeze(0) 
        y = torch.tensor(seq[1:])
        
        # forward pass (pytorch handles the zero-init hidden state for us)
        out, i = vRnn(vEmb(x))
        loss = lossFn(vFc(out).squeeze(0), y)
        
        loss.backward()
        vOpt.step()
        lossSum += loss.item()
        
    print(f"Epoch {ep+1} | Loss: {lossSum/len(train):.4f}")

# generation
genVanilla = []
with torch.no_grad():
    for i in range(100):
        word = []
        x = torch.tensor([[charToIdx['<SOS>']]])
        h = None # starts empty, updates in the loop
        
        for j in range(15): # max length of 15 chars
            out, h = vRnn(vEmb(x), h)
            logits = vFc(out).view(-1)
            
            # sample from the probability distribution instead of just taking the argmax
            idx = torch.multinomial(torch.softmax(logits, dim=0), 1).item()
            
            if idx == charToIdx['<EOS>']: break # stop if it guesses EOS
            word.append(idxToChar[idx])
            x = torch.tensor([[idx]]) # feed the guess into the next step
            
        genVanilla.append(''.join(word))

nov = len([n for n in genVanilla if n not in train])
div = len(set(genVanilla))
print(f"Vanilla Results -> Novelty: {nov}% | Diversity: {div}% | Samples: {genVanilla[:5]}")

# 2. BLSTM SEQ2SEQ
print("\nBLSTM Seq2Seq ->")
bEmb = nn.Embedding(vocabSz, embedSz)
bEnc = nn.LSTM(embedSz, hiddenSz, bidirectional=True, batch_first=True)
bDec = nn.LSTM(embedSz, hiddenSz * 2, batch_first=True) # hidden is 2x because of bidirectional concat
bFc = nn.Linear(hiddenSz * 2, vocabSz)

bParams = list(bEmb.parameters()) + list(bEnc.parameters()) + list(bDec.parameters()) + list(bFc.parameters())
bOpt = optim.Adam(bParams, lr=lr)

for ep in range(epochs):
    random.shuffle(train)
    lossSum = 0
    for name in train:
        seq = [charToIdx['<SOS>']] + [charToIdx[c] for c in name] + [charToIdx['<EOS>']]
        if len(seq) < 3: continue
        
        # split into source, target input, and target output
        src = torch.tensor(seq[1:-1]).unsqueeze(0)
        trgIn = torch.tensor(seq[:-1]).unsqueeze(0)
        trgOut = torch.tensor(seq[1:])
        
        bOpt.zero_grad()
        
        # encode the sequence
        i, (h, c) = bEnc(bEmb(src))
        
        # smash the forward and backward states together for the decoder
        h = torch.cat((h[0:1], h[1:2]), dim=2)
        c = torch.cat((c[0:1], c[1:2]), dim=2)
        
        # decode and calculate loss
        out, j = bDec(bEmb(trgIn), (h, c))
        loss = lossFn(bFc(out).squeeze(0), trgOut)
        
        loss.backward()
        bOpt.step()
        lossSum += loss.item()
        
    print(f"Epoch {ep+1} | Loss: {lossSum/len(train):.4f}")

# generation
genBlstm = []
with torch.no_grad():
    for i in range(100):
        word = []
        # give the encoder some random noise as a seed to generate new names
        src = torch.randint(0, vocabSz, (1, 5)) 
        
        j, (h, c) = bEnc(bEmb(src))
        h = torch.cat((h[0:1], h[1:2]), dim=2)
        c = torch.cat((c[0:1], c[1:2]), dim=2)
        
        trgIn = torch.tensor([[charToIdx['<SOS>']]])
        for k in range(15):
            out, (h, c) = bDec(bEmb(trgIn), (h, c))
            logits = bFc(out).view(-1)
            idx = torch.multinomial(torch.softmax(logits, dim=0), 1).item()
            
            if idx == charToIdx['<EOS>']: break
            word.append(idxToChar[idx])
            trgIn = torch.tensor([[idx]])
            
        genBlstm.append(''.join(word))

nov = len([n for n in genBlstm if n not in train])
div = len(set(genBlstm))
print(f"BLSTM Results -> Novelty: {nov}% | Diversity: {div}% | Samples: {genBlstm[:5]}")

# 3. ATTENTION RNN
print("\nAttention RNN ->")
aEmb = nn.Embedding(vocabSz, embedSz)
aEnc = nn.RNN(embedSz, hiddenSz, batch_first=True)
aDec = nn.RNN(embedSz + hiddenSz, hiddenSz, batch_first=True) # takes char + context vector
aAttnW = nn.Linear(hiddenSz * 2, hiddenSz)
aAttnV = nn.Linear(hiddenSz, 1, bias=False)
aFc = nn.Linear(hiddenSz, vocabSz)

aParams = list(aEmb.parameters()) + list(aEnc.parameters()) + list(aDec.parameters()) + list(aAttnW.parameters()) + list(aAttnV.parameters()) + list(aFc.parameters())
aOpt = optim.Adam(aParams, lr=lr)

for ep in range(epochs):
    random.shuffle(train)
    lossSum = 0
    for name in train:
        seq = [charToIdx['<SOS>']] + [charToIdx[c] for c in name] + [charToIdx['<EOS>']]
        if len(seq) < 3: continue
        
        src = torch.tensor(seq[1:-1]).unsqueeze(0)
        trgIn = torch.tensor(seq[:-1]).unsqueeze(0)
        trgOut = torch.tensor(seq[1:])
        
        aOpt.zero_grad()
        encOut, h = aEnc(aEmb(src))
        
        logitsList = []
        # step-by-step decoding to apply attention at each time step
        for t in range(trgIn.shape[1]):
            charIn = aEmb(trgIn[:, t:t+1])
            
            # Bahdanau attention scoring: figure out which encoder states to focus on
            hExp = h.permute(1, 0, 2).expand(-1, encOut.size(1), -1)
            energy = torch.tanh(aAttnW(torch.cat((hExp, encOut), dim=2)))
            alpha = torch.softmax(aAttnV(energy).squeeze(2), dim=1).unsqueeze(1)
            
            # create the weighted context vector
            ctx = torch.bmm(alpha, encOut)
            
            # feed char and context into decoder
            out, h = aDec(torch.cat((charIn, ctx), dim=2), h)
            logitsList.append(aFc(out))
            
        loss = lossFn(torch.cat(logitsList, dim=1).squeeze(0), trgOut)
        loss.backward()
        aOpt.step()
        lossSum += loss.item()
        
    print(f"Epoch {ep+1} | Loss: {lossSum/len(train):.4f}")

# generation
genAttn = []
with torch.no_grad():
    for i in range(100):
        word = []
        src = torch.randint(0, vocabSz, (1, 5))
        encOut, h = aEnc(aEmb(src))
        
        trgIn = torch.tensor([[charToIdx['<SOS>']]])
        for j in range(15):
            charIn = aEmb(trgIn)
            
            # calculate attention for the current generation step
            hExp = h.permute(1, 0, 2).expand(-1, encOut.size(1), -1)
            energy = torch.tanh(aAttnW(torch.cat((hExp, encOut), dim=2)))
            alpha = torch.softmax(aAttnV(energy).squeeze(2), dim=1).unsqueeze(1)
            
            ctx = torch.bmm(alpha, encOut)
            out, h = aDec(torch.cat((charIn, ctx), dim=2), h)
            logits = aFc(out).view(-1)
            
            idx = torch.multinomial(torch.softmax(logits, dim=0), 1).item()
            if idx == charToIdx['<EOS>']: break
            word.append(idxToChar[idx])
            trgIn = torch.tensor([[idx]])
            
        genAttn.append(''.join(word))

nov = len([n for n in genAttn if n not in train])
div = len(set(genAttn))
print(f"Attention Results -> Novelty: {nov}% | Diversity: {div}% | Samples: {genAttn[:5]}")