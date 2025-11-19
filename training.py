import torch
import torch.nn as nn
from torch.nn import functional as F

#Parameters
batchSize = 64
blockSize = 256
maxIters = 5000
evalInterval = 500
nEmbd = 64
batchSize = 32
learningRate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters = 200
nHead = 6
nLayer = 6
dropout = 0.2

with open('input.txt', 'r', encoding = 'utf-8') as file: #open the file for reading
    data = file.read()

characters = sorted(list(set(data))) #get all unique characters sorted
vocabSize = len(characters) #the amount of unique characters we have

#tokenization(characters to integer mapping)
stoi = { ch:i for i,ch in enumerate(characters) } #string to integer mapping
itos = { i:ch for i,ch in enumerate(characters) } #integer to string
encode = lambda s: [stoi[c] for c in s] #encoder: string to integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: integers to string(reverse of encoder)

#encode all of input.txt and store in torch.Tensor
data = torch.tensor(encode(data), dtype=torch.long)

#spliting up the data into training and validation sets
mid = int(0.9*len(data)) #first 90% of data will be training
trainData = data[:mid]
valData = data[mid:]

trainData[:blockSize+1] #first 8 characters as inputs, next 8 characters as targets
x = trainData[:blockSize]
y = trainData[1:blockSize+1]


torch.manual_seed(1337) #for reproducibility

def getBatch(split):
    #This generages a small batch of data for inputs x and targets y
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - blockSize, (batchSize,)) #random batches of starting indices
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])
    x, y = x.to(device), y.to(device)
    #outputs in 4x8 tensors
    return x, y #return the input and target tensors

@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#feed inputs to transformer

class Head(nn.Module):
    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbd, headSize, bias = False)
        self.query = nn.Linear(nEmbd, headSize, bias = False)
        self.value = nn.Linear(nEmbd, headSize, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList((Head(headSize) for _ in range(numHeads)))
        self.proj = nn.Linear(headSize * numHeads, nEmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, nEmbd):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(nEmbd, 4 * nEmbd),
             nn.ReLU(),
             nn.Linear(4 * nEmbd, nEmbd),
             nn.Dropout(dropout),
         )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nEmbd, nHead):
        super().__init__()
        headSize = nEmbd // nHead
        self.sa = MultiHeadAttention(nHead, headSize)
        self.ffwd = FeedForward(nEmbd)
        self.ln1 = nn.LayerNorm(nEmbd)
        self.ln2 = nn.LayerNorm(nEmbd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocabSize):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        self.blocks = nn.Sequential(*(Block(nEmbd, nHead) for _ in range(nLayer)))
        self.lnF = nn.LayerNorm(nEmbd)
        self.lmHead = nn.Linear(nEmbd, vocabSize)
        self.apply(self._initWeights)

    def _initWeights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tokEmb = self.tokenEmbeddingTable(idx) # (Batch,Time,Channel)
        posEmb = self.positionEmbeddingTable(torch.arange(T, device = device))
        x = tokEmb + posEmb
        x = self.blocks(x)
        x = self.lnF(x)
        logits = self.lmHead(x) # (Batch,Time,VocabSize)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, maxNewTokens):
        for _ in range(maxNewTokens):
            idxCond = idx[:, -blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idxNext = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idxNext), dim = 1)
        return idx

model = BigramLanguageModel(vocabSize)
m = model.to(device)

#PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(maxIters):

    xb, yb = getBatch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    torch.save(m.state_dict(), 'model_weights.pth')

startSentence = input("Enter the start of a sentence: ")
encodedInput = encode(startSentence)
#context = torch.tensor(encodedInput, dtype=torch.long, device=device).unsqueeze(0)
#generatedTokens = m.generate(context, maxNewTokens=100)[0].tolist()

