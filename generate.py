import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Parameters and Definitions ---
# These must match the parameters used during training in training.py
batchSize = 32
blockSize = 128
maxIters = 5000
evalInterval = 300
nEmbd = 64
batchSize = 32
learningRate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters = 200
nHead = 6
nLayer = 6
dropout = 0.2

# --- Vocabulary and Tokenization ---
# We still need to know how to encode/decode text, so we rebuild the vocabulary
with open('input.txt', 'r', encoding = 'utf-8') as file:
    data = file.read()
characters = sorted(list(set(data)))
vocabSize = len(characters)
stoi = { ch:i for i,ch in enumerate(characters) }
itos = { i:ch for i,ch in enumerate(characters) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- Model Architecture ---
# The model class definitions must be present so PyTorch knows how to structure the loaded weights.
# Just copy the Head, MultiHeadAttention, FeedForward, Block, and BigramLanguageModel classes here.

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        self.blocks = nn.Sequential(*(Block(nEmbd, nHead) for _ in range(nLayer)))
        self.lnF = nn.LayerNorm(nEmbd)
        self.lmHead = nn.Linear(nEmbd, vocabSize)
    def forward(self, idx, targets = None):
        B, T = idx.shape
        tokEmb = self.tokenEmbeddingTable(idx)
        posEmb = self.positionEmbeddingTable(torch.arange(T, device = device))
        x = tokEmb + posEmb
        x = self.blocks(x)
        x = self.lnF(x)
        logits = self.lmHead(x)
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

# --- Load Model and Generate ---
# Instantiate the model with the same architecture
model = BigramLanguageModel(vocabSize)
m = model.to(device)

# Load the saved weights
m.load_state_dict(torch.load('model_weights.pth'))
m.eval() # Set the model to evaluation mode

# Interactive sentence completion loop
print("Sentence completer is ready. Type 'exit' to quit.")
while True:
    start_sentence = input("> ")
    if start_sentence.lower() == 'exit':
        break
    
    # Encode, convert to tensor, and add batch dimension
    context = torch.tensor(encode(start_sentence), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate and decode the output
    generated_tokens = m.generate(context, maxNewTokens=100)[0].tolist()
    print(decode(generated_tokens))
    print("-" * 20)