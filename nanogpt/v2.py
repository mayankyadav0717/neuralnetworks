import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else cpu
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#-------------------------------------------------------------------------------------

torch.manual_seed(1337)
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
# create a vocabulary of all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]     #encoder: encodes a string, output is a list of integers
decode = lambda l: ''.join(itos[i] for i in l)  #decoder: takes a list of integer, output is string

#Train and validation split
data = torch.tensor(encode(text),dtype=torch.long,device=device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in {'train','val'}:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''one head of self-attention'''

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores("affinities")
        wei = q @ k.transpose(-2,-1) * C**0.5 #(B,T,C)@(B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))    #(B,T,T)
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        #perform the weighted aggregation of the values
        v = self.value(x)   #B,T,C
        out = wei @ v # B,T,T @ B,T,C -> B,T,C
        return out
    
class MultiHeadAttention(nn.Module):
    '''multiple heads of sel-attention in parallel'''

    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)     #concatenation over channel dimension
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    '''a simple linear layer followed by non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd), #multiply by 4 according to the paper
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    '''Transformer block: communication followed by computation'''

    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
       
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#simple bigram model
class BigramLangaugeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)    #Creates a lookup table of the 65 vocab telling the probability of the next character
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)   #B,T,C This C is different from the one below
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))  # T,C 
        x = tok_emb+pos_emb
        #x = self.sa_heads(x)     #apply one head of self-attention (B,T,C)
        #x = self.ffwd(x)    #B,T,C
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,C) Batch , Time, Channel batch is the batch_size =4, time is block_size = 8, and channel is the vocab_size
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)              #Stretches the 4,8,65 to 32,65 
            targets = targets.view(B*T)              #stretches 4,8 to 32
            loss = F.cross_entropy(logits,targets)   #cross_entropy wants B,C as input 
        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        #idx is [B,T] array of indices in current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            #get the predictions
            logits,loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #becoms B,C
            #apply softmax to get probabilites
            probs = F.softmax(logits,dim=-1)
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1)
            #append the sampled index to running sentence
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = BigramLangaugeModel()
m = model.to(device)

#create optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for iter in range(max_iters):

    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb,yb = get_batch('train')

    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from model
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))