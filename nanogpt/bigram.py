import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else cpu
eval_iters = 200
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

#simple bigram model
class BigramLangaugeModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)    #Creates a lookup table of the 65 vocab telling the probability of the next character

    def forward(self,idx,targets=None):
        logits = self.token_embedding_table(idx) #(B,T,C) Batch , Time, Channel batch is the batch_size =4, time is block_size = 8, and channel is the vocab_size
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
            #get the predictions
            logits,loss = self(idx)
            #focus only on the last time step
            logits = logits[:,-1,:] #becoms B,C
            #apply softmax to get probabilites
            probs = F.softmax(logits,dim=1)
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1)
            #append the sampled index to running sentence
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = BigramLangaugeModel(vocab_size)
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