import os
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #how many independent sequences will we process in parallel?
block_size = 256 #what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2



torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, 'input.txt'), 'r', encoding='utf-8') as f:
  text=f.read()


chars = sorted(list(set(text)))
vocab_size=len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s] #Taking a string and output list of integers
decode = lambda l : ''.join([itos[i] for i in l]) #Taking a list of integers output a string

#Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading

def get_batch(split):
  #Generating a small batch data of inputs x and targets y
  data=train_data if split=='train' else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size]for i in ix])
  y=torch.stack([data[i+1:i+block_size+1]for i in ix])
  x,y=x.to(device),y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
  """One head of Self Attention"""
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias=False)
    self.query = nn.Linear(n_embd,head_size,bias=False)
    self.value = nn.Linear(n_embd,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    self.dropout=nn.Dropout(dropout)
  
  def forward(self,x):
    B,T,C = x.shape
    k= self.key(x) # B,T,16
    q= self.query(x) # B,T,16

    wei= q @ k.transpose(-2,-1) * C**-0.5 # (B,T,16) @ (B,16,T) ---> (B,T,T)
    wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei=F.softmax(wei,dim=-1)
    wei=self.dropout(wei)
    v=self.value(x)
    out =wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  """Multiple heads of Self Attention in parallel"""

  def __init__(self,n_head,head_size):
    super().__init__()
    self.heads= nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj=nn.Linear(n_embd,n_embd)
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    out= torch.cat([h(x) for h in self.heads],dim=-1)
    out=self.proj(out)
    return out

class FeedForward(nn.Module):
  """A simple linear layer followed by a non-linearity"""

  def __init__(self,n_embd):
      super().__init__()
      self.net=nn.Sequential(
         nn.Linear(n_embd,4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd,n_embd),
         nn.Dropout(dropout)
      )

  def forward(self,x):
      return self.net(x)
  


class LayerNorm: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


class Block(nn.Module):
  """Transformer block: communication followed by a feed forward"""

  def __init__(self,n_embd,n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa=MultiHeadAttention(n_head,head_size)
    self.ffwd=FeedForward(n_embd)
    self.ln1=nn.LayerNorm(n_embd)
    self.ln2=nn.LayerNorm(n_embd)

  def forward(self,x):
    x= x + self.sa(self.ln1(x))
    x= x + self.ffwd(self.ln2(x))
    return x



#Super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #Each token directly reads the logits of the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size,n_embd)
    self.blocks=nn.Sequential(
    *[Block(n_embd,n_head=n_head) for _ in range(n_layer)]
    )
    self.ln_f=nn.LayerNorm(n_embd) #Final layer norm
    self.lm_head=nn.Linear(n_embd,vocab_size)
    

  def forward(self,idx,targets=None):
    B,T =idx.shape
    #idx and targets are bith (B,T) tensor of integers
    token_embd=self.token_embedding_table(idx) # (B,T,C) Batch Time Channel
    pos_embd=self.position_embedding_table(torch.arange(T,device=device))
    x=token_embd + pos_embd
    x=self.blocks(x)
    logits=self.lm_head(x)
    #Reshaping the logits beacause Cross Entropy needs B C T not B T C

    if targets is None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C) #Creating the array 2  Dimensional
      targets=targets.view(B*T)

      loss = F.cross_entropy(logits,targets)

    return logits , loss

  def generate(self,idx,max_new_tokens):
    #idx is (B,T) arrays of indices in current context
    for _ in range(max_new_tokens):
      #croping idx to last block_size tokens
      idx_cond=idx[:,-block_size:]
      #get the prediction
      logits,loss=self(idx_cond)
      #focusing only on the last time step
      logits=logits[:,-1,:] #become (B,C)
      #Applying softmax to get probabilities
      probs=F.softmax(logits,dim=1) # (B,C)
      #sample from the distribution
      idx_next= torch.multinomial(probs,num_samples=1) #(B,1)
      #append sampled index to the running sequence
      idx=torch.cat((idx,idx_next),dim=1) # (B,T+1)
    return idx
  

      
model=BigramLanguageModel()
m=model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6,'M parameters')

#create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

for iter in range(max_iters):
   
    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb,yb=get_batch('train')
    #evaluate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))