import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 1
n_layer = 1
dropout = 0.2

with open("input.txt", 'r', encoding = 'utf-8') as f:
  text = f.read()

#unique chars in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

#encode and decode
stoi = {ch:i for i, ch in enumerate(chars)}
itoi = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itoi[i] for i in l])

#train and test split
data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
  data = train_data if split =='train' else val_data
  ix = torch.randint(len(data) - block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device) , y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() 
  return out

class Head(nn.Module):
  
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) #B,T,C
    q = self.query(x) #B,T,C

    #compute attenstion scores(affinities), normalise = c**-0.5
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(B,T,C) @ (B,C,T) ----> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) #(B,T,T)

    wei = F.softmax(wei, dim = -1)
    wei = self.dropout(wei)
    #perform weighted aggregation
    v = self.value(x)
    out = wei @ v #(B,T,T) @ (B,T,C) ---> (B,T,C)

    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size*num_heads,n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out =  torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd),
      nn.Dropout(dropout),
    )
  
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  
  """Transformer block: communication followed by computation"""

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa = MultiHeadAttention(n_head, head_size) #Communication
    self.ffwd = FeedForward(n_embd) #Computation
    self.ln1 = nn.LayerNorm(n_embd)    
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

#Bigram Model
class BigramLanguageModel(nn.Module):

    def  __init__(self):
      super().__init__()
      #each token reads off the logits for the next token from a lookup table
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)
      self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
      self.ln_f = nn.LayerNorm(n_embd)
      self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
      B,T =idx.shape
      tok_emb = self.token_embedding_table(idx) #Follow (B,T,C)
      pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
      x = tok_emb + pos_emb
      x = self.blocks(x)
      x = self.ln_f(x) #(B,T,C)
      logits = self.lm_head(x)  #Follow (B,T,vocab_size)

      if targets == None:
        loss = None
      else:
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

      return logits, loss

    def generate(self, idx, max_new_tokens):
      # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
        #crop idx to the last block_size
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond) #predictions
        #focus only on the last time step
        logits = logits[:, -1, :] #becomes (B,C)
        probs = F.softmax(logits, dim=-1) #(B,C)
        idx_next = torch.multinomial(probs,num_samples =1)
        idx = torch.cat((idx, idx_next), dim=1)
      return idx

model = BigramLanguageModel()
m = model.to(device)

#no of paramaters
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
#optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
  
  #evaluate loss on train and val sets one in a while
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")

  #sample data
  xb, yb = get_batch('train')
  
  #loss
  logits, loss = m(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

#generate from model
context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
