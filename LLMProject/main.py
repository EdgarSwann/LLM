from sympy.printing.pytorch import torch
import torch
import torch.nn as nn
from torch.nn import functional as F


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Load the text file
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

# Testing if the file loaded correctly
# print("There are a total of {} characters in the dataset".format(len(text)))
# print(text[:1000])

# Create the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("There are a total of {} unique characters in the dataset".format(vocab_size))
# print("".join(chars))

# Create mappings from characters to integers and vice versa
# Creates dictonary from string to index and vice versa
string_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_symbol = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_index[character] for character in s] # map the string to integers
decode = lambda l: "".join(index_to_symbol[index] for index in l) # map the integer to string
#
# print(encode("Buenos Dias"))
# print(decode(encode("Buenos Dias")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.type)
print(data[:1000])

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8 # context length
batch_size = 4 # how many independent sequences will we process in parallel

def get_batch(split):
    # generate a random batch of data, but only from complete words
    data = train_data if split == 'train' else val_data

    # generate a random number, that is the starting index of the batch and if it doesn't correspont to a space character, increase it by 1 until it does
    while True:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        if all(data[i+block_size-1] != string_to_index[' '] for i in ix):
            break

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

'''
inputs:
torch.Size([4, 8])
tensor([[ 0, 15, 24, 13, 30, 17, 26, 15],
        [52,  1, 46, 39, 58, 46,  1, 40],
        [52, 47, 43, 56,  8,  1, 19, 53],
        [43, 43, 42, 47, 52, 45,  1, 57]])
        
        
targets
torch.Size([4, 8])
tensor([[15, 24, 13, 30, 17, 26, 15, 17],
        [ 1, 46, 39, 58, 46,  1, 40, 43],
        [47, 43, 56,  8,  1, 19, 53,  1],
        [43, 42, 47, 52, 45,  1, 57, 50]])
'''


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C) where C is the vocab size

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens): # idx is the current context, size (B, T)
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # pluck out the last time step, becoming (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

xb, yb = get_batch('train')

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape) # (B, T, C)
print(loss)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# create a 1x1 tensor of zeros (new line character) and generate 100 tokens from it and decode it to string
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

torch.manual_seed(1337)
B,T,C = 4, 8, 2 # batch size, time steps, channels
x = torch.randn(B,T,C)
print(x.shape)

# weights = torch.tril(torch.ones(T,T))
# weights = weights / weights.sum(1, keepdim=True)
# print(weights)
# xbow = weights @ x # (B, T, T) @ (B, T, C) --> (B, T, C)

# Version 3: using softmax
tril = torch.tril(torch.ones(T,T))
mask = tril.masked_fill(tril == 0, float('-inf')).masked_fill(tril == 1, float(0.0))
weights = F.softmax(mask, dim=-1)
xbow = weights @ x

# Version 4: self-attention
B,T,C = 4,8,32 # batch size, time steps, channels
x = torch.randn(B,T,C)

# let's see a single head attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, head_size)
q = query(x) # (B, T, head_size)

wei = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
wei = wei / head_size**0.5 # scale the weights by the square root of the head size

tril = torch.tril(torch.ones(T,T))
# deleting this line causes the model to look into the future ( all the nodes will talk to each other)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # (B, T, T)

v = value(x) # (B, T, head_size)
out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

print(out.shape)
