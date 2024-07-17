# Libraries
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Attributes to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 4
batch_size = 8
torch.manual_seed(1337)
vocabulary_size = 0
max_iters = 5000
eval_interval = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0



# Reading the data from file
path = 'Data/forms'
# opening every folder within the directory and reading each of the text files within the folder and saving it in a list
data = []
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        with open(os.path.join(path, folder, file), 'r', encoding='utf-8') as f:
            data.append(f.read())



# Getting all unique words from the data
words = []
for i in data:
    words.extend(i.split())
words = set(words)

vocabulary_size = len(set(words))

# Getting all the unique characters
char = sorted(list(set(''.join(data))))

# Encoder & Decoder for converting characters to integers and vice versa
char_to_int = {char: i for i, char in enumerate(char)}
int_to_char = {i: char for i, char in enumerate(char)}


# Encoder and Decoder Functions
def text_to_tensor(text):
    tensor = torch.tensor([char_to_int[char] for char in text], dtype=torch.long).to(device)
    return tensor.tolist()
def tensor_to_text(tensor):
    return ''.join([int_to_char.get(i, '<UNK>') for i in tensor])


# Converting the data to tensors
Data = [torch.tensor(text_to_tensor(text), dtype=torch.long).to(device) for text in data]


# Splitting the data into training and testing
train_data = Data[:int(0.8*len(Data))]
test_data = Data[int(0.8*len(Data)):]

# Function to get a batch of data
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = pad_sequence([data[i][:block_size] for i in ix], batch_first=True, padding_value=0)
    y = pad_sequence([data[i][1:block_size+1] for i in ix], batch_first=True, padding_value=0)
    return x, y


@torch.no_grad()
def estimate_loss(model,eval_interval):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

'''
    The Transformer model is a model that uses self-attention to process the input. We do not allow the model to view 
    the future tokens as they have not been generated and we can accomplish this by making the upper triangle matrix to 
    be -inf.
'''

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

'''
    The MultiLevelAttention model is a model that uses multiple heads of self-attention to process the input.
    This is just more than one head of self-attention.
'''
class MultiLevelAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super(MultiLevelAttention, self).__init__()
        self.layers = nn.ModuleList([Head(head_size) for _ in range(n_head)])

    def forward(self, x):
        return torch.cat([l(x) for l in self.layers], dim=-1)

'''
    The FeedFoward model is a simple model that uses a feedforward neural network to process the input.
    In this case feedforward means that the input is passed through a neural network and the output is generated.
'''
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiLevelAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


'''
    The Bigram Language Model is a simple model that uses the previous word to predict the next word.
    It is a simple model that can be used to generate text.
    It works on a statistical principle that the probability of a word depends on the previous word.
'''
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, n_embd)
        self.Position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def forward(self, x, targets=None):

        B, T = x.shape

        token_embedding = self.token_embedding(x)
        positional_embedding = self.Position_embedding(torch.arange(T).to(x.device))

        idx = token_embedding + positional_embedding

        idx = self.blocks(idx)  # (B,T,C)
        idx = self.ln_f(idx)  # (B,T,C)
        logits = self.lm_head(idx)

        if targets is None:
            loss_Value = None
        else:
            B, T, C = logits.shape

            Product = B * T

            logits = logits.view(Product, C)

            targets = targets.view(-1)

            # print("Logits shape: ", logits.shape)
            # print("Targets shape: ", targets.shape)

            loss_Value = F.cross_entropy(logits, targets)

        return logits, loss_Value

    def generate(self, idx, n):
        # Generate n tokens
        for _ in range(n):

            idx_cond = idx[:, -block_size:]

            # get predication for the next token
            '''
                We are getting the prediction for the next token.
                This is done by passing the input through the model.
                Ignoring the loss as we are not training the model.
            '''
            logit, _ = self(idx_cond)

            # focusing on the last token
            '''
                We are focusing on the last token as it tells us the probability of the next token to be generated.
            '''
            Temp = logit[:, -1, :]

            # getting the token with the probability using softmax
            '''
                There are multiple possible tokens that can be generated. So we generative them into a probability distribution.
                This is done using the softmax function. The softmax function converts the logits into a probability distribution.
                This probability distribution gives each token a probability of being generated out of 100%.
            '''
            probability = F.softmax(Temp, dim=-1)

            # sample from the probability distribution
            '''
                We are sampling from the probability distribution to get the next token.
                This is done using the multinomial function.
                The multinomial function generates a random sample from the probability distribution.
                By passing '1' as the second argument, we are generating a single token.
            '''
            next_token = torch.multinomial(probability, 1)

            # add the token to the input
            '''
                We are adding the token to the input.
                This is done by concatenating the token to the input.
                This is done so that we can predict the next token using the previously added one as well.
            '''
            idx = torch.cat([idx, next_token], dim=-1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m, eval_interval)
        print(f'Iter {iter} Train loss: {losses["train"]:.3f} Test loss: {losses["test"]:.3f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print (tensor_to_text(model.generate(context, 100)[0].tolist()))
