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
torch.manual_seed(1)
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
    return ''.join([int_to_char[i] for i in tensor])



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
    The Bigram Language Model is a simple model that uses the previous word to predict the next word.
    It is a simple model that can be used to generate text.
    It works on a statistical principle that the probability of a word depends on the previous word.
'''


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def forward(self, x, targets=None):
        token_embedding = self.token_embedding(x)
        logits = self.lm_head(token_embedding)

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
            # get predication for the next token
            '''
                We are getting the prediction for the next token.
                This is done by passing the input through the model.
                Ignoring the loss as we are not training the model.
            '''
            logit, _ = self(idx)

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


model = BigramLanguageModel().to(device)
'''
    This will update the parameters by taking the gradient decent.
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num = 1000
for i in range(num):

    if i % eval_interval == 0:
        Losses = estimate_loss(model, eval_interval)
        print(f"Train Loss: {Losses['train']} Test Loss: {Losses['test']}")

    x, y = get_batch('train')
    output, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i == 0:
        print("Initial Loss = ", loss.item())

print("Final Loss = ", loss.item())