# Language Model with Transformer Architecture

## Overview
This project implements a language model using a Transformer architecture. The model is capable of processing text, predicting the next word, and generating new text based on learned patterns.

## Features
- Implements a Transformer model with self-attention mechanism
- Supports both simple and multi-headed attention
- Includes a feed-forward neural network layer
- Trains on text data to predict and generate text

## Requirements
- Python 3.10
- PyTorch
- CUDA-capable GPU (optional, for faster training)

## Installation
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`

## Usage
1. Prepare your text data in the `Data/forms` directory aquired from [Kaggle](https://www.kaggle.com/datasets/michaelarman/poemsdataset).
2. Run the script: `python Script.py`

## Model Architecture
The model consists of several key components:
- Token and Position Embeddings
- Self-Attention Heads
- Multi-Level Attention
- Feed-Forward Layers
- Layer Normalization

## Training
The model is trained using:
- AdamW optimizer
- Cross-entropy loss function
- Batch processing of text data

## Results
The script outputs:
- Training and testing loss at regular intervals
- Generated text sample after training

## Performance Comparison
The code compares three different approaches:
1. Simple Attention
2. Multi-Headed Attention
3. Feed Forward Layer

Each approach's initial and final loss are reported for comparison.

## Customization
You can adjust various parameters in the script:
- `block_size`: Context length for predictions
- `batch_size`: Number of samples per training batch
- `max_iters`: Maximum training iterations
- `n_embd`: Embedding dimension
- `n_head`: Number of attention heads
- `n_layer`: Number of transformer layers

## Note
This implementation was for my learning purposes and will need future optimization.