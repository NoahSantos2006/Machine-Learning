import torch

# The input has 1 feature, the output has 1 value
D_in = 1
D_out = 1

# Create the Linear layer
linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

# You can look inside and see the parameters it created for you
print(f"Layer's Weight (W): {linear_layer.weight.item()}\n")
print(f"Layer's Bias (b): {linear_layer.bias.item()}\n")

X = torch.randn(10, 1, requires_grad=True)

# You use it just like a function. This is the forward pass.
# (Assume X is a tensor of shape [10, 1] from previous chapters
y_hat_nn = linear_layer(X)

print(f"Output of nn.Linear (first 3 rows):\n {y_hat_nn[:3]}\n")

"""

What is a parameter

    - A special tensor that:

        - requires_grad=True by default
        - Auto-registers with the model
        - handles all bookkeeping

        
Activation Function

To learn complex, messy patterns of the real world, you need to introduce
'kinks' or non-linearities between your linear layers

"""

# 'NN.RELU' (Rectified Linear Unit)
# ReLU(x) = max(0, x)

relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = relu(sample_data)

print(f"Original Data:     {sample_data}")
print(f"Data after ReLU:     {activated_data}\n")

# 'NN.GELU' (Gaussian Error Linear Unit)
# Modern STandard for Transformers (GPT, Llama). A smoother, gently curving version of ReLU

gelu = torch.nn.GELU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = gelu(sample_data)

print(f"Original Data:     {sample_data}")
print(f"Data after GELU:     {activated_data}\n")

# 'NN.softmax'
# Used on final output layer for classification
# Converts logits --> probability distribution

softmax = torch.nn.Softmax(dim=-1)
# Raw model scores for 2 items, across 4 possible classes
logits = torch.tensor([[1.0, 3.0, 0.5, 1.5], [-1.0, 2.0, 1.0, 0.0]])
probabilities = softmax(logits)

print(f"Output Probabilities:\n {probabilities}")
print(f"Sum of probabilities for item 1:\n {probabilities[0].sum()}\n")

"""

NN.EMBEDDING

    - Words --> Numbers
    - Learnable lookup table
    - Each word gets a unique vector

"""

vocab_size = 10     # Our language has 10 unique words
embedding_dim = 3   # We'll represent each word with a 3D vector

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

# Input: A sentence where each word is an ID. e.g., 1, 5, 0, 8
input_ids = torch.tensor([[1, 5, 0, 8]])
word_vectors = embedding_layer(input_ids)

"""

NN.LAYERNORM

    - Prevents values from exploding/vanishing
    - Rescales to stable range
    - Essential for deep networks

"""

norm_layer = torch.nn.LayerNorm(normalized_shape=3)
input_features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
normalized_features = norm_layer(input_features)

print(f"Mean (should be ~0): {normalized_features.mean(dim=-1)}")
print(f"Std Dev (should be ~1): {normalized_features.std(dim=-1)}\n")


"""

NN.DROPOUT

    - Prevents overfitting
    - Randomly zeros neurons during training
    - Forces network robustness

"""

dropout_layer = torch.nn.Dropout(p=0.5)
input_tensor = torch.ones(1, 10)

# ACtiavte dropout for training
dropout_layer.train()
output_during_train = dropout_layer(input_tensor)

# Deactivate dropout for evaluation/predictino
dropout_layer.eval()
output_during_eval = dropout_layer(input_tensor)

print(f"Output during training (randomly zeroed and scaled):\n {output_during_train}")
print(f"Output during evaluation (identity function):\n {output_during_eval}")