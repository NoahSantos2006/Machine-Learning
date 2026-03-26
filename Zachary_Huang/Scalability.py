import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, embedding_dim, ffn_dim):

        super().__init__()
        # In an LLM, embedding_dim might be 4096, ffn_dim might be 14336

        # We use the LEGO bricks we already know:
        self.layer1 = nn.Linear(embedding_dim, ffn_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(ffn_dim, embedding_dim)
    
    def foward(self, x):
        # The data flow is exactly what you'd expect:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

"""

Parameter                   Our Toy Model           A Typical LLM (e.g., Llama 3 8B)
Model                       LinearRegressionModel   Transformer
Layer                       nn.Linear               nn.Linear(inside an FFN)
Weight Matrix 'W' Shape     (1, 1)                  (4096, 14336)
Matrix Multiplication       X @ W                   X @ W
Total Parameters            2                       ~8,000,000,000


The Five-Step Logic is Universal

1. y_hat = model(X)

    - Us: Single linear layer
    - LLM: Dozens of Transformers

2. loss = loss_fn(...)

    - Us: MSE
    - LLM: Cross-Entropy Loss

3. optimizer.zero_grad()

    - Identical

4. loss.backward()

    - Identical

5. optimizer.step()

    - Identical

    

We know:

    - A model is just an nn.Module containing layers
    - A layer is just a container for parameters that performs a mathematical operation
    - Learning is just updating parameters with zero_grad, backward, step


"""



