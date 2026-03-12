import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) -->
    # Hidden Layer 1 (number of neurons) -->
    # H2 (n) -->
    # output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):

        super().__init__() # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)


        return x

if __name__ == "__main__":

    # Pick a manual seed for randomization
    torch.manual_seed(41)

    # Create an instance of model
    model = Model()

    url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    df = pd.read_csv(url)

    df['species'] = df['species'].map({

        'setosa': 0.0,
        'versicolor': 1.0,
        'virginica': 2.0
    
    }).astype(float)
    

    # Train Test Split. Set X, y
    X = df.drop(columns=['species'], axis=1)
    y = df['species']

    # Convert these to numpy arrays
    X = X.values
    y = y.values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    # Convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Convert y features to long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Set the criterion of model to measure the error, how far off the predictions are from the data
    criterion = nn.CrossEntropyLoss()

    # Choose Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train our model
    # Epochs? (one run thru all the training data in our notebook)
    epochs = 100
    losses = []

    for i in range(epochs):

        # Go forward and get a prediction
        y_pred = model.forward(X_train) # Get predicted results

        # Measure the loss/error, gonna be high at first
        loss = criterion(y_pred, y_train) # predicted values vs y_train

        # Keep track of our losses
        losses.append(loss.detach().numpy())

        # print every 10 epoch
        if i % 10 == 0:

            print(f"Epoch: {i} and loss: {loss}")

        # Do some back propagation: take the error rate of forward propagation and feed it back
        # through the network to find tune the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Graph it out
    # plt.plot(range(epochs), losses)
    # plt.ylabel("loss/error")
    # plt.xlabel("Epoch")

    # plt.show()

    # Evaluate Model on Test Data Set (validate model on test set)
    with torch.no_grad(): # Turn off back propogation

        y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
        loss = criterion(y_eval, y_test) # Find the loss/error 

    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model.forward(data)

            if y_test[i] == 0:
                x = "Setosa"
            elif y_test[i] == 1:
                x="Versicolor"
            else:
                x = 'Virginica'

            # Will tell us what type of flower class out network thinks it is (highest number in tensor array)
            print(f"{i+1}.) {str(y_val)} \t {x} \t {y_val.argmax().item()}")

            # Correct or not
            if y_val.argmax().item() == y_test[i]:

                correct += 1
    
    print(f"We got {correct} correct!")

    # Using model on new data

    new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

    with torch.no_grad():
        print(model(new_iris))
    
    newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

    with torch.no_grad():
        print(model(newer_iris))

    # Save our NN model
    torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')

    # Load the Saved Model
    new_model = Model()
    new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))

    new_model.eval()