import torch

x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Get the third column (at index 2)
# tensor([2, 6, 10])
col_2 = x[:, 2]


"""

Dynamic Selection: 'ARGMAX'

    Find the index of the highest value. Thisis how you find the model's final prediction

"""

scores = torch.tensor([
    # Best score is at index 3
    [10, 0, 5, 20, 1],
    # Best score is at index 1
    [1, 30, 2, 5, 0]
])

# Find the index of the best score for each
best_indices = torch.argmax(scores, dim=1) # --> tensor([3, 1])


"""

Standard Indexing: "Column 2 for ALL Rows"

"""

data = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]
])

# Our "shopping list" of which column to get from each row
indices_to_select = torch.tensor([[2], [0], [3]])

# Gather from data along dim=1 (the columns)
selected_values = torch.gather(data, dim=1, index = indices_to_select)
# tensor([[12],     # From row 0, it got index 2
#         [20],     # From row 1, it got index 0
#         [33]])    # From row 2, it got index 3

print(selected_values)