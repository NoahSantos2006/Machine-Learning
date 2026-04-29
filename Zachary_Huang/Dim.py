import torch

scores = torch.tensor([[10., 20., 30.], [5., 10., 15.]])

# This calculates: (10 + 20 + 30 + 5 + 10 + 15) / 6
average_score = scores.mean()

"""

dim argument lets you control *which direction to collapse*

dim=0: collapses rows. Operates "vertically"
dim=1: collapses columns. Operates "horizontally"

"""

# Get average FOR EACH ASSIGNMENT, collapse student dimension (dim=0)
avg_per_assigment = scores.mean(dim=0)

# Get average FOR EACH STUDENT, collapse assignment dimension (dim=1)
avg_per_student = scores.mean(dim=1)



