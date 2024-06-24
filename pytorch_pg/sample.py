import torch
import torch.nn as nn
import torch.optim as optim

# Define the dataset (simple linear relationship: y = 2x + 3)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
y = torch.tensor([[5.0], [7.0], [9.0], [11.0]], requires_grad=True)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output size are both 1

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegressionModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing X to the model
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass: Compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predicted = model(X).detach().numpy()

print("Predicted values: ", predicted)
