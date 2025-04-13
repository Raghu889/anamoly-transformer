import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
from anomaly_model import AnomalyAttention, AnomalyTransformer  # Import from the model definition file

# Data loading and preparation
df = pd.read_csv('train.csv.txt')
# df.drop('Open time', axis=1, inplace=True)
# df.drop('Close time', axis=1, inplace=True)
# df.drop('Is Anomaly', axis=1, inplace=True)
X = df.values
X_padded = np.pad(X, ((0, 0), (0, 512 - X.shape[1])), mode='constant')
X_train = torch.FloatTensor(X_padded)

# Creating DataLoader for batching
batch_size = 1024
dataset = TensorDataset(X_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
N = X_train.shape[0]
d_model = X_train.shape[1]

model = AnomalyTransformer(N, d_model, hidden_dim=64)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training function with batching
def train(model, train_loader, optimizer, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (data_batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            x_hat, P_list, S_list = model(data_batch)
            min_loss = model.min_loss(x_hat, data_batch, P_list, S_list)
            max_loss = model.max_loss(x_hat, data_batch, P_list, S_list)

            if torch.isnan(min_loss) or torch.isnan(max_loss):
                print("Encountered NaN in loss. Skipping this batch.")
                continue

            min_loss.backward(retain_graph=True)
            max_loss.backward()
            optimizer.step()

            epoch_loss += min_loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}, Min Loss: {min_loss:.4f}, Max Loss: {max_loss:.4f}")

        # Average loss per epoch
        losses.append((epoch_loss/len(train_loader)))

        print(f"Epoch {epoch+1}/{epochs}, Average Min Loss: {epoch_loss / len(train_loader):.4f}")

    # Save model after training
    torch.save(model.state_dict(), 'anomaly_transformer_weights.pth')
    plt.plot(losses)
    plt.show()
# Train the model
train(model, train_loader, optimizer, epochs=10)
