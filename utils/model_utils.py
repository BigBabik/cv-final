import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # For progress bars


class FundamentalMatrixModel(nn.Module):
    def __init__(self):
        super(FundamentalMatrixModel, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, K1, R1, T1, K2, R2, T2):
        """
        Forward pass to predict the fundamental matrix.

        Args:
            K1, R1, T1: Intrinsic matrix, rotation, and translation for image 1.
            K2, R2, T2: Intrinsic matrix, rotation, and translation for image 2.

        Returns:
            Predicted fundamental matrix (batch_size x 3 x 3).
        """
        x = torch.cat([
            K1.flatten(start_dim=1), R1.flatten(start_dim=1), T1,
            K2.flatten(start_dim=1), R2.flatten(start_dim=1), T2
        ], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        F_pred = self.fc3(x).view(-1, 3, 3)
        return F_pred


def sampson_loss(F_pred, F_gt):
    """
    Compute the Sampson loss between predicted and ground truth matrices.

    Args:
        F_pred: Predicted fundamental matrix (batch_size x 3 x 3).
        F_gt: Ground truth fundamental matrix (batch_size x 3 x 3).

    Returns:
        Mean Sampson loss over the batch.
    """
    return torch.mean(torch.norm(F_pred - F_gt, dim=(1, 2)))


def train_model(model, data_loader, learn_rate=0.001, num_epochs=20, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress:
            # Move batch data to the correct device
            K1 = batch['K1'].to(device)
            R1 = batch['R1'].to(device)
            T1 = batch['T1'].to(device)
            K2 = batch['K2'].to(device)
            R2 = batch['R2'].to(device)
            T2 = batch['T2'].to(device)
            F_gt = batch['F_gt'].to(device)

            # Forward pass
            optimizer.zero_grad()
            F_pred = model(K1, R1, T1, K2, R2, T2)

            # Normalize predictions and ground truth
            norm_pred = torch.norm(F_pred, dim=(1, 2), keepdim=True)
            norm_gt = torch.norm(F_gt, dim=(1, 2), keepdim=True)
            F_pred = F_pred / norm_pred
            F_gt = F_gt / norm_gt

            # Compute loss and backpropagation
            loss = sampson_loss(F_pred, F_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")

    return model


