import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import mean_absolute_percentage_error
from datasets.aig_dataloader import load_data
from src.model import AIGModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

target_name = 'alu2'
train_data_path = f'data/{target_name}.pt'

# Load data
train_loader, valid_loader, test_loader = load_data(batch_size=256,
                                                    aig_tensor_path='data/origin/aig_tensor.pt',
                                                    train_data_path=train_data_path,
                                                    )
# Save loaders
torch.save(train_loader, 'loaders/train_loader.pth')
torch.save(valid_loader, 'loaders/valid_loader.pth')
torch.save(test_loader, 'loaders/test_loader.pth')
print("Data loaded successfully")

# Define model
model = AIGModel().to(device)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)

# Training loop with early stopping
num_epochs = 100  # You can set a high number due to early stopping
best_valid_loss = float('inf')
num_epochs_no_improve = 0
best_epoch = 0

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for graphs in tqdm(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(graphs, device).to(device).squeeze()
        labels = graphs.y.to(device).squeeze()

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    # Validation phase
    model.eval()
    total_valid_loss = 0
    total_valid_mape = 0
    with torch.no_grad():
        for graphs in valid_loader:
            outputs = model(graphs, device).to(device).squeeze()
            labels = graphs.y.to(device).squeeze()

            loss = criterion(outputs, labels)
            mape = mean_absolute_percentage_error(labels.cpu().numpy(),
                                                  outputs.cpu().numpy())
            total_valid_loss += loss.item()
            total_valid_mape += mape

    avg_valid_loss = total_valid_loss / len(valid_loader)
    avg_valid_mape = total_valid_mape / len(valid_loader)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_valid_loss:.4f}")
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Validation MAPE: {avg_valid_mape:.4f}")

    # Check for improvement
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'model/model_mae.pth')
        print("Best model saved with Validation Loss: {:.4f}".format(
            best_valid_loss))

print("Training complete")
print(
    f"Best model found at epoch {best_epoch+1} with Validation Loss: {best_valid_loss:.4f}")
