import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from src.model import AIGModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load test data
test_loader = torch.load('loaders/test_loader.pth')

# Define model
model = AIGModel().to(device)
model.load_state_dict(torch.load('model/model_mae.pth'))

# Evaluate the model
model.eval()
all_outputs = []
all_targets = []

with torch.no_grad():
    for graphs in tqdm(test_loader):
        outputs = model(graphs, device).to(device)
        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(graphs.y.cpu().numpy())

for i in range(2):
    print("prediction: ", all_outputs[i][0:])
    print("target: ", all_targets[i][0:5])

# Flatten the list of outputs and targets
all_outputs = np.concatenate(all_outputs, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Compute MAE
mae = mean_absolute_error(all_targets, all_outputs)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Compute MSE
mse = mean_squared_error(all_targets, all_outputs)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Compute MAPE
mape = mean_absolute_percentage_error(all_targets, all_outputs)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
