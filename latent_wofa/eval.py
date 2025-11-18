# Evaluation Script

# Import necessary modules
import torch
from models import Stage2Model  # Assuming your model is defined in models
from metrics import WatermarkMetrics  # Assuming your metrics are defined in metrics

# Load the model
model = Stage2Model().to('cuda')  # Assuming using GPU
model.load_state_dict(torch.load('path_to_weights.pth'))  # Load your weights
model.eval()

# Evaluation function
def evaluate(test_loader):
    metrics = WatermarkMetrics()
    results = []

    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        with torch.no_grad():
            outputs = model(inputs)
            result = metrics.evaluate_all(labels, outputs)  # Assuming labels and outputs are your original and watermarked images
            results.append(result)

    return results

# Example usage
# test_loader = DataLoader(test_dataset, batch_size=16)
# evaluation_results = evaluate(test_loader)