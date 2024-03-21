import numpy as np

# Define the softmax function
def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

# Generate synthetic data
np.random.seed(0)  # For reproducibility
data = np.random.randn(3, 100)  # Simulating 100 samples with 3 features each

# Generate synthetic labels for visualization purposes
true_labels = np.argmax(data, axis=0)  # Assume the highest feature value as 'true' label

# Apply softmax to the data
predicted_probabilities = softmax(data)

# Predict labels based on the highest probability from softmax
predicted_labels = np.argmax(predicted_probabilities, axis=0)

# Compare predicted labels with true labels (in a real scenario, these would come from the dataset)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy based on simulated data: {accuracy * 100:.2f}%")

# Example usage of softmax results
print("Predicted probability distribution for the first sample:", predicted_probabilities[:, 0])
