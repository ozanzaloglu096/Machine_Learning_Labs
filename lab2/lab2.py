import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# 2. Scale inputs by dividing by 16 (ranges between 0 and 1)
X_scaled = X / 16.0

# 3. One-hot encode target labels
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

# 4. Stratified split (70% train / 30% test) with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.30, stratify=y, random_state=42
)

# Sanity check
print(f"Train features shape: {X_train.shape}")
print(f"Test features shape:  {X_test.shape}")
print(f"Train labels shape:   {y_train.shape}")
print(f"Test labels shape:    {y_test.shape}")

# 2. Initialize network parameters for given hidden width W
def initialize_parameters(input_dim=64, hidden_dim=16, output_dim=10):
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    b2 = np.zeros(hidden_dim)
    
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    b3 = np.zeros(output_dim)
    
    parameters = {
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3
    }
    return parameters

# Test for W = 16 and W = 64 as specified in the lab manual
params_16 = initialize_parameters(hidden_dim=16)
params_64 = initialize_parameters(hidden_dim=64)

print("\n--- W = 16 Parameter Shapes ---")
for key, val in params_16.items():
    print(f"{key}: {val.shape}")

print("\n--- W = 64 Parameter Shapes ---")
for key, val in params_64.items():
    print(f"{key}: {val.shape}")
# ReLU activation function
def relu(Z):
    return np.maximum(0, Z)

# Softmax activation function for output layer
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True)) # Numeric stability
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

# 3. Forward propagation layer by layer
def forward_propagation(X, parameters):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    
    # Layer 1 (Hidden 1)
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # Layer 2 (Hidden 2)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # Layer 3 (Output Softmax)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    
    # Cache intermediate values for backpropagation later
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

# Sanity check for Forward Propagation (W = 16)
sample_batch = X_train[:32] # Batch size 32
output_probs, cache = forward_propagation(sample_batch, params_16)

print("\n--- Step 3: Forward Propagation Sanity Check ---")
print(f"Output shape (batch, 10): {output_probs.shape}")
row_sums = np.sum(output_probs, axis=1)
print(f"Row sums (should be all ~1.0 within 1e-6):")
print(row_sums[:5]) # Print first 5 rows
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row sums do not equal 1!"
print("Forward propagation check passed successfully!")
# 4. Implement cross-entropy loss
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Adding a small epsilon for numerical stability (prevent log(0))
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss

# Sanity check for untrained network loss
untrained_probs, _ = forward_propagation(X_train[:32], params_16)
initial_loss = compute_loss(y_train[:32], untrained_probs)

print("\n--- Step 4: Cross-Entropy Loss Sanity Check ---")
print(f"Initial loss of untrained network: {initial_loss:.4f}")
print(f"Expected theoretical loss (ln(10)): {np.log(10):.4f}")
# 5. Backpropagation implementation
def backward_propagation(X, y_true, parameters, cache):
    m = X.shape[0]
    W1, W2, W3 = parameters['W1'], parameters['W2'], parameters['W3']
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    
    # Output layer gradient
    dZ3 = A3 - y_true
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0) / m
    
    # Hidden layer 2 gradient
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (A2 > 0) # Derivative of ReLU
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    
    # Hidden layer 1 gradient
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (A1 > 0) # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    
    gradients = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2,
        'dW3': dW3, 'db3': db3
    }
    return gradients

# Sanity check for Backpropagation
sample_batch = X_train[:32]
sample_labels = y_train[:32]
probs, cache = forward_propagation(sample_batch, params_16)
grads = backward_propagation(sample_batch, sample_labels, params_16, cache)

print("\n--- Step 5: Backpropagation Gradient Shapes ---")
for key, val in grads.items():
    print(f"{key}: {val.shape}")
print("Backpropagation shapes verified successfully!")
# 6. Training Loop and Evaluation
def train_network(X_train, y_train, X_test, y_test, hidden_dim=16, epochs=50, learning_rate=0.1, batch_size=64):
    # Initialize parameters
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    parameters = initialize_parameters(input_dim, hidden_dim, output_dim)
    
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle training data each epoch
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward propagation
            probs, cache = forward_propagation(X_batch, parameters)
            
            # Backpropagation
            grads = backward_propagation(X_batch, y_batch, parameters, cache)
            
            # Gradient descent parameter update
            parameters['W1'] -= learning_rate * grads['dW1']
            parameters['b1'] -= learning_rate * grads['db1']
            parameters['W2'] -= learning_rate * grads['dW2']
            parameters['b2'] -= learning_rate * grads['db2']
            parameters['W3'] -= learning_rate * grads['dW3']
            parameters['b3'] -= learning_rate * grads['db3']
            
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_probs, _ = forward_propagation(X_train, parameters)
            loss = compute_loss(y_train, train_probs)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
            
    return parameters

def evaluate_accuracy(X, y_true, parameters):
    probs, _ = forward_propagation(X, parameters)
    predictions = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy

# Train for W = 16
print("\n--- Step 6: Training Network with W = 16 ---")
trained_params_16 = train_network(X_train, y_train, X_test, y_test, hidden_dim=16, epochs=50, learning_rate=0.1)

# Evaluate on Test Set
test_accuracy = evaluate_accuracy(X_test, y_test, trained_params_16)
print(f"\nFinal Test Accuracy (W = 16): {test_accuracy:.2f}%")
