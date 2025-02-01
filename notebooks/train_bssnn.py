import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import networkx as nx
import matplotlib.pyplot as plt
import shap
from bssnn_model import BSSNN  # Import the BSSNN model class


# Step 1: Generate Synthetic Data
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    return X_train, X_val, y_train, y_val


# Step 2: Train the BSSNN Model
def train_model(model, X_train, y_train, num_epochs=100, lr=0.001):
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Step 3: Evaluate the Model
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        val_preds = (val_outputs >= 0.5).float()  # Convert probabilities to binary predictions

        # Calculate accuracy and AUC
        accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
        auc = roc_auc_score(y_val.numpy(), val_outputs.numpy())

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation AUC: {auc:.4f}")


# Step 4: Extract Feature Importance
def get_feature_importance(model):
    joint_weights = model.fc1_joint.weight.data.numpy()
    marginal_weights = model.fc1_marginal.weight.data.numpy()

    # Average the absolute weights across hidden units
    joint_importance = np.mean(np.abs(joint_weights), axis=0)
    marginal_importance = np.mean(np.abs(marginal_weights), axis=0)

    # Combine joint and marginal importance
    total_importance = joint_importance + marginal_importance
    return total_importance


# Step 5: Create and Plot DAG
def create_and_plot_dag(feature_names, importance_scores):
    G = nx.DiGraph()

    # Add nodes for features and the target
    for feature in feature_names:
        G.add_node(feature)
    G.add_node("Target (y)")

    # Add edges from features to the target, weighted by importance scores
    for feature, importance in zip(feature_names, importance_scores):
        G.add_edge(feature, "Target (y)", weight=importance)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def compute_shapley_values(model, X_train, X_val, feature_names, background_samples=100):
    model.eval()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    def model_wrapper(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            joint = model.fc2_joint(model.relu_joint(model.fc1_joint(x_tensor)))
            marginal = model.fc2_marginal(model.relu_marginal(model.fc1_marginal(x_tensor)))
            logits = joint - marginal
            # Ensure we return a 2D array
            return logits.numpy().reshape(-1)

    # Generate background data
    background = shap.kmeans(X_train.numpy(), background_samples)
    explainer = shap.KernelExplainer(model_wrapper, background)

    # Take a smaller validation sample and compute SHAP values
    val_sample = X_val[:100].numpy()
    shap_values = explainer.shap_values(val_sample)

    print(f"Type of shap_values: {type(shap_values)}")
    print(f"Shape of shap_values: {np.array(shap_values).shape}")

    # Fix: Ensure SHAP values are in the correct format
    # Remove the extra dimension if it exists
    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze(axis=-1)

    print(f"Final shap_values shape: {shap_values.shape}")

    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    print(f"Feature importance shape: {feature_importance.shape}")

    try:
        # Create bar plot
        plt.figure(figsize=(10, 6))
        # Sort features by importance
        idx = np.argsort(feature_importance)
        pos = np.arange(len(feature_names))

        plt.barh(pos, feature_importance[idx])
        plt.yticks(pos, [feature_names[i] for i in idx])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance Based on SHAP Values')
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300)
        plt.close()

        # Try the regular SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            val_sample,
            feature_names=feature_names,
            show=False,
            plot_type="violin"
        )
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        # If all else fails, create a simple bar plot
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)
        plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
        plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('shap_feature_importance_fallback.png', dpi=300)
        plt.close()

    return shap_values, explainer.expected_value


def main():
    X_train, X_val, y_train, y_val = generate_data()

    # Initialize model
    input_size = X_train.shape[1]
    hidden_size = 64
    model = BSSNN(input_size, hidden_size)

    # Train the model
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Extract feature importance
    feature_names = [f"Feature {i + 1}" for i in range(input_size)]
    feature_importance = get_feature_importance(model)
    print("Feature Importance Scores:")
    for feature, importance in zip(feature_names, feature_importance):
        print(f"{feature}: {importance:.3f}")

    # Create and plot DAG
    create_and_plot_dag(feature_names, feature_importance)

    # Compute Shapley values
    hap_values, expected_value = compute_shapley_values(model, X_train, X_val, feature_names, background_samples=100)


# Main Script
if __name__ == "__main__":
    main()
