import torch
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn
from model import ToxicityClassifierLSTM
from tqdm import tqdm


HIDDEN_DIM = 128
OUTPUT_DIM = 1
NUM_LAYERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(model, train_loader, optimizer, criterion, device):
    """
    Train a ToxicityClassifierLSTM model on given data.

    Parameters
    ----------
    model : ToxicityClassifierLSTM
        The model to be trained.
    train_loader : DataLoader
        DataLoader containing the training data.
    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters.
    criterion : torch.nn.Module
        Loss function used to evaluate the model.
    device : torch.device
        The device (CPU or GPU) on which the model is evaluated.

    Returns
    -------
    A tuple containing:
    - Average loss over the training dataset.
    - Accuracy of the model on the training dataset.
    """
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        predictions = model(inputs).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_preds.extend((predictions > 0.5).int().cpu().numpy())
        all_labels.extend(labels.int().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(train_loader), accuracy


def evaluate_step(model, test_loader, criterion, device):
    """
    Evaluate the model on the test data.

    Args:
        model: The model to be evaluated.
        test_loader: DataLoader for the test dataset.
        criterion: Loss function used to evaluate the model.
        device: The device (CPU or GPU) on which the model is evaluated.

    Returns:
        A tuple containing:
        - Average loss over the test dataset.
        - Accuracy of the model on the test dataset.
    """

    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            predictions = model(inputs).squeeze(1)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            all_preds.extend((predictions > 0.5).int().cpu().numpy())
            all_labels.extend(labels.int().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(test_loader), accuracy


def train(train_loader, test_loader, glove_embeddings, epochs=5):
    # Initialize model
    """
    Train a ToxicityClassifierLSTM model on given data.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader containing the training data.
    test_loader : DataLoader
        DataLoader containing the test data.
    glove_embeddings : torch.Tensor
        Pre-trained GloVe embeddings.

    Returns
    -------
    model : ToxicityClassifierLSTM
        The trained model.
    """
    model = ToxicityClassifierLSTM(
        embedding_matrix=glove_embeddings,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
    )
    model.to(DEVICE)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction='mean') #nn.BCELoss()

    # Train and evaluate
    EPOCHS = epochs
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        test_loss, test_acc = evaluate_step(model, test_loader, criterion, DEVICE)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return model
