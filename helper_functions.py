"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""

# HELPER FUNCTION

# import requests
# from pathlib import Path

# # Download helper functions from your new GitHub repo (if not already downloaded)
# helper_functions_url = "https://raw.githubusercontent.com/PatelRaj45/helperfunction/main/helper_functions.py"
# helper_functions_path = "helper_functions.py"

# if Path(helper_functions_path).is_file():
#     print(f"{helper_functions_path} already exists, skipping download")
# else:
#     print(f"Downloading {helper_functions_path}")
#     request = requests.get(helper_functions_url)
#     with open(helper_functions_path, "wb") as f:
#         f.write(request.content)

# # Import functions from helper_functions.py
# from helper_functions import accuracy_fn, train_step, test_step, make_predictions, eval_model,print_train_time




# # Download custom image

# import requests

# # Setup custom image path
# custom_image_path = data_path / "04-pizza-dad.jpeg"

# # Download the image if it doesn't already exist
# if not custom_image_path.is_file():
#   with open(custom_image_path, "wb") as f:
#     # When downloading from GitHub, need to use the "raw" file link
#     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
#     print(f"Downloading {custom_image_path}...")
#     f.write(request.content)
# else:
#   print(f"{custom_image_path} already exists, skipping download...")




import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer
device = "cuda" if torch.cuda.is_available() else "cpu"

from torch import nn

import os
import zipfile

from pathlib import Path

import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: Accuracy percentage (0-100).
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device) -> dict:
    """
    Evaluates a model on a given dataset.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        loss_fn (torch.nn.Module): Loss function.
        accuracy_fn (function): Function to compute accuracy.
        device (torch.device, optional): Device to run evaluation on. Defaults to device.

    Returns:
        dict: Dictionary containing model name, loss, and accuracy.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            loss += loss_fn(y_logit, y).item()
            acc += accuracy_fn(y_true=y, y_pred=torch.argmax(y_logit, dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss,
        "model_acc": acc
    }


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device) -> torch.Tensor:
    """
    Makes predictions on a given dataset.

    Args:
        model (torch.nn.Module): Trained model.
        data (list): List of input data samples.
        device (torch.device, optional): Device for inference. Defaults to device.

    Returns:
        torch.Tensor: Stacked tensor of prediction probabilities.
    """
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)


def print_train_time(start: float, end: float, device: torch.device = None) -> float:
    """
    Prints and returns the total training time.

    Args:
        start (float): Start time of computation.
        end (float): End time of computation.
        device (torch.device, optional): Device the computation ran on. Defaults to None.

    Returns:
        float: Total training time in seconds.
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Required Installations
# !pip install torch torchvision

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import os
from PIL import Image

def train_step(model: nn.Module, dataloader: data.DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """
    Perform a single training step.
    
    Args:
        model (nn.Module): The neural network model.
        dataloader (data.DataLoader): DataLoader for training data.
        loss_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the training (CPU or GPU).
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def test_step(model: nn.Module, dataloader: data.DataLoader, loss_fn: nn.Module, device: torch.device):
    """
    Perform a single testing step.
    
    Args:
        model (nn.Module): The neural network model.
        dataloader (data.DataLoader): DataLoader for test data.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run the testing (CPU or GPU).
    
    Returns:
        float: Average test loss.
        float: Test accuracy.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

def train(model: nn.Module, train_dataloader: data.DataLoader, test_dataloader: data.DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device, epochs: int):
    """
    Train and evaluate the model for multiple epochs.
    
    Args:
        model (nn.Module): The neural network model.
        train_dataloader (data.DataLoader): DataLoader for training data.
        test_dataloader (data.DataLoader): DataLoader for test data.
        loss_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the training (CPU or GPU).
        epochs (int): Number of training epochs.
    
    Returns:
        dict: Dictionary containing training and testing loss and accuracy per epoch.
    """
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    return history

def pred_on_custom_image(model: nn.Module, image_path: str, transform: transforms.Compose, device: torch.device):
    """
    Predict the class of a custom image using the trained model.
    
    Args:
        model (nn.Module): The trained neural network model.
        image_path (str): Path to the image file.
        transform (transforms.Compose): Transformations to apply to the image.
        device (torch.device): Device to run the inference (CPU or GPU).
    
    Returns:
        int: Predicted class index.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(1).item()
    
    return pred_class

def find_classes(directory: str):
    """
    Find class names from a dataset directory.
    
    Args:
        directory (str): Path to the dataset directory.
    
    Returns:
        list: List of class names.
    """
    return sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
