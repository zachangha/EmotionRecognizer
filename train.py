import torch
import numpy as np
from dataloader import load_data
from model import EmotionClassifier, save_model, load_model, model_path
from compute_accuracy import compute_accuracy

def train(num_epoch: int = 10,
          lr: float = 1e-3,
          batch_size: int = 16,
          seed: int = 2024,
          **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    
    train_dataset = load_data("train", "augmented", batch_size=batch_size)
    test_dataset = load_data("test", "augmented", batch_size=batch_size)

    global_step = 0
    metrics = {"train_acc": [], "test_acc": []}

    for epoch in range(num_epoch):

        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch_idx, (images, labels) in enumerate(train_dataset):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = compute_accuracy(outputs, labels)
            metrics["train_acc"].append(train_loss)

            global_step += 1

        with torch.inference_mode():
            model.eval()

            for images, labels in test_dataset:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                
                test_loss = compute_accuracy(outputs, labels)
                metrics["test_acc"].append(test_loss)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_test_acc = torch.as_tensor(metrics["test_acc"]).mean()

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"test_acc={epoch_test_acc:.4f}"
        )
            
    save_model(model)   
    print(f"Model saved to {model_path}")