import torch
import torch.nn as nn
from pathlib import Path

class EmotionClassifier(nn.Module):

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = nn.BatchNorm2d(out_channels)
            self.r1 = nn.ReLU()

            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = nn.BatchNorm2d(out_channels)
            self.r2 = nn.ReLU()

            if in_channels != out_channels or stride != 1:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x0):
            x = self.r1(self.n1(self.c1(x0)))

            x = self.n2(self.c2(x))
            x = x + self.skip(x0)
            x = self.r2(x)

            return x

    def __init__(self, in_channels, num_classes, num_blocks = 2):
        super().__init__()

        c1 = 64
        cnn_layers = [
            torch.nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        ]
        for _ in range(num_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x):
        features = self.network(x)
        logits = features.mean(dim=(2, 3))

        return logits

    def predict(self, x):
        return self(x).argmax(dim=1)
        
def save_model(model, model_name="emotion_classifier", root="models"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    model_path = root / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")
    return model_path

def load_model(device, model_name="emotion_classifier", root="models"):
    model = EmotionClassifier(in_channels=1, num_classes=7)

    root = Path(root)
    model_file = root / f"{model_name}.pt"

    if model_file.exists():
        print(f"Loading model from {model_file}")
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
    else:
        print("No saved model found — initializing new model")

    return model