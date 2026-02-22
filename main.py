import dataloader

if __name__ == "__main__":
    train_loader = dataloader.load_data("train", "augmented")
    test_loader  = dataloader.load_data("test", "augmented")
