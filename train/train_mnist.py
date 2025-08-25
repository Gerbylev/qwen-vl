import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision.transforms as transforms
from datasets import load_dataset


def encode(example):
    image = example["image"]
    label = example["label"]
    x = transforms.ToTensor()(image)
    x = x.flatten()
    y = F.one_hot(torch.tensor(label), num_classes=10).float()
    return {"x": x, "y": y}


def collate_fn(batch):
    x = torch.stack([torch.tensor(item["x"], dtype=torch.float32) for item in batch])
    y = torch.stack([torch.tensor(item["y"], dtype=torch.float32) for item in batch])
    return {"x": x, "y": y}


def get_dataloaders(dataset):
    raw_dataset = load_dataset("ylecun/mnist")
    dataset = raw_dataset.map(encode, remove_columns=["image", "label"])
    train_loader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


class MNISTLightningNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def compute_loss_and_accuracy(self, batch):
        inputs, labels = batch["x"], batch["y"]
        labels = torch.argmax(labels, dim=1)  # Convert one-hot to class indices
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.compute_loss_and_accuracy(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        self.log("train_acc", accuracy, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.compute_loss_and_accuracy(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class CustomPrintingCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"]
        train_acc = trainer.callback_metrics["train_acc"]
        val_loss = trainer.callback_metrics["val_loss"]
        val_acc = trainer.callback_metrics["val_acc"]

        print(f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print()


if __name__ == "__main__":
    model = MNISTLightningNet()
    custom_callback = CustomPrintingCallback()
    train_loader, val_loader = get_dataloaders()
    trainer = L.Trainer(max_epochs=10, callbacks=[custom_callback])
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    """ 
    Epoch 1/10
    Train Loss: 0.3547, Train Acc: 0.8953
    Val Loss: 0.1518, Val Acc: 0.9535

    Epoch 2/10
    Train Loss: 0.1648, Train Acc: 0.9503
    Val Loss: 0.1063, Val Acc: 0.9664

    Epoch 3/10
    Train Loss: 0.1254, Train Acc: 0.9620
    Val Loss: 0.0914, Val Acc: 0.9728

    Epoch 4/10
    Train Loss: 0.1057, Train Acc: 0.9686
    Val Loss: 0.0910, Val Acc: 0.9718

    Epoch 5/10
    Train Loss: 0.0921, Train Acc: 0.9711
    Val Loss: 0.0764, Val Acc: 0.9755

    Epoch 6/10
    Train Loss: 0.0832, Train Acc: 0.9744
    Val Loss: 0.0773, Val Acc: 0.9764

    Epoch 7/10
    Train Loss: 0.0740, Train Acc: 0.9761
    Val Loss: 0.0761, Val Acc: 0.9768

    Epoch 8/10
    Train Loss: 0.0704, Train Acc: 0.9776
    Val Loss: 0.0791, Val Acc: 0.9771

    Epoch 9/10
    Train Loss: 0.0665, Train Acc: 0.9785
    Val Loss: 0.0728, Val Acc: 0.9784

    `Trainer.fit` stopped: `max_epochs=10` reached.
    Epoch 10/10
    Train Loss: 0.0631, Train Acc: 0.9803
    Val Loss: 0.0822, Val Acc: 0.9756
    """