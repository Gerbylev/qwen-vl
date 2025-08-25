import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from models.qwen2 import Qwen2ForCausalLM
from data.anthropic import init_anthropic_dataloader_for_sft_example
from huggingface_hub import PyTorchModelHubMixin


class Qwen2ForSFT(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.qwen_model = Qwen2ForCausalLM(config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.train_loss = []
        self.val_loss = []

    def forward(self, input_ids):
        return self.qwen_model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.val_loss.append(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        base_model = Qwen2ForCausalLM.from_pretrained(*args, **kwargs)
        qwen2_for_sft = cls(base_model.config)
        qwen2_for_sft.qwen_model = base_model
        return qwen2_for_sft


if __name__ == "__main__":

    # run with:
    #   PYTHONPATH=. python train/train_sft.py

    model = Qwen2ForSFT.from_pretrained(
        repo_id="Qwen/Qwen2.5-3B",
        local_dir="cache/model/Qwen2.5-3B",
    )
    tokenizer = Tokenizer.from_file(path="cache/model/Qwen2.5-3B/tokenizer.json")
    tokenizer.add_special_tokens(["<|pad|>"])
    train_dataloader = init_anthropic_dataloader_for_sft_example(
        tokenizer, batch_size=4, split="train"
    )
    val_dataloader = init_anthropic_dataloader_for_sft_example(
        tokenizer, batch_size=4, split="test"
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        logger=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    plt.plot(model.train_loss, label="train_loss")
    plt.plot(model.val_loss, label="val_loss")
    plt.legend()
    plt.show()

    # model.save_pretrained("cache/model/Qwen2.5-3B-SFT")
    # model.push_to_hub("iiTzEddy/Qwen2.5-3B-SFT")