import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model.qwen2_5_vl import Qwen2VL, Qwen2Config
from model.vision import VisionConfig
from model.processor import Processor


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, max_length=512):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if "conversations" in item:
            # Anthropic/ShareGPT format
            messages = item["conversations"]
        elif "messages" in item:
            # Standard chat format
            messages = item["messages"]
        elif "image" in item and "text" in item:
            # Image-text pair format
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item.get("question", "Describe this image.")}
                    ]
                },
                {"role": "assistant", "content": item["text"]}
            ]
        else:
            raise ValueError(f"Unsupported dataset format: {item.keys()}")

        # Process the conversation
        processed = self.processor(messages)
        
        # Create labels (shift input_ids for language modeling)
        input_ids = processed["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask user tokens (only train on assistant responses)
        # This is a simple approach - you might want more sophisticated masking
        labels[:-1] = -100  # Mask everything except the last token for now
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            if processed["pixels"] is not None:
                # Adjust d_image if needed for truncation
                pass
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixels": processed["pixels"],
            "d_image": processed["d_image"]
        }


def collate_fn(batch):
    # Find the maximum sequence length in the batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    labels = []
    all_pixels = []
    all_d_image = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids and labels
        padded_input_ids = torch.cat([
            item["input_ids"], 
            torch.full((pad_len,), 0, dtype=torch.long)  # 0 is typically pad token
        ])
        padded_labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        
        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        
        # Handle pixels and d_image
        if item["pixels"] is not None:
            all_pixels.append(item["pixels"])
            all_d_image.append(item["d_image"])
    
    # Stack tensors
    batch_input_ids = torch.stack(input_ids)
    batch_labels = torch.stack(labels)
    
    # Handle multimodal data
    if all_pixels:
        batch_pixels = torch.cat(all_pixels, dim=0)
        batch_d_image = torch.cat(all_d_image, dim=0)
    else:
        batch_pixels = None
        batch_d_image = None
    
    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "pixels": batch_pixels,
        "d_image": batch_d_image
    }


class Qwen2VLForTraining(L.LightningModule):
    def __init__(self, config: Qwen2Config, learning_rate: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = Qwen2VL(config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = learning_rate
        
        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, pixels=None, d_image=None):
        return self.model(input_ids=input_ids, pixels=pixels, d_image=d_image)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        pixels = batch["pixels"]
        d_image = batch["d_image"]
        
        # Forward pass
        logits = self.forward(input_ids=input_ids, pixels=pixels, d_image=d_image)
        
        # Calculate loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        self.train_losses.append(loss.detach().cpu())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        pixels = batch["pixels"]
        d_image = batch["d_image"]
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, pixels=pixels, d_image=d_image)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        self.val_losses.append(loss.detach().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        # Print epoch statistics
        avg_train_loss = torch.stack(self.train_losses[-100:]).mean()  # Last 100 steps
        print(f"Epoch {self.current_epoch}: Avg Train Loss: {avg_train_loss:.4f}")

    def save_pretrained(self, path: str):
        """Save model in HuggingFace format"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config
        }, f"{path}/pytorch_model.bin")


def create_model_config() -> tuple[Qwen2Config, Processor]:
    """Create model configuration for Qwen2.5-VL"""
    
    # Vision configuration
    vision_config = VisionConfig(
        n_embed=1280,
        n_layer=24,
        n_heads=16,
        output_n_embed=3584,  # Should match the LLM embedding size
        in_channels=3,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
        intermediate_size=5120,  # For Qwen2.5-VL gated MLP
        hidden_act="quick_gelu"
    )
    
    # Main model configuration
    config = Qwen2Config(
        n_embed=3584,
        n_heads=28,
        n_kv_heads=4,
        n_layer=28,
        n_mlp=18944,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        vocab_size=152064,
        tie_word_embeddings=False,
        vision_config=vision_config,
        head_dim=128
    )
    
    # Create processor
    processor = Processor(
        repo_id="Qwen/Qwen2.5-3B",  # Use existing tokenizer
        vision_config=vision_config
    )
    
    return config, processor


def load_multimodal_dataset(dataset_name: str = "HuggingFaceM4/VQAv2", split: str = "train"):
    """Load a multimodal dataset"""
    # You can replace this with other multimodal datasets:
    # - "nlphuji/flickr30k"
    # - "HuggingFaceM4/COCO" 
    # - "MMInstruction/M3IT"
    # - Your custom dataset
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        # Fallback to a simple synthetic dataset for testing
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """Create a simple synthetic dataset for testing"""
    print("Creating synthetic dataset for testing...")
    
    # Create dummy data
    dummy_data = []
    for i in range(100):  # Small dataset for testing
        # Create a simple colored image
        img = Image.new('RGB', (224, 224), color=(i*10 % 255, 50, 100))
        
        dummy_data.append({
            "image": img,
            "question": f"What color is dominant in this image {i}?",
            "text": f"The dominant color in this image is a shade of red-purple (RGB: {i*10 % 255}, 50, 100)."
        })
    
    return dummy_data


class CustomTrainingCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer.callback_metrics, "train_loss_epoch"):
            train_loss = trainer.callback_metrics["train_loss_epoch"]
            print(f"\nEpoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
        
        if hasattr(trainer.callback_metrics, "val_loss"):
            val_loss = trainer.callback_metrics["val_loss"]
            print(f"Val Loss: {val_loss:.4f}")
        print()


def main():
    # Configuration
    batch_size = 2  # Small batch size for memory efficiency
    max_epochs = 3
    learning_rate = 1e-5
    max_length = 512
    
    print("Setting up multimodal training...")
    
    # Create model and processor
    config, processor = create_model_config()
    model = Qwen2VLForTraining(config, learning_rate=learning_rate)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_multimodal_dataset("HuggingFaceM4/VQAv2", "train")
    val_dataset = load_multimodal_dataset("HuggingFaceM4/VQAv2", "validation")
    
    # If datasets failed to load, use synthetic data
    if isinstance(train_dataset, list):
        print("Using synthetic data for training")
        val_dataset = train_dataset[-20:]  # Last 20 samples for validation
        train_dataset = train_dataset[:-20]  # Rest for training
    else:
        # Limit dataset size for faster training
        train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(200, len(val_dataset))))
    
    # Create PyTorch datasets
    train_torch_dataset = MultimodalDataset(train_dataset, processor, max_length=max_length)
    val_torch_dataset = MultimodalDataset(val_dataset, processor, max_length=max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_torch_dataset)} samples")
    print(f"Val dataset: {len(val_torch_dataset)} samples")
    
    # Setup trainer
    callback = CustomTrainingCallback()
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Use GPU if available, otherwise CPU
        devices=1,
        callbacks=[callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Effective batch size = 2 * 4 = 8
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    print("Starting training...")
    
    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("Training completed!")
    
    # Plot training curves
    if model.train_losses and model.val_losses:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([x.item() for x in model.train_losses[::10]], label="Train Loss")  # Sample every 10 steps
        plt.title("Training Loss")
        plt.xlabel("Steps (x10)")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot([x.item() for x in model.val_losses], label="Validation Loss")
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.show()
        print("Training curves saved to training_curves.png")
    
    # Save the trained model
    model.save_pretrained("./trained_multimodal_model")
    print("Model saved to ./trained_multimodal_model")


if __name__ == "__main__":
    # Run with: PYTHONPATH=. python train/train_multimodal.py
    main()