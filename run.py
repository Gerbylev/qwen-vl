import os
import re
import torch

# Disable HuggingFace progress bars globally
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from model.vision import VisionConfig
from model.processor import Processor
from model.qwen2_5_vl import Qwen2VL
from rich.console import Console
console = Console(highlight=False)

hf_repo_id = 'Qwen/Qwen2.5-VL-3B-Instruct'

def parse_user_input(text):
    """Convert @path/to/image.jpg syntax to standard messages format."""
    image_pattern = r"@([^\s]+\.(?:jpg|jpeg|png|gif|webp))"
    matches = list(re.finditer(image_pattern, text, re.IGNORECASE))

    if not matches:
        # No images, return simple text message
        return [{"role": "user", "content": text}]

    # Build content list with text and images
    content = []
    last_end = 0

    for match in matches:
        # Add text before image
        if match.start() > last_end:
            text_part = text[last_end : match.start()].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})

        # Add image
        image_path = match.group(1)
        if os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
            console.print(f"✓ Found image: {image_path}", style="green")
        else:
            console.print(f"Warning: Image not found: {image_path}", style="yellow")

        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})

    return [{"role": "user", "content": content}]


def generate_local_response(
    messages, model, processor, max_tokens=2048, stream=False
):
    """Generate response using local model."""
    # Use processor directly - it now handles both message formats
    inputs = processor(messages)

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs["input_ids"] = inputs["input_ids"].to(device)
    if inputs["pixels"] is not None:
        inputs["pixels"] = inputs["pixels"].to(device)
    if inputs["d_image"] is not None:
        inputs["d_image"] = inputs["d_image"].to(device)

    # Default stop tokens for Qwen models
    stop_tokens = [151645, 151644, 151643]  # <|im_end|>, <|im_start|>, <|endoftext|>

    # Generate
    with torch.no_grad():
        if inputs["pixels"] is not None:
            # Vision model with images
            generation = model.generate(
                input_ids=inputs["input_ids"],
                pixels=inputs["pixels"],
                d_image=inputs["d_image"],
                max_new_tokens=max_tokens,
                stop_tokens=stop_tokens,
                stream=stream,
            )
        else:
            # Vision model, text-only
            generation = model.generate(
                input_ids=inputs["input_ids"],
                pixels=None,
                d_image=None,
                max_new_tokens=max_tokens,
                stop_tokens=stop_tokens,
                stream=stream,
            )
    if stream:
        # Streaming: yield decoded tokens one by one
        for token_id in generation:
            token_text = processor.tokenizer.decode([token_id])
            yield token_text
    else:
        # Non-streaming: decode full response
        input_length = inputs["input_ids"].shape[1]
        response_ids = generation[:, input_length:]
        response = processor.tokenizer.decode(response_ids[0].tolist())
        return response

def main():
    try:
        model = Qwen2VL.from_pretrained(hf_repo_id)
        console.print("Model loaded successfully!")
    except Exception as e:
        console.print(f"Failed to load model: {e}")
        return

    vision_config = VisionConfig(
        n_embed=model.config.vision_config.n_embed,
        n_layer=model.config.vision_config.n_layer,
        n_heads=model.config.vision_config.n_heads,
        output_n_embed=model.config.n_embed,
        in_channels=model.config.vision_config.in_channels,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        spatial_patch_size=model.config.vision_config.spatial_patch_size,
        temporal_patch_size=model.config.vision_config.temporal_patch_size,
        intermediate_size=getattr(
            model.config.vision_config, "intermediate_size", None
        ),
        hidden_act=getattr(
            model.config.vision_config, "hidden_act", "quick_gelu"
        ),
    )
    processor = Processor(repo_id=hf_repo_id, vision_config=vision_config)

    messages=[]
    current_messages = parse_user_input("Привет, что изображено на картинке @/home/oleg/Pictures/photo_2025-08-19_21-52-52.jpg ?")
    messages.extend(current_messages)

    try:
        # Generate response using local model with streaming
        print("ASSISTANT: ", end="", flush=True)

        response_tokens = []
        for token in generate_local_response(
                current_messages,
                model,
                processor,
                stream=True,
        ):
            print(token, end="", flush=True)
            response_tokens.append(token)

        # Complete the response
        print()  # New line after streaming
        response = "".join(response_tokens).strip()

        # Add assistant's response to conversation
        messages.append({"role": "assistant", "content": response})

    except Exception as e:
        console.print(f"Error generating response: {e}", style="red")
        # Remove the failed user message
        if messages and messages[-1]["role"] == "user":
            messages.pop()


if __name__ == "__main__":
    print("Запуск")
    main()