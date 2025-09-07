#!/usr/bin/env python3
"""
Stable Diffusion LoRA Fine-Tuning Script with PEFT
"""

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse


def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="all",  # Use "wandb" if you prefer
    )
    set_seed(42)

    # Load the pre-trained model
    print("Loading model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Freeze all components
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Set up LoRA using PEFT
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank
        lora_alpha=args.lora_rank * 2,  # Typically 2 * rank
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Attention projection matrices
        lora_dropout=0.0,
        bias="none",
    )

    # Apply LoRA to the UNet
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Print trainable parameters
    pipe.unet.print_trainable_parameters()

    # Enable TF32 for faster math on Ampere GPUs (optional)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("imagefolder", data_dir=args.dataset_dir, split="train")

    # Create dataloader with custom collate function
    def collate_fn(examples):
        images = [example["image"] for example in examples]
        texts = [example["text"] for example in examples]

        # Convert PIL Images to tensors
        pixel_values = torch.stack([
            pipe.image_processor(image, return_tensors="pt").pixel_values[0]
            for image in images
        ])

        # Tokenize the captions
        text_inputs = pipe.tokenizer(
            texts,
            max_length=pipe.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": text_inputs.input_ids}

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Set up optimizer (only optimizes LoRA parameters)
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),  # Only LoRA params are trainable
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader
    )

    # Move other components to accelerator device
    pipe.vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.scheduler.to(accelerator.device)

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latents
                with torch.no_grad():
                    latents = pipe.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                # Add noise to the latents
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = pipe.text_encoder(batch["input_ids"].to(accelerator.device))[0]

                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Log progress
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    accelerator.log({"loss": loss.item()}, step=global_step)
                    print(f"Step {global_step}/{total_training_steps}, Loss: {loss.item():.4f}")

            if global_step >= total_training_steps:
                break

    # Save the LoRA weights
    print("Training complete! Saving weights...")
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Save the entire UNet (with LoRA weights)
    unwrapped_unet.save_pretrained(args.output_dir)

    # Alternatively, just save the LoRA weights
    # unwrapped_unet.save_attn_procs(args.output_dir)

    print(f"LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA using PEFT")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to folder containing images and metadata.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per GPU (default: 1)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm (default: 1.0)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision setting")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log loss every X steps (default: 10)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of data loader workers (default: 0)")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 (default: 0.9)")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 (default: 0.999)")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay (default: 1e-2)")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon (default: 1e-8)")

    args = parser.parse_args()
    main(args)