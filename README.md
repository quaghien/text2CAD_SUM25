# Bilingual-Text-to-CAD-Model-Generation-English-Vietnamese (Code Train)

## Project Description
A project for generating CAD models from natural language descriptions, supporting both English and Vietnamese.

## Project Structure

**Root directory:**
- `prompt.md`: template prompts for CAD generation
- `requirements.txt`: Python dependencies
- `.env`: environment variables (HF_TOKEN, API keys)

**config/**
- `default_config.yaml`: deepspeed configuration for multi-GPU training
- `zero_stage_config.json`: ZeRO optimizer state partitioning configuration

**src/**
- `sft_ds.py`: Supervised Fine-tuning with DeepSpeed
- `sft_galore.py`: SFT with GaLore memory-efficient optimizer  
- `sft_multi_lora.ipynb`: notebook multiturn training with LoRA adapters

**src/inference/**
- `inference_test.py`: test inference model with single sample
- `gen_test_all.py`: generate predictions for entire test dataset
- `gen_test_index.py`: generate predictions for specific index range

**src/process_data/**
- `create_reasoning_en.py`: create English reasoning dataset with Gemini
- `create_reasoning_vi.py`: create Vietnamese reasoning dataset with Gemini
- `create_multi_en.py`: process multiturn conversation data EN
- `create_multi_vi.py`: process multiturn conversation data VI
- `process_train_no_reasoning.py`: process training data without reasoning
- `retry_failed_samples_en.py`: retry failed samples EN
- `retry_failed_samples_vi.py`: retry failed samples VI

## Technologies
- **Transformers**: HuggingFace transformers library
- **TRL**: Training with Supervised Fine-tuning
- **DeepSpeed**: Distributed training  
- **LoRA/GaLore**: Parameter-efficient fine-tuning
- **Gemini API**: Generate synthetic data
- **Wandb**: Experiment tracking

## Environment
```bash
# install torch and nvcc match =< cuda driver version
# install nvcc in https://anaconda.org/nvidia/cuda-nvcc
# conda install cuda -c nvidia/label/cuda-12.8.0
# conda install nvidia/label/cuda-12.8.1::cuda-nvcc
# conda install -c nvidia cuda-toolkit=12.8

# pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
# python -c "import torch; print(torch.cuda.is_available())"
# check: nvcc --version ; which nvcc
# pip install flash-attn --no-build-isolation
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port=29501 --config_file default_config.yaml sft_ds.py
```
