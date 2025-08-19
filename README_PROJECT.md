# Multiturn AutoCAD Project

## Mô tả dự án
Dự án tạo CAD models từ mô tả bằng ngôn ngữ tự nhiên, hỗ trợ cả tiếng Anh và tiếng Việt.

## Cấu trúc dự án

**Root directory:**
- `prompt.md`: template prompts cho CAD generation
- `requirements.txt`: Python dependencies
- `.env`: environment variables (HF_TOKEN, API keys)

**config/**
- `default_config.yaml`: cấu hình deepspeed cho multi-GPU training
- `zero_stage_config.json`: cấu hình ZeRO optimizer state partitioning

**src/**
- `sft_ds.py`: Supervised Fine-tuning với DeepSpeed
- `sft_galore.py`: SFT với GaLore memory-efficient optimizer  
- `sft_multi_lora.ipynb`: notebook multiturn training với LoRA adapters

**src/inference/**
- `inference_test.py`: test inference model với single sample
- `gen_test_all.py`: generate predictions cho toàn bộ test dataset
- `gen_test_index.py`: generate predictions cho phạm vi index cụ thể

**src/process_data/**
- `create_reasoning_en.py`: tạo reasoning dataset tiếng Anh với Gemini
- `create_reasoning_vi.py`: tạo reasoning dataset tiếng Việt với Gemini
- `create_multi_en.py`: xử lý multiturn conversation data EN
- `create_multi_vi.py`: xử lý multiturn conversation data VI
- `process_train_no_reasoning.py`: xử lý training data không có reasoning
- `retry_failed_samples_en.py`: retry các samples thất bại EN
- `retry_failed_samples_vi.py`: retry các samples thất bại VI

## Technologies
- **Transformers**: HuggingFace transformers library
- **TRL**: Training với Supervised Fine-tuning
- **DeepSpeed**: Distributed training  
- **LoRA/GaLore**: Parameter-efficient fine-tuning
- **Gemini API**: Tạo synthetic data
- **Wandb**: Experiment tracking

## Environment
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