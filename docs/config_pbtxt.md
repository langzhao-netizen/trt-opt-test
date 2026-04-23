# config.pbtxt 使用说明

- **不要复用已有的 config.pbtxt**；需按当前部署重新生成或手写。
- 其中会**硬编码**以下字段，需根据本机/需求设置：
  - `lora_cache_host_memory_bytes`
  - `lora_cache_gpu_memory_fraction`

（本仓库当前未包含 config.pbtxt；此文档供后续 trtllm-serve / LoRA 等配置时参考。）
