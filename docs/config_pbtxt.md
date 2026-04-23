# config.pbtxt Notes

- **Do not reuse an existing config.pbtxt**; regenerate or write it fresh for the current deployment.
- The following fields are **hardcoded** and must be set per machine / requirements:
  - `lora_cache_host_memory_bytes`
  - `lora_cache_gpu_memory_fraction`

(This repo does not include a config.pbtxt; this document is a reference for future trtllm-serve / LoRA configuration.)
