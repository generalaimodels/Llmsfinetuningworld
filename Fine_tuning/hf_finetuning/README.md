
## LoraConfig

- 1. **r** (`int`) → `r`: The LoRA attention dimension or rank (e.g., 8).
- 2. **target_modules** (`Optional[Union[List[str], str]]`) → `target_modules`: Specifies the names of the modules to apply the adapter to. In the YAML, this is set to `null`, meaning no specific modules are currently selected.
- 3. **lora_alpha** (`int`) → `lora_alpha`: The scaling factor for LoRA (e.g., 8).
- 4. **lora_dropout** (`float`) → `lora_dropout`: The dropout probability for LoRA layers (e.g., 0.0).
- 5. **fan_in_fan_out** (`bool`) → `fan_in_fan_out`: Specifies whether the layer stores weight as (fan_in, fan_out) (e.g., false).
- 6. **bias** (`str`) → `bias`: The bias type for LoRA. Can be `'none'`, `'all'`, or `'lora_only'` (e.g., none).
- 7. **use_rslora** (`bool`) → `use_rslora`: If true, uses Rank-Stabilized LoRA (e.g., false).
- 8. **modules_to_save** (`List[str]`) → `modules_to_save`: List of additional modules apart from the adapter layers to be trained and saved in the checkpoint (e.g., null).
- 9. **init_lora_weights** (`bool | Literal["gaussian", "loftq"]`) → `init_lora_weights`: Specifies how to initialize LoRA weights (e.g., `true` for default initialization).
- 10. **layers_to_transform** (`Union[List[int], int]`) → `layers_to_transform`: The indices of the layers to apply the adapter (e.g., null).
- 11. **layers_pattern** (`str`) → `layers_pattern`: The pattern name for layers (e.g., null).
- 12. **rank_pattern** (`dict`) → `rank_pattern`: A mapping from layer names to specific ranks (e.g., `{}`).
- 13. **alpha_pattern** (`dict`) → `alpha_pattern`: A mapping from layer names to specific alpha values (e.g., `{}`).
- 14. **megatron_config** (`Optional[dict]`) → `megatron_config`: Configuration for applying LoRA to Megatron’s layers (e.g., `{}`).
- 15. **megatron_core** (`Optional[str]`) → `megatron_core`: Specifies the Megatron core module to use (e.g., `megatron.core`).
- 16. **loftq_config** (`Optional[LoftQConfig]`) → `loftq_config`: Configuration for LoftQ initialization and quantization (e.g., `{}`).



