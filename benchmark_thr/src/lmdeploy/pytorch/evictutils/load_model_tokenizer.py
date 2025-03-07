from transformers import LlamaConfig, MistralConfig, AutoTokenizer

def load_model_tokenizer(dtype, parsed_args_dict, perform_timing=False):
    model_args = parsed_args_dict["model_args"]
    training_args = parsed_args_dict["training_args"]

    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                            use_fast=False if 'Llama-3' not in model_args.model_name_or_path else True,
                                            trust_remote_code=True,
                                            tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                            use_fast=False,
                                            trust_remote_code=True)
    else:
        raise NotImplementedError

    # Llama / LongChat models
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        if model_args.quant_method == 'kivi':
            from models.llama_kivi import LlamaForCausalLM_KIVI
            kivi_args = parsed_args_dict["kivi_args"]
            if kivi_args.k_bits < 16 and kivi_args.v_bits < 16:
                config.k_bits = kivi_args.k_bits
                config.v_bits = kivi_args.v_bits
                config.group_size = kivi_args.group_size
                config.residual_length = kivi_args.residual_length
                # NOTE: XINYUZHOU, disable this for non-Ampere GPU
                # config.use_flash = True # Note: We activate the flashattention to speed up the inference
                config.use_flash = False
                model = LlamaForCausalLM_KIVI.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                )
        elif model_args.quant_method == 'gear':
            from GEARLM import GearLlamaForCausalLM
            gear_args = parsed_args_dict["gear_args"]
            compress_config = {
                "compress_mode": gear_args.compress_mode,
                "quantize_bit": gear_args.quantize_bit,
                "left": gear_args.left,
                "rank": gear_args.rank,
                "loop": gear_args.loop,
                "stream": gear_args.stream,
                "streaming_gap": gear_args.streaming_gap
            }
            model = GearLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                compress_config=compress_config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        elif model_args.eviction_method in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
            import torch
            from utils.sparsekv_monkeypatch import replace_llama, replace_mistral, replace_llama_forward_timing
            from transformers import LlamaForCausalLM

            sparsekv_args = parsed_args_dict["sparsekv_args"]
            max_capacity_prompts = sparsekv_args.max_capacity_prompts
            window_sizes = sparsekv_args.window_sizes
            kernel_sizes = sparsekv_args.kernel_sizes
            pooling = sparsekv_args.pooling

            replace_llama(model_args.eviction_method)
            print(f"replace llama successfully")
            if perform_timing:
                replace_llama_forward_timing()
                print(f"replace llama forward timing successfully")

            print(f"current attn_implementation: {sparsekv_args.attn_impl}")
            model = LlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    cache_dir=training_args.cache_dir,
                    device_map="auto",
                    use_cache=True,
                    attn_implementation=sparsekv_args.attn_impl
                )

            if model_args.eviction_method in ["snapkv","pyramidkv","h2o"]:
                window_sizes = 8
            elif model_args.eviction_method in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
        else:
            # customized LlamaForCausalLM for timing
            if perform_timing:
                from models.llama_hf import LlamaForCausalLM
            else:
                from transformers import LlamaForCausalLM

            config._attn_implementation = 'eager'
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                use_flash_attention_2=False,
                device_map="auto",
            )
    # Mistral models
    elif 'mistral' in model_args.model_name_or_path.lower():
        if model_args.quant_method == 'kivi':
            kivi_args = parsed_args_dict["kivi_args"]
            if kivi_args.k_bits < 16 and kivi_args.v_bits < 16:
                from models.mistral_kivi import MistralForCausalLM_KIVI
                config.k_bits = kivi_args.k_bits
                config.v_bits = kivi_args.v_bits
                config.group_size = kivi_args.group_size
                config.residual_length = kivi_args.residual_length
                # config.use_flash = True
                config.use_flash = False
                model = MistralForCausalLM_KIVI.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                )
        elif model_args.quant_method == 'gear':
            from GEARLM import GearMistralForCausalLM
            gear_args = parsed_args_dict["gear_args"]
            compress_config = {
                "compress_mode": gear_args.compress_mode,
                "quantize_bit": gear_args.quantize_bit,
                "left": gear_args.left,
                "rank": gear_args.rank,
                "loop": gear_args.loop,
                "stream": gear_args.stream,
                "streaming_gap": gear_args.streaming_gap
            }
            model = GearMistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                compress_config=compress_config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        elif model_args.eviction_method in ["snapkv", "pyramidkv", "h2o", "streamingllm"]:
            import torch
            from utils.sparsekv_monkeypatch import replace_llama, replace_mistral, replace_mistral_forward_timing
            from transformers import MistralForCausalLM

            sparsekv_args = parsed_args_dict["sparsekv_args"]
            max_capacity_prompts = sparsekv_args.max_capacity_prompts
            window_sizes = sparsekv_args.window_sizes
            kernel_sizes = sparsekv_args.kernel_sizes
            pooling = sparsekv_args.pooling

            replace_mistral(model_args.eviction_method)
            print(f"replace mistral successfully")
            if perform_timing:
                replace_mistral_forward_timing()
                print(f"replace mistral forward timing successfully")

            print(f"current attn_implementation: {sparsekv_args.attn_impl}")
            model = MistralForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    cache_dir=training_args.cache_dir,
                    device_map="auto",
                    use_cache=True,
                    attn_implementation=sparsekv_args.attn_impl
                )

            if model_args.eviction_method in ["snapkv","pyramidkv","h2o"]:
                window_sizes = 8
            elif model_args.eviction_method in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
        else:
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                use_flash_attention_2=False,
                device_map="auto",
            )

    else:
        raise NotImplementedError
    if 'Llama-3' not in model_args.model_name_or_path: 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
