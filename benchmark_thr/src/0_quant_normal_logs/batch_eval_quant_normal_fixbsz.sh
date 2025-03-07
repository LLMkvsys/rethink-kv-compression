tp=1
quant_bits=4

model_path=lmsys/longchat-7b-v1.5-32k # to support long-sequence 

for bsz in 1 
do 
    for prompt_length in 256 512 1024 2048 3072 4000 
    do
        for quant_policy in KIVI GEAR None
        do
            quant_bits=4
            if [[ ${quant_policy} == "None" ]]; then
                quant_bits=0
            fi 
            echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == prefill \n"
            CUDA_VISIBLE_DEVICES=0 python -u 0_quant_normal_logs/1_quant_prefill_thr.py --batch_size $bsz --tp $tp \
                --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}

            echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == decoding \n"
            CUDA_VISIBLE_DEVICES=0 python -u 0_quant_normal_logs/0_quant_decoding_thr.py --batch_size $bsz --tp $tp \
                --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}
            echo "------------------------ end \n"
            sleep 2
        done 
    done 
done 
