tp=1
bsz=4

model_path=lmsys/longchat-7b-v1.5-32k

for bsz in 1 2 4 8 16
do 
    for prompt_length in 6144 
    do
        for quant_policy in None KIVI GEAR
        do
            quant_bits=4
            if [[ ${quant_policy} == "None" ]]; then
                quant_bits=0
            fi 

            echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == prefill \n"
            CUDA_VISIBLE_DEVICES=0 python -u 0_quant_long_logs/1_quant_prefill_thr.py --batch_size $bsz --tp $tp \
                --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}
            
            echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == decoding \n"
            CUDA_VISIBLE_DEVICES=0 python -u 0_quant_long_logs/0_quant_decoding_thr.py --batch_size $bsz --tp $tp \
                --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}
            echo "------------------------ end \n"
            sleep 2
        done 
    done

done
