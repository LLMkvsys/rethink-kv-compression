# export CUDA_VISIBLE_DEVICES="1,2,3"
# for bsz in 16 # 1 2 4 # 4 8 16 
bsz=1
quant=0

# pip install -e . 

tp=1
bsz=4
quant_bits=4

model_path=lmsys/longchat-7b-v1.5-32k

for bsz in 1 2 4 8 16
do 
    for prompt_length in 6144 
    do
        for sparse_policy in StreamingLLM H2O
        do
            echo "\n ------------------------ bsz == $bsz prompt_length == ${prompt_length}  sparse_policy == ${sparse_policy} phase == prefill \n"
            CUDA_VISIBLE_DEVICES=3 python -u 0_sparse_long_logs/1_quant_prefill_thr.py --batch_size $bsz --tp $tp \
                --sparse_policy ${sparse_policy} --prompt_length ${prompt_length} --model_path ${model_path}
            
            echo "\n ------------------------ bsz == $bsz prompt_length == ${prompt_length}  sparse_policy == ${sparse_policy} phase == decoding \n"
            CUDA_VISIBLE_DEVICES=3 python -u 0_sparse_long_logs/0_quant_decoding_thr.py --batch_size $bsz --tp $tp \
                --sparse_policy ${sparse_policy} --prompt_length ${prompt_length} --model_path ${model_path}
            echo "------------------------ end \n"
            sleep 2
            
        done 
    done

done 
