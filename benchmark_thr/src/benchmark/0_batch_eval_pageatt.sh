# export CUDA_VISIBLE_DEVICES="1,2,3"
# for bsz in 16 # 1 2 4 # 4 8 16 
bsz=1
quant=0

# pip install -e . 

tp=1
# bsz=4
# quant_bits=4

# lmsys/longchat-7b-v1.5-32k
# 
model_path=meta-llama/Llama-2-7b-hf

for bsz in 1 2 4 8 16 32
do 
    for quant_bits in 0
    do 
        for prompt_length in 256 512 1024 2048 3072 4000
        do
            for quant_policy in None
            do
                echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == prefill \n"
                CUDA_VISIBLE_DEVICES=0 python -u benchmark/1_quant_prefill_thr.py --batch_size $bsz --tp $tp \
                    --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}

                echo "\n ------------------------ bsz == $bsz quant == $quant_bits  prompt_length == ${prompt_length}  quant_policy == ${quant_policy} phase == decoding \n"
                CUDA_VISIBLE_DEVICES=0 python -u benchmark/0_quant_decoding_thr.py --batch_size $bsz --tp $tp \
                    --quant_bits $quant_bits --quant_policy $quant_policy --prompt_length ${prompt_length} --model_path ${model_path}
                echo "------------------------ end \n"
                sleep 2
            done 
            # exit 
        done
    done 
done 
