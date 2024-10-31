
model_ident="Mistral-7B-v0.1_8192" # "Meta-Llama-3-8B-instruct_8192"

# model_name="DorinSht/ShareGPT_llama2_68M"
model_name="allenai/longformer-base-4096" # google-t5/t5-base


for alg in gear hf streamingllm h2o
do 
    for lr in 1e-5 
    do 
         python -u 0_src/bert/length/finetune_bert_length.py \
                --model_name=${model_name} \
                --data-path 0_data/leninfo/model_${model_ident}_alg_${alg}.csv \
                --train_batch_size 16 --learning_rate $lr --ga_steps 2 --max_len 2048 \
                --epoch 100 --alg=${alg} --llm=${model_ident} --save_check_pt --fixed_dropout=False &
    done 
done 

for alg in kivi
do 
    for lr in 1e-4
    do 
         python -u 0_src/bert/length/finetune_bert_length.py \
                --model_name=${model_name} \
                --data-path 0_data/leninfo/model_${model_ident}_alg_${alg}.csv \
                --train_batch_size 16 --learning_rate $lr --ga_steps 2 --max_len 2048 \
                --epoch 100 --alg=${alg} --llm=${model_ident} --save_check_pt --fixed_dropout=False &
    done 
done 
wait 