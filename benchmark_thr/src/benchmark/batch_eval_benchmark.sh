# export CUDA_VISIBLE_DEVICES="1,2,3"
# for bsz in 16 # 1 2 4 # 4 8 16 
bsz=1
quant=0
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES="2"
# export HF_HOME="/opt/huggingface/"

# ' \n\n are several news stories about the Occupy movement, specifically about the Occupy LA movement anniversary celebration. The first story is about the'

tp=1
quant_bits=4
for quant_bits in 0
do 
    for bsz in 1 # 2 4 8 # 16 # 32 64 
    do 
        for quant_policy in None # H2O # None HEAD KIVI  # 0 # 4 # 0 8 4 # 1 2 4 # 1 2 4 # 1 
        do  
            # pip install -e . 
            echo "--------------------------------------------------------\t bsz == $bsz \t tp == $tp  \t quant == $quant \n"
            CUDA_VISIBLE_DEVICES=2 python -u benchmark/quant_demo_benchmark.py --batch_size $bsz --tp $tp --quant_bits $quant_bits --quant_policy $quant_policy # > 0_logs/bsz_${bsz}_tp_${tp}_quant_${quant_bits}_quant_policy_${quant_policy}.log
            echo "-------------------------------------------------------- end \n"
            sleep 5
        done 
        # exit 
    done
done 

# tp==2
# num_kv_heads is  4
# num_q_heads is 16

# tp==1
# num_kv_heads is  8
# num_q_heads is  32


# ' \n\nProtests and demonstrations continued across the United States, with Occupy Wall Street movements in various cities, including Los Angeles, Philadelphia'
# ' \n\nMikhail Kalashnikov, the designer of the AK-47 rifle, has written a letter to Russian Orthodox Church'
# ' \n\nOn November 1, a deadly rampage occurred at Los Angeles International Airport (LAX) when a gunman, identified as'
# " \n\nThe European Union's High Representative for Foreign Affairs, Catherine Ashton, has visited Egypt to facilitate a political solution to the country"
# ' \n\nThe world is on high alert as the situation in Syria continues to escalate. The United States has deployed two destroyers in'
# ' \nKleiner Perkins Caufield & Byers, a Silicon Valley venture firm, has filed legal paperwork seeking $972'
# ' \n\nA man, John Tessier, also known as John McCullough, was convicted of murdering a 7-year-old'
# ' \n\nRush Limbaugh, a popular American radio host, has been facing backlash after making controversial comments about birth control and women'