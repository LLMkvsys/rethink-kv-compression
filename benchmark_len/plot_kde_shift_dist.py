import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy import stats

# Function to read and process the JSONL file
def read_jsonl_and_extract_eos_positions(file_path):
    print(file_path)
    eos_positions = []
    
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            data = json.loads(line)
            eos_positions.append(data["EOS pos"])
    return eos_positions



# Function to plot the difference proportion histogram
def plot_histogram_diff_prop(eos_positions_diff_props, saving_pths, method_idents, take_log=False):
    plt.figure(figsize=(10, 6))
    
    # sns.histplot(eos_positions_diff_prop, bins=20, kde=True, log=take_log, label='Effective output length difference', stat="density", common_norm=False)
    for (eos_positions_diff_prop, save_path, method_ident) in zip(eos_positions_diff_props, saving_pths, method_idents):
        sns.histplot(eos_positions_diff_prop, bins=20, kde=True, log=take_log, label=method_ident, stat="density", common_norm=False) # , kde_kws={"kwargs": {"linewidth":10}})
        
        # break
    # plt.xlim(-200, 100)
    plt.xticks(fontsize=29)
    plt.yticks(fontsize=29)
    


    plt.xlabel('Response Length Difference (%)', fontsize=29)
    if take_log:
        plt.ylabel('Log Density', fontsize=25+4)
    else:
        plt.ylabel('Density', fontsize=25+4)
    
    # plt.legend()
    plt.legend(fontsize=25+4, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"0_kde_{saving_pths[0][0][:10]}.pdf", dpi=300)
    print(f"0_kde_{saving_pths[0][0][:10]}.pdf")


# Main function
def main():
    # eval_ident = 'mistral'
    eval_ident = 'llama'
    baseline_path = 'hf_Meta-Llama-3-8B-instruct_8192_b1_requests1000_gen1024'
    if eval_ident == 'llama': 
        model_ident = 'Meta-Llama-3-8B-instruct_8192'
    
    compress_path_list = [
        [
            (f'kivi_{model_ident}_b1_2bits_group32_residual128_requests1000_gen1024', "KIVI-2"), 
            (f'kivi_{model_ident}_b1_4bits_group32_residual128_requests1000_gen1024', "KIVI-4"), 
        ], 
        
        [
            (f'gear_{model_ident}_b1_4bits_requests1000_gen1024',  "GEAR-2"),
            (f'gear_{model_ident}_b1_8bits_requests1000_gen1024',  "GEAR-4"),
        ], 
        [
            (f"h2o_{model_ident}_b1_c256_w64_k7_maxpool", "H2O-256"), 
            (f"h2o_{model_ident}_b1_c512_w64_k7_maxpool", "H2O-512"), 
        ],
        
        [
            (f"streamingllm_{model_ident}_b1_c256_w64_k7_maxpool", "Stream-256"), 
            (f"streamingllm_{model_ident}_b1_c1024_w64_k7_maxpool", "Stream-512"),  
        ],
    ]
    
    for hyp_compress_paths in compress_path_list:
        print("\n" + "-" * 20 )
        eos_diffs_props = list() 
        method_idents = list() 
        for compress_path, method_ident in hyp_compress_paths: 
            if not os.path.exists(f'{compress_path}/sharegpt.jsonl'):
                print(f"File not found: {compress_path}/sharegpt.jsonl, skipping...")
                continue
            
            # if not ('streaming' in compress_path or 'h2o' in compress_path): continue
            file_paths = [
                f'{baseline_path}/sharegpt.jsonl', # FP
                f'{compress_path}/sharegpt.jsonl' # kv-compressed
            ]

            eos_positions_list = []
            # NOTE: plot the EOS distributions
            for file_path in file_paths:
                dirname = os.path.dirname(file_path)
                target_dir = os.path.join('../0_cached_length_dst/', dirname)
                if not os.path.exists(target_dir): 
                    os.makedirs(target_dir)
                import shutil 
                shutil.copy(file_path, os.path.join(target_dir, 'sharegpt.jsonl'))
                
                eos_positions = read_jsonl_and_extract_eos_positions(file_path)
                if len(eos_positions) == 0: continue 
                eos_positions_list.append(eos_positions)
            
            # NOTE: plot the EOS difference proportion distribution
            outlier_cnt = 0
            eos_diffs_prop = []
            for idx in range(len(eos_positions_list[0])):
                hf_eos = eos_positions_list[0][idx]
                compress_eos = eos_positions_list[1][idx]

                # avoid zero division
                if hf_eos == 0:
                    continue
                
                diff_prop = round(((hf_eos - compress_eos) / hf_eos) * 100)
                
                outlier_threshold = 200
                if abs(diff_prop) > outlier_threshold:
                    # print(f"outlier values: {hf_eos} {compress_eos}")
                    outlier_cnt += 1
                    continue
                else:
                    eos_diffs_prop.append(diff_prop)
            eos_diffs_props.append(eos_diffs_prop)
            method_idents.append(method_ident)
        plot_histogram_diff_prop(eos_diffs_props, hyp_compress_paths, method_idents, take_log=True)
    
    

if __name__ == "__main__":
    main()
