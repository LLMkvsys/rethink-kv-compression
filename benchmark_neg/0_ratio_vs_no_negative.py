#!/usr/bin/env python
# coding: utf-8
import os
import re
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import jsonlines
import sys

import matplotlib.pyplot as plt 
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


color_list = ['tab:orange',
            'tab:blue',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']

hatch_list = [
    '', 
    '/', 
    '\\'
    '///', 
    '--', 
    '+', 
    'x'
    '*', 
    'o', 
    'O', 
    '.'
]

line_style_list = [
    '-', 
    '--', 
    '-.', 

]

marker_list = [
    '',
    'o', 
    'v',
    '^', 
    'X', 
    'D'
    's', 
]

template = {
    'fontsize': 18, 
    'linewidth': 6, 
    'scatter_markersize': 400, 
    'line_markersize': 20, 
    'width': 0.3, 
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
    
def autolabel_percent(rects, ax, value_list, error_list=None, str_func=None):
    if str_func is None: 
        str_func = lambda x: '%.2f'%(x)

    if error_list is None: 
        error_list = [0 for _ in value_list]

    for idx, rect in enumerate(rects):
        if value_list[idx] is None: continue
        height = rect.get_height()
        ax.annotate(str_func(value_list[idx]),
                    xy=(rect.get_x() + rect.get_width() / 2, height+error_list[idx]),
                    xytext=(0, 3),  # 3 points vertical offset
#                     color='white',
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='light')


def check_before_run(**kwargs): 
    if   kwargs['full'] + kwargs['half'] + kwargs['forth'] > 1: 
        return False 
    return True 


def apply_grid(ax, **kwargs): 
    if kwargs.get('grid'): 
        if not (kwargs.get('ygrid') or kwargs.get('xgrid')): 
            ax.grid(linestyle='-.', linewidth=1, alpha=0.5)

    if kwargs.get('ygrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='y')
    if kwargs.get('xgrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='x')


def apply_spine(ax, **kwargs): 
    if kwargs.get('spines'): 
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')


def apply_font(kwargs): 
    font = {'family' : 'serif',
            'size'   : 18}
    if kwargs.get('font'): 
        font.update(kwargs.get('font'))
    matplotlib.rc('font', **font)


def apply_log(ax, **kwargs): 
    if kwargs.get('logx'): 
        ax.set_xscale('log', base=kwargs.get('logx'))
    if kwargs.get('logy') > 0: 
        ax.set_yscale('log', base=kwargs.get('logy'))

def init_plot(ncols, **kwargs): 
    apply_font(kwargs)
    fig, axes = matplotlib.pyplot.subplots(1, ncols)
    if ncols == 1: 
        axes = [axes]
    fig.set_size_inches(w=ncols* 4.2, h=3)

    for ax in axes: 
        apply_grid(ax, **kwargs)
        apply_spine(ax, **kwargs)
        apply_log(ax, **kwargs)

    return fig, axes 



def parse_filename(filename):
    # Updated regex pattern to handle 'multi_news' as a special case and handle baseline files
    pattern = r"(?P<algorithm_name>.+?)_(?P<model_name>.+?)_(?P<model_len>\d+?)_b(?P<batch_size>\d+?)_(?:(?P<alg_params>.+?)_)?(?P<dataset_name>|passage_retrieval_en|multifieldqa_en|gov_report|multi_news|[^_]+?)_eval.json"
    match = re.match(pattern, filename)
    
    if match:
        groups = match.groupdict()
        
        # Handle special case for baseline files (those starting with 'hf')
        if groups['algorithm_name'] == 'hf':
            groups['alg_params'] = None
            for key_ident in ['passage_retrieval_en', 'multifieldqa_en', 'passage_count', 'multi_news', 'gov_report']:
                if key_ident in filename:
                    groups['dataset_name'] = key_ident
                    
            # print(f"Matched filename '{filename}' with groups: {groups['dataset_name']}")
        return groups
    
    # If filename doesn't match, print it for debugging
    print(f"Filename '{filename}' does not match the expected pattern.")
    
    return None


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def compare_json_files(baseline_data, other_data_list, high_threshold=0.7, low_threshold=0.6):
    """
    Compare baseline data with other algorithm data based on high/low thresholds.

    Parameters:
    - baseline_data (list of dict): List of dictionaries with 'id' and 'score' from the baseline file.
    - other_data_list (list of list of dict): List of lists, where each inner list contains dictionaries with 'id' and 'score' from other algorithm files.
    - high_threshold (float): Minimum score value to consider as high.
    - low_threshold (float): Maximum score value to consider as low.

    Returns:
    - List of IDs where the baseline score is high and all other algorithm scores are low.
    """
    failure_ids = []

    # Loop through each entry in the baseline data
    for index, baseline_entry in enumerate(baseline_data):
        baseline_id = baseline_entry['id']
        baseline_score = baseline_entry['score']
        
        # Check if the baseline score is high
        if baseline_score >= high_threshold:
            all_low = True
            
            # Loop through all other algorithm data
            for other_data in other_data_list:
                if index < len(other_data):
                    other_entry = other_data[index]
                    if other_entry['score'] >= low_threshold:
                        all_low = False
                        break

            # If all other scores are low, this is a failure case
            if all_low:
                failure_ids.append(baseline_id)

    return failure_ids


def compare_json_files_by_difference(baseline_data, other_data_list, min_difference=0.1, intersection=True):
    """
    Compare baseline data with other algorithm data based on score differences.

    Parameters:
    - baseline_data (list of dict): List of dictionaries with 'id' and 'score' from the baseline file.
    - other_data_list (list of list of dict): List of lists, where each inner list contains dictionaries with 'id' and 'score' from other algorithm files.
    - min_difference (float): Minimum difference between baseline score and other algorithm scores to consider as a failure case.

    Returns:
    - List of IDs where the baseline score is higher than all other algorithm scores by at least `min_difference`.
    """
    failure_ids = []
    # import pdb; pdb.set_trace() 
    # BaselineTHR = 0.5
    # BaselineTHR = np.median([baseline_entry['score'] for baseline_entry in baseline_data])
    BaselineTHR = np.median([baseline_entry['score'] for baseline_entry in baseline_data])
    # Loop through each entry in the baseline data
    for index, baseline_entry in enumerate(baseline_data):
        baseline_id = baseline_entry['id']
        baseline_score = baseline_entry['score']
        # print("intersection == ", intersection)
        if intersection: 
            # Assume that baseline score should be higher than all other scores by `min_difference`
            is_failure = True
            # Loop through all other algorithm data
            other_entry_scores = []
            # import pdb; pdb.set_trace() 
            for idx, other_data in enumerate(other_data_list):
                assert len(other_data) == len(baseline_data)
                if index < len(other_data):
                    other_entry = other_data[index]
                    # if (baseline_score - other_entry['score']) < min_difference and baseline_score > 0.1:
                    if (baseline_score * (1-min_difference) < other_entry['score']) or baseline_score < BaselineTHR:
                        is_failure = False
                        break
                    else: 
                        pass 
                other_entry_scores.append(other_data[index]['score'])
        else: 
            is_failure = False 
            other_entry_scores = []
            for other_data in other_data_list:
                if index < len(other_data):
                    other_entry = other_data[index]
                    if (baseline_score * (1-min_difference) < other_entry['score']) or baseline_score < BaselineTHR:
                        is_failure = True
                        break
        
        # If the baseline score is higher by the minimum difference for all other algorithms
        if is_failure:
            failure_ids.append(baseline_id)

    return failure_ids


def find_common_failures(folder_path, model_name, base_alg, other_algs, all_datasets, threshold):
    datasets = {}
    failure_ids_dict = {}  # Dictionary to store failure IDs for each dataset

    # add baseline algorithms 
    for dataset_name in all_datasets: 
        filename = '_'.join([base_alg[0], model_name, base_alg[1], dataset_name, 'eval.json']) 
        if not os.path.exists(os.path.join(folder_path, filename)): 
            print(filename + '  does not exist.')
            continue 
        
        if filename.endswith(".json"):
            parameters = parse_filename('_'.join([base_alg[0], model_name, base_alg[1], dataset_name, 'eval.json']))
            if parameters:
                if dataset_name not in datasets:
                    datasets[dataset_name] = {'baseline': filename, 'others': []}
    
    # add other compression algorithms 
    for dataset_name in all_datasets: 
        for other_alg in other_algs: 
            filename = '_'.join([other_alg[0], model_name, other_alg[1], dataset_name, 'eval.json'])
            if not os.path.exists(os.path.join(folder_path, filename)): 
                print(filename + '  does not exist.')
                continue 
            datasets[dataset_name]['others'].append(filename)
                
    # Now, compare each baseline with the corresponding algorithm files
    for dataset_name, files in datasets.items():
        # print(dataset_name)
        baseline_file = files['baseline']
        if baseline_file:
            baseline_data = load_json(os.path.join(folder_path, baseline_file))
            other_data_list = [load_json(os.path.join(folder_path, other_file)) for other_file in files['others']]
            # failure_ids = compare_json_files(baseline_data, other_data_list)
            failure_ids = compare_json_files_by_difference(baseline_data, other_data_list, min_difference=threshold, intersection=True)

            if failure_ids:
                failure_ids_dict[dataset_name] = failure_ids
            else:
                failure_ids_dict[dataset_name] = []
                # print(f"No failure cases found for dataset {dataset_name}: {len(baseline_data)} v.s. {len(other_data_list)}.")
        else:
            print(f"No baseline file found for dataset {dataset_name}.")
    
    # print(failure_ids_dict)
    return failure_ids_dict


def inspect_false_predictions(dataset_name, failure_ids_dict):
    failure_ids = failure_ids_dict.get(dataset_name, [])
    print("*" * 50)
    
    root_dir = "./0-failure_case_candidates"
    algo_false_pred = {}
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        algo_false_pred[folder_name] = []

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.jsonl') and file_name.startswith(dataset_name):
                    file_path = os.path.join(folder_path, file_name)
                    # print(file_path)
                    with jsonlines.open(file_path) as reader:
                        for obj in reader:
                            failure_id = obj.get('id')
                            algo_false_pred[folder_name] 

                            if failure_id in failure_ids:
                                algo_false_pred[folder_name].append(obj)

    # Ensure the output directory exists
    output_dir = os.path.join("./0-failure_cases_datasets", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the algo_false_pred to a JSON file for each folder
    for folder_name, entries in algo_false_pred.items():
        output_file = os.path.join(output_dir, f"{folder_name}.json")
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)
        print(f"Saved compression predictions on {dataset_name} failure cases to {output_file}")
    print("*" * 50)


def save_failure_cases(dataset_name, failure_ids, data, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    failure_cases = [data[idx] for idx in failure_ids if idx < len(data)]
    
    # Define the path for the failure cases file
    failure_cases_file = os.path.join(output_dir, f"failure_cases_{dataset_name}.json")

    # Save the failure cases to a JSON file
    with open(failure_cases_file, 'w') as file:
        json.dump(failure_cases, file, indent=4)



def find_common_failures(folder_path, model_name, base_alg, other_algs, all_datasets, threshold):
    datasets = {}
    failure_ids_dict = {}  # Dictionary to store failure IDs for each dataset

    # add baseline algorithms 
    for dataset_name in all_datasets: 
        filename = '_'.join([base_alg[0], model_name, base_alg[1], dataset_name, 'eval.json']) 
        if not os.path.exists(os.path.join(folder_path, filename)): 
            print(filename + '  does not exist.')
            continue 
        
        if filename.endswith(".json"):
            parameters = parse_filename('_'.join([base_alg[0], model_name, base_alg[1], dataset_name, 'eval.json']))
            if parameters:
                if dataset_name not in datasets:
                    datasets[dataset_name] = {'baseline': filename, 'others': []}
    
    # add other compression algorithms 
    for dataset_name in all_datasets: 
        for other_alg in other_algs: 
            filename = '_'.join([other_alg[0], model_name, other_alg[1], dataset_name, 'eval.json'])
            if not os.path.exists(os.path.join(folder_path, filename)): 
                print(filename + '  does not exist.')
                continue 
            datasets[dataset_name]['others'].append(filename)
                
    # Now, compare each baseline with the corresponding algorithm files
    for dataset_name, files in datasets.items():
        # print(dataset_name)
        baseline_file = files['baseline']
        if baseline_file:
            baseline_data = load_json(os.path.join(folder_path, baseline_file))
            other_data_list = [load_json(os.path.join(folder_path, other_file)) for other_file in files['others']]
            # failure_ids = compare_json_files(baseline_data, other_data_list)
            failure_ids = compare_json_files_by_difference(baseline_data, other_data_list, min_difference=threshold, intersection=True)

            if failure_ids:
                failure_ids_dict[dataset_name] = failure_ids
            else:
                failure_ids_dict[dataset_name] = []
                # print(f"No failure cases found for dataset {dataset_name}: {len(baseline_data)} v.s. {len(other_data_list)}.")
        else:
            print(f"No baseline file found for dataset {dataset_name}.")
    
    # print(failure_ids_dict)
    return failure_ids_dict

def collect_failure_num(failure_dict, all_datasets): 
    return sum([len(failure_dict[dataset]) for dataset in all_datasets])
    


std_methods = {
    "h2o": "H2O", 
    "streamingllm": "StreamingLLM", 
    "kivi": "KIVI", 
    "gear": "GEAR", 
}

def method2std(method): 
    lower_method = method.lower()
    return std_methods[lower_method]


seed_everything(42)

score_path = "./0-failure_case_id_scores"
all_datasets = ["2wikimqa", "gov_report",  "hotpotqa", "lcc", "multi_news", "multifieldqa_en",
            "qmsum", "passage_count", "passage_retrieval_en", "qasper", "repobench-p", "samsum", "passage_count", 
            "trec", "triviaqa"]

h2o_alg_idents = ('h2o', '8192_b1_c2048_w32_k7_maxpool')
streamingllm_alg_idents = ('streamingllm', '8192_b1_c2048_w32_k7_maxpool')
kivi_alg_idents = ('kivi', '8192_b1_4bits_group32_residual128')
gear_alg_idents = ('gear', '8192_b1_8bits')


info_by_method = dict() 
threshold_list = [1, 2, 4, 8, 10, 16, 24, 32]
for other_algs in [h2o_alg_idents, streamingllm_alg_idents, kivi_alg_idents, gear_alg_idents]: 
    x_list, y_list = list(), list() 
    for threshold in threshold_list: 
        failure_ids_dict = find_common_failures(score_path, model_name='Meta-Llama-3-8B-instruct', 
                                                base_alg=('hf', '8192_b1'), 
                                                other_algs=[other_algs], 
                                                all_datasets=all_datasets,
                                                threshold=threshold/100)
        # print(failure_ids_dict)
        sample_number = collect_failure_num(failure_ids_dict, all_datasets)
        x_list.append(threshold)
        y_list.append(sample_number)
        info_by_method[method2std(other_algs[0])] = (x_list, y_list)
        print(f"thresold is {threshold}, sample num is {sample_number}")

    
    if other_algs == streamingllm_alg_idents or other_algs == gear_alg_idents: 
        x_list, y_list = list(), list() 
        for threshold in threshold_list: 
            failure_ids_dict = find_common_failures(score_path, model_name='Meta-Llama-3-8B-instruct', 
                                        base_alg=('hf', '8192_b1'), 
                                        other_algs=[h2o_alg_idents, streamingllm_alg_idents] if other_algs == streamingllm_alg_idents else [kivi_alg_idents, gear_alg_idents], 
                                        all_datasets=all_datasets,
                                        threshold=threshold/100)
            # print(failure_ids_dict)
            sample_number = collect_failure_num(failure_ids_dict, all_datasets)
            x_list.append(threshold)
            y_list.append(sample_number)
        
        if other_algs == streamingllm_alg_idents: 
            method_ident = 'Sparse (C)'
        elif other_algs == gear_alg_idents: 
            method_ident = 'Quant (C)'
        info_by_method[method_ident] = (x_list, y_list)
        print(f"thresold is {threshold}, sample num is {sample_number}")
           

template.update(
        {
            "norm": False, 
            "width": 0.6, 
            "autolabel": True, 
            'norm': True,
            'logy': 0,
            'logx': 2,
            'barh': False,
        }
    )
method2color = {
    "H2O": "tab:red", 
    "StreamingLLM": "tab:purple", 
    "KIVI": "tab:orange", 
    "GEAR": "tab:green", 
}
method2marker = {
    "H2O": "d", 
    "StreamingLLM": "*", 
    "KIVI": "v", 
    "GEAR": "^", 
}
fig, axes = init_plot(1, grid=True, logy=template.get('logy', 0), logx=template.get('logx', 0))

for method in ["KIVI", "GEAR"]: 
    info_data = info_by_method[method]
    axes[0].plot(info_data[0], info_data[1], color=method2color[method], marker=method2marker[method], alpha=1.0, 
                 label=method, linewidth=2, markersize=10)

info_data = info_by_method['Quant (C)']
# print(info_data)
axes[0].plot(info_data[0], info_data[1], linestyle='-.', color="tab:brown", marker='8', alpha=1.0, 
                 label='Quant (C)', linewidth=2, markersize=10)

# axes[0].legend(fontsize=template['fontsize'], loc='upper center', ncol=1, bbox_to_anchor=(0.65, 1.0), fancybox=True, shadow=False)
axes[0].legend(fontsize=template['fontsize']-2, loc='lower left', ncol=1, bbox_to_anchor=(0., 0.0), fancybox=True, shadow=False)

axes[0].set_ylabel('# Negative Samples', fontsize=template['fontsize'])
axes[0].set_xlabel('Thresold (%)', fontsize=template['fontsize'])

# axes[0].set_ylim([400, 800])

plt.savefig('./images/llama_ratio_vs_quant_method.pdf', bbox_inches='tight') 


fig, axes = init_plot(1, grid=True, logy=template.get('logy', 0), logx=template.get('logx', 0))

for method in ["H2O", "StreamingLLM"]: 
    info_data = info_by_method[method]
    axes[0].plot(info_data[0], info_data[1], color=method2color[method], marker=method2marker[method], alpha=1.0, 
                 label=method.replace("StreamingLLM", "Stream"), linewidth=2, markersize=10)


info_data = info_by_method['Sparse (C)']
axes[0].plot(info_data[0], info_data[1], linestyle='-.', color="tab:gray", marker='8', alpha=1.0, 
                 label='Sparse (C)', linewidth=2, markersize=10)

axes[0].set_ylabel('# Negative Samples', fontsize=template['fontsize'])
axes[0].set_xlabel('Thresold (%)', fontsize=template['fontsize'])
axes[0].legend(fontsize=template['fontsize']-2, loc='lower left', ncol=1, bbox_to_anchor=(0., 0.0), fancybox=True, shadow=False)


plt.savefig('./images/llama_ratio_vs_sparse_method.pdf', bbox_inches='tight') 