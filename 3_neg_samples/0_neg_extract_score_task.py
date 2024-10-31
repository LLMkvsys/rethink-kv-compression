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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

dataset2task = {
    # "Multi-doc QA": ["hotpotqa", "2wikimqa"], 
    # "Single-doc QA": ["multifieldqa_en", "qasper"], 
    "QA": ["hotpotqa", "2wikimqa"] + ["multifieldqa_en", "qasper"], 
    "Summarization": ["gov_report", "qmsum", "multi_news"], 
    "Few shot": ["triviaqa",  "samsum", "trec"], 
    "Synthetic": ["passage_retrieval_en", "passage_count", "passage_count" ], 
    "Code": ["lcc",  "repobench-p"], 
}

def rename_dataset_to_task(dataset): 
    for task, value in dataset2task.items(): 
        if dataset in value: return task 
    return None 


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
    BaselineTHR = np.mean([baseline_entry['score'] for baseline_entry in baseline_data])
    # print(BaselineTHR)
    # Loop through each entry in the baseline data
    for index, baseline_entry in enumerate(baseline_data):
        baseline_id = baseline_entry['id']
        baseline_score = baseline_entry['score']
        if baseline_score < BaselineTHR: continue 
        # print("intersection == ", intersection)
        if intersection: 
            # Assume that baseline score should be higher than all other scores by `min_difference`
            is_failure = False
            is_failure_times = 0 
            # Loop through all other algorithm data
            other_entry_scores = []
            # import pdb; pdb.set_trace() 
            for idx, other_data in enumerate(other_data_list):
                assert len(other_data) == len(baseline_data)
                if index < len(other_data):
                    other_entry = other_data[index]
                    # if (baseline_score - other_entry['score']) < min_difference and baseline_score > 0.1:
                    if (baseline_score * (1-min_difference) < other_entry['score']):
                        is_failure_times += 1
                    #     break
                    # else: 
                    #     pass 
                        # if index < 10: 
                        #     print("index == ", index, "other_entry == ", other_entry['score'], baseline_score, min_difference)
            if is_failure_times <= 0: 
                is_failure = True 
                # other_entry_scores.append(other_data[index]['score'])
        else: 
            is_failure = False 
            other_entry_scores = []
            for other_data in other_data_list:
                if index < len(other_data):
                    other_entry = other_data[index]
                    # print(f"baseline_score: {baseline_score}, other_entry['score']: {other_entry['score']}")
                    # import pdb; pdb.set_trace() 
                    # print(baseline_score)
                    if (baseline_score * (1-min_difference) < other_entry['score']):
                        is_failure = True
                        break
        
        # If the baseline score is higher by the minimum difference for all other algorithms
        if is_failure:
            failure_ids.append(baseline_id)

    return failure_ids


def rename_dataset_to_task(dataset): 
    for task, value in dataset2task.items(): 
        if dataset in value: return task 
    return None 


dataset2task = {
    # "Multi-doc QA": ["hotpotqa", "2wikimqa"], 
    # "Single-doc QA": ["multifieldqa_en", "qasper"], 
    "QA": ["hotpotqa", "2wikimqa"] + ["multifieldqa_en", "qasper"], 
    "Summarization": ["gov_report", "qmsum", "multi_news"], 
    # "Few shot": ["triviaqa",  "samsum", "trec"], 
    # "Synthetic": ["passage_retrieval_en", "passage_count", "passage_count" ], 
    "Code": ["lcc",  "repobench-p"], 
}

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
    
    
    
    dataset_counts = dict() 
    base_score_info = dict() 
    other_score_infos = [dict() for _ in range(len(other_algs))]
    # Now, compare each baseline with the corresponding algorithm files
    for dataset_name, files in datasets.items():
        # print(dataset_name)
        baseline_file = files['baseline']
        if baseline_file:
            baseline_data = load_json(os.path.join(folder_path, baseline_file))
            BaselineTHR = np.mean([baseline_entry['score'] for baseline_entry in baseline_data])
            dataset_counts[dataset_name] = len([baseline_entry['score'] for baseline_entry in baseline_data if baseline_entry['score'] >= BaselineTHR])
            other_data_list = [load_json(os.path.join(folder_path, other_file)) for other_file in files['others']]
            # failure_ids = compare_json_files(baseline_data, other_data_list)
            failure_ids = compare_json_files_by_difference(baseline_data, other_data_list, min_difference=threshold, intersection=True)
            if len(failure_ids) > 0: 
                base_score_list = [baseline_data[failure_id]['score'] for failure_id in failure_ids]
                base_score_info[dataset_name] = np.mean(base_score_list)
                
                for i in range(len(other_algs)): 
                    other_score_list = [other_data_list[i][failure_id]['score'] for failure_id in failure_ids]
                    other_score_infos[i][dataset_name] = np.mean(other_score_list)
                # print(dataset_name)
                # print(base_score_list)
                # print(other_score_list)
                # import pdb; pdb.set_trace() 
                # print(f"base score is {np.mean(base_score_list)}, other score is {np.mean(other_score_list)}, dataset_name {dataset_name}")
            
            if failure_ids:
                failure_ids_dict[dataset_name] = failure_ids
            else:
                failure_ids_dict[dataset_name] = []
                # print(f"No failure cases found for dataset {dataset_name}: {len(baseline_data)} v.s. {len(other_data_list)}.")
        else:
            print(f"No baseline file found for dataset {dataset_name}.")

    output_ident = 'task' # #  'all'
    # exit(0)
    if output_ident == 'all': 
        print(f"base score is {np.mean(list(base_score_info.values()))}, other score is {np.mean(list(other_score_infos[i].values()))}")
    elif output_ident == 'task': 
        for task_name in dataset2task.keys(): 
            print(task_name)
            for i, other_alg in enumerate(other_algs): 
                print(other_alg)
                all_task_datasets = dataset2task[task_name]
                # import pdb; pdb.set_trace() 
                base_scores = [base_score_info[dainfo] for dainfo in all_task_datasets if dainfo in base_score_info]
                other_scores = [other_score_infos[i][dainfo] for dainfo in all_task_datasets if dainfo in base_score_info]
                # print(base_scores, other_scores)
                if len(base_scores) == 0: 
                    print(f"\t base score is 0, other score is 0")
                else: 
                    print(f"\t base score is {np.mean(base_scores)}, other score is {np.mean(other_scores)}")
        
        
    failure_infos = dict() 
    for dataset_name in failure_ids_dict.keys(): 
        task_name = rename_dataset_to_task(dataset_name)
        if task_name not in failure_infos: failure_infos[task_name] = list() 
        failure_infos[task_name].append((len(failure_ids_dict[dataset_name]), dataset_counts[dataset_name]))
    
    def compute_ratio(fail_ratio): 
        return sum([item[0] for item in fail_ratio]) / sum([item[1] for item in fail_ratio])

    task_fail_ratios = dict() 
    for task_name, fail_ratio in failure_infos.items(): 
        task_fail_ratios[task_name] = compute_ratio(fail_ratio)
    return task_fail_ratios

    if False: 
        results = {}
        for dataset_name in failure_ids_dict.keys(): 
            data = load_dataset('THUDM/LongBench', f"{dataset_name}", split='test')
            failure_info = failure_ids_dict[dataset_name] # this is index, [19, 49, 69, 106, 165, 166, 178, 199]
            # save to josn 
            failures = [data[i] for i in failure_info]
            results[dataset_name] = failures
    
    
    
    

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

score_path = "../0-failure_case_id_scores"
all_datasets = ["2wikimqa", "gov_report",  "hotpotqa", "lcc", "multi_news", "multifieldqa_en",
            "qmsum", "passage_count", "passage_retrieval_en", "qasper", "repobench-p", "samsum", "passage_count", 
            "trec", "triviaqa"]


# h2o_alg_idents = ('h2o', '8192_b1_c1024_w992_k7_maxpool')
# streamingllm_alg_idents = ('streamingllm', '8192_b1_c1024_w16_k7_maxpool')
# kivi_alg_idents = ('kivi', '8192_b1_4bits_group32_residual128')
# gear_alg_idents = ('gear', '8192_b1_8bits')




h2o_alg_idents = ('h2o', '8192_b1_c2048_w32_k7_maxpool')
streamingllm_alg_idents = ('streamingllm', '8192_b1_c2048_w32_k7_maxpool')
kivi_alg_idents = ('kivi', '8192_b1_4bits_group32_residual128')
gear_alg_idents = ('gear', '8192_b1_8bits')

info_by_method = dict() 
# threshold_list = [1, 2, 4, 8, 16, 32]
threshold_list = [10]
# for other_algs in [[h2o_alg_idents, streamingllm_alg_idents, kivi_alg_idents, gear_alg_idents]]: 
# h2o_alg_idents, streamingllm_alg_idents,  
# for other_algs in [[h2o_alg_idents, streamingllm_alg_idents], [kivi_alg_idents, gear_alg_idents]]: 
# for other_algs in [[kivi_alg_idents, gear_alg_idents]]: # , [streamingllm_alg_idents, h2o_alg_idents]]: # , [gear_alg_idents], [h2o_alg_idents], [streamingllm_alg_idents]]: 
for other_algs in [[streamingllm_alg_idents, h2o_alg_idents]]:
    print("-" * 20)
    x_list, y_list = list(), list() 
    for threshold in threshold_list: 
        print(other_algs)
        failure_ids_dict = find_common_failures(score_path, model_name='Meta-Llama-3-8B-instruct', 
                                                base_alg=('hf', '8192_b1'), 
                                                other_algs=other_algs, 
                                                all_datasets=all_datasets,
                                                threshold=threshold/100)
        
        
        
        
