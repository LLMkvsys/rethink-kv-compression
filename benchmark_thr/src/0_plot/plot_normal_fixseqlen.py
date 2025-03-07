import os, sys
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


if not os.path.exists('./demo_figs'): 
    os.makedirs('demo_figs')
    

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
font = {'family': 'serif', 'size': 18}
plt.rc('font', **font)

method2color = {
    "None": "tab:blue",
    "GEAR": "tab:green",
    "KIVI": "tab:orange",
    "H2O": "tab:red",
    "StreamingLLM": "tab:purple",
}

method2marker = {
    "None": "o",
    "GEAR": "^",
    "KIVI": "v", 
    "H2O": "d",
    "StreamingLLM": "*",
}

bit2linestyle = {
    "16": "solid",
    "2": "solid",
    "4": "solid",
    "8": "solid",
}

# 1. vllm 
# prefill_thr: bsz * prompt_len  / np.mean(time_costs)
# decoding_thr: bsz / np.mean(time_costs)

# 2. trl
# prefill_thr: bsz * prompt_len / elapsed_time 
# decoding_thr: bsz * (64) / elpased_time 

def compute_detailed_thr(path, phase, prompt_length, bsz, print_info=False):
    # deal with non-existing files / empty logs
    if not os.path.exists(path):
        print(f"----- no log file ----- \n {path}")
        import pdb; pdb.set_trace() 
        return None

    if print_info:
        print(path)
    with open(path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        print(f'----- empty log file ----- \n {path}')
        return None

    # calculate thr using detailed logs
    thr = None
    if phase in ['prompt']: 
        thr = int(prompt_length / time_cost)
    else:
        time_cost = 0
        time_costs = []
        for line in lines[:-1]:
            if phase in line and f"bsz {bsz}" in line: # ensure phase, bsz matches
                start = line.index("took") + len("took ")
                end = line.index("ms")
                time_costs.append(float(line[start:end].strip()))

        # exclude min and max
        if len(time_costs) != 0:
            time_costs.sort()
            # if time_cost_list[-1] / time_cost_list[0] > 1.5:
            #     print(f"min {time_cost_list[0]}, max {time_cost_list[-1]}")
            if print_info:
                print(f"{phase} {time_costs}")
            # time_costs = time_costs[1:-1]
            # time_cost = np.mean(time_costs) / 1000

           # NOTE: if time_costs <= 2 elements, fail to run and only has 1 or 2 exceptionally large val
            if len(time_costs) <= 2:
                # time_cost = time_costs[0] / 1000
                return None
            else:
                time_costs = time_costs[1:-1]
                time_cost = np.mean(time_costs) / 1000

        # calculate thr
        try:
            if phase == 'prefill':
                thr = round((bsz * prompt_length) / time_cost,  2)
                if print_info:
                    print(f"prefill {bsz} * {prompt_length} / {time_cost} = {thr}")
            elif phase == 'decoding':
                thr = round((bsz * 1) / time_cost,  2)
                if print_info:
                    print(f"decode {bsz} / {time_cost} = {thr}")
        except:
            print(f'----- cannot find matched bsz data for {phase}:----- \n {path}')

    # print(f"vllm thr: {thr}")
    return thr


def compute_raw_thr(path, phase, prompt_length, bsz):
    with open(path, 'r') as f: 
        lines = f.readlines()

    thr = None
    time_cost = float(lines[-1].split("elapsed time is ")[-1][:-2])
    if phase in ['prompt']: 
        thr = int(prompt_length / time_cost)
    elif phase == 'prefill':
        thr = round(bsz * prompt_length / time_cost, 2)
    elif phase == 'decoding':
        thr = round(bsz * 64 / time_cost, 2)

    # print(f"trf thr: {thr}")
    return thr




# NOTE: add this for figure suffix
suffix = ""
model = "lmsys-longchat-7b-v1.5-32k"
prompt_length_list = [1024]
bsz_list = [1, 2, 4, 8, 16]


output_folder = "./demo_figs"


##### parse quant logs
log_folder = "./0_quant_normal_logs/"
quant_policy_list = [(0, "None"), (4, "KIVI"), (4, "GEAR")]

quant_normal_full_data = {
    "prefill": [],
    "decoding": [],
}

for phase in ['prefill', 'decoding']: 
    print("\n\n", phase)
    tab = PrettyTable(["bsz", "bits",  "prtlen" , "policy", "thr"])

    thr_dict = {
        "16_None": [],
        "4_KIVI": [],
        "4_GEAR": [],
    }

    phase_data = []
    for prompt_length in prompt_length_list:
        print(prompt_length)
        for bsz in bsz_list:
            time_infos = list()
            for (quant, policy) in quant_policy_list:
                # FIXME: path, collect together?
                path = os.path.join(log_folder, f"{phase}_model_{model}_batch_size_{bsz}_tp_1_quant_bits_{quant}_quant_policy_{policy}_prompt_length_{prompt_length}.log")

                if quant == 0: quant = 16
                thr = compute_detailed_thr(path, phase, prompt_length, bsz, print_info=False)

                tab.add_row([bsz, quant, prompt_length, policy, thr])
                # print([bsz, quant, prompt_length, policy, thr])
                thr_dict[f"{quant}_{policy}"].append(thr)
        phase_data.append((bsz_list, thr_dict))

        # clear thr_dict
        thr_dict = {
            "16_None": [],
            "2_KIVI": [],
            "4_KIVI": [],
            "8_KIVI": [],
            "2_GEAR": [],
            "4_GEAR": [],
            "8_GEAR": [],
        }

    quant_normal_full_data[phase] = phase_data
    # print(phase_data)
    # print(len(phase_data))
    assert len(phase_data) == len(prompt_length_list)
    phase_data.clear




##### parse sparse logs
log_folder = "./0_sparse_normal_logs/"
quant_policy_list = [(0, "H2O"), (0, "StreamingLLM")]

sparse_normal_full_data = {
    "prefill": [],
    "decoding": [],
}

for phase in ['prefill', 'decoding']: 
    print("\n\n", phase)
    tab = PrettyTable(["bsz", "bits",  "prtlen" , "policy", "thr"])

    thr_dict = {
        "16_None": [],
        "16_H2O": [],
        "16_StreamingLLM": []
    }

    phase_data = []
    for prompt_length in prompt_length_list:
        # print(prompt_length)
        for bsz in bsz_list:
            time_infos = list()
            for (quant, policy) in quant_policy_list:
                path = os.path.join(log_folder, f"{phase}_model_{model}_batch_size_{bsz}_tp_1_quant_bits_{quant}_quant_policy_{policy}_prompt_length_{prompt_length}.log")

                if quant == 0: quant = 16
                thr = compute_detailed_thr(path, phase, prompt_length, bsz, print_info=False)

                tab.add_row([bsz, quant, prompt_length, policy, thr])
                # print([bsz, quant, prompt_length, policy, thr])
                thr_dict[f"{quant}_{policy}"].append(thr)
        phase_data.append((bsz_list, thr_dict))

        # clear thr_dict
        thr_dict = {
            "16_None": [],
            "16_H2O": [],
            "16_StreamingLLM": []
        }
    sparse_normal_full_data[phase] = phase_data
    phase_data.clear
    assert len(phase_data) == len(prompt_length_list)
    # print(tab)





##### plot
for phase in ['prefill', 'decoding']:
    for idx, prompt_length in enumerate([1024]):

        # NOTE: plot quant normal
        # import pdb; pdb.set_trace() 
        x_data, y_data = quant_normal_full_data[phase][idx]
        plt.figure(figsize=(4, 3), dpi=150)
        # import pdb; pdb.set_trace() 
        for k, v in y_data.items():
            quant_name, policy_name = k.split('_')
            # print(f"{quant_name} {policy_name}")

            if policy_name != 'None' and quant_name != '4': continue # FIXME:

            label = f'{policy_name}-{quant_name}'
            if policy_name == 'None':
                label = 'FP16'
            elif policy_name == "KIVI":
                label = f'K-{quant_name}'
            elif policy_name == "GEAR":
                label = f'G-{quant_name}'

            plt.plot(x_data, v,
                     marker=method2marker[policy_name],
                     markersize=10,
                     linestyle=bit2linestyle[quant_name],
                     linewidth=2,
                     color=method2color[policy_name],
                     label=label
                    )
        
        # NOTE: plot sparse normal, the min prompt len of sparse settingas start from 1024?
        # if prompt_length >=1024:
        #     x_data, y_data = sparse_normal_full_data[phase][idx-2]
        x_data, y_data = sparse_normal_full_data[phase][idx]
        for k, v in y_data.items():
            quant_name, policy_name = k.split('_')
            # print(f"{quant_name} {policy_name}")
            label = policy_name
            if policy_name == 'None':
                label = 'FP-16'
                continue # avoid two FP16, quant and sparse
            elif policy_name == "StreamingLLM":
                label = 'Stream'
            plt.plot(x_data, v,
                    marker=method2marker[policy_name],
                    markersize=10,
                    linestyle=bit2linestyle[quant_name],
                    linewidth=2,
                    color=method2color[policy_name],
                    label=label
                    )

        plt.xlabel('Batch Size', fontsize=18)
        if phase == "prefill":
            plt.ylabel('P Thr (T/S)', fontsize=18)
        else:
            plt.ylabel('D Thr (T/S)', fontsize=18)
        # plt.xticks(ticks=list(range(len(bsz_list))), labels=[str(bsz) for bsz in bsz_list], fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.savefig(f"{output_folder}/normal_phase_{phase}_promptlen_{prompt_length}{suffix}.pdf", bbox_inches='tight')
        print(f"{output_folder}/normal_phase_{phase}_promptlen_{prompt_length}{suffix}.pdf", flush=True)
        # plt.show()