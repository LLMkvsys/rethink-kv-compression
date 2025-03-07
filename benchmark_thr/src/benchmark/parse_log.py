import os, sys 

from prettytable import PrettyTable

model = "meta-llama-Llama-2-7b-hf"
for phase in ['prefill', 'decoding']: 
    print("\n\n", phase)
    tab = PrettyTable(["bsz", "bits",  "prtlen" , "policy", "thr"])
    for bsz in [1, 2, 4, 8, 16, 32]:
        for prompt_length in [256, 512, 1024, 2048, 3072, 4000]: # , 4096]: 
            time_infos = list() 
            # Q_Policy = "KIVI"
            # Q_Policy = "HEAD"
            # for (quant, policy) in [(0, "None"), (8, "KIVI"), (4, "KIVI")]: 
            # for (quant, policy) in [(0, "None"), (8, Q_Policy), (4, Q_Policy)]: 
            for (quant, policy) in [(0, "None")]: 
                path = os.path.join("logs", f"{phase}_model_{model}_batch_size_{bsz}_tp_1_quant_bits_{quant}_quant_policy_{policy}_prompt_length_{prompt_length}.log")
                if not os.path.exists(path): 
                    continue 
                with open(path, 'r') as f: 
                    lines = f.readlines() 
                time_cost = float(lines[-1].split("elapsed time is ")[-1][:-2])
                if quant == 0: 
                    quant = 16
                if phase in ['prompt']: 
                    thr = int(prompt_length / time_cost)
                else: 
                    # thr = int(32 / time_cost)
                    thr = time_cost
                tab.add_row([bsz, quant, prompt_length, policy, thr])
                
    
    print(tab)

# exit(0)


# from prettytable import PrettyTable
# tab = PrettyTable(["bsz", "bits", "policy", "time"])

# for bsz in [1, 2, 4, 8]: 
#     for quant in [4, 8]: 
#         time_infos = list() 
#         for policy in ["None", "HEAD", "KIVI", "H2O"]: 
#             path = os.path.join("0_logs", f"bsz_{bsz}_tp_1_quant_{quant}_quant_policy_{policy}.log")
#             with open(path, 'r') as f: 
#                 lines = f.readlines() 
#             time_cost = float(lines[-1].split("elapsed time is ")[-1][:-2])
#             time_infos.append(time_cost)
        
#         tab.add_row([bsz, quant, "None", time_infos[0]])
#         tab.add_row([bsz, quant, "KIVI", min(time_infos[1:2])])
#         tab.add_row([bsz, quant, "H2O", (time_infos[3])])
# print(tab)