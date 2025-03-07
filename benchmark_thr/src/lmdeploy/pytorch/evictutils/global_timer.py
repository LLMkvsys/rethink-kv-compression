# record the starting timestamps, len: number of prompts
global st_times
# record the prefill completion timestamps, len: number of prompts
global prefill_times
# record the ending timestamps, len: number of prompts
global end_times
# distinguish the prefill completion (count==1) and decode starting timestamp
global forward_count

st_times = []
prefill_times = []
end_times = []
forward_count = 0