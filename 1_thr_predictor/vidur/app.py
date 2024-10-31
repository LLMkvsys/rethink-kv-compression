import collections
import glob
import math
import os
import pandas
import pandas as pd 

from scipy.interpolate import interp1d, LinearNDInterpolator



def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper


method2speed = {
    "Baseline": 1.0, 
    "H2O": 0.95, 
    "StreamingLLM": 0.89, 
    "KIVI": 0.92, 
    "GEAR": 0.93453, 
}

class Application(object):
    def __init__(self, speed_df):
        # model,method,bsz,prompt_len,stage,thr
        self.config2thr = speed_df 
        
    # @memoize
    def get_throughput(self, bsz, prompt_len, method, stage):
        # Normalize placement to the lexicographically smallest rotation.
        method_df = self.config2thr[self.config2thr.method == method]
        df = method_df[method_df.stage == stage]
        
        # import pdb; pdb.set_trace() 
        xs = ["prompt_len"]
        ys = ["thr"]
        
        if bsz in df.bsz.values:
            # print(type(prompt_len), type(stage))
            # if prompt_len == 721 and stage == 'prefill': 
            #     import pdb; pdb.set_trace()
            # print(prompt_len, stage)
            
            bsz_df = df[df.bsz == bsz]
            tmp_prompt_len = prompt_len
            prompt_len = min(max(min(bsz_df.prompt_len.values) + 1, prompt_len), max(bsz_df.prompt_len.values) - 1)
            interpolator = interp1d(bsz_df.prompt_len.values, bsz_df[ys].values, axis=0)
            ret = interpolator(prompt_len) # / prompt_len * tmp_prompt_len
        else: 
            import pdb; pdb.set_trace() 
        
        if sum(ret) != sum(ret): 
            import pdb; pdb.set_trace() 
        assert sum(ret) == sum(ret), "{} {}".format(prompt_len, bsz)
        return ret # * method2speed[]

if __name__ == '__main__': 
    speed_df = pd.read_csv("0_thr_dataset.csv")
    app = Application(speed_df=speed_df)
    result= app.get_throughput(1, 1024, 'prefill')
    print(result)
    # import pdb; pdb.set_trace() 