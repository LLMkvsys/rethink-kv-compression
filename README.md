# LLMkvsys Artifact for MLsys 25

In this repository, we provide the artifact for the paper **Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving** 🚀.
In this artifact, we provide throughput and length analysis for various KV cache compression algorithms. We have several key findings. (1) While compressing KV cache can reduce memory consumption, current implementations (e.g., FlashAttention, PagedAttention) do not optimize for production-level LLM serving, resulting in suboptimal throughput performance. (2) Compressing KV cache may lead to longer outputs, resulting in increased end-to-end latency.


Key Features:
- 🚀 LMdeploy for various KV cache compression algorithms.


## Step 1: Set-up

Clone the repository and prepare environment. 

```bash
  git clone https://github.com/LLMkvsys/rethink-kv-compression
  cd rethink-kv-compression/
  conda env create -f mlsys_environment.yml
  conda activate lmdeploy
  conda clean -a
  pip install --no-cache-dir -r mlsys_requirements.txt
  cd  benchmark_thr/src
  pip install -e . 
```
 We also provide ``Dockerfile`` to to help you build a docker image. 


## Step 2: Evaluation Process (Est. Time: 30 mins)
You shoud ``cd benchmark_thr/src/`` to start the following experiments. 

### Step 2.1: Run various KV cache compression algorithms with fixed batch size 
```bash
  bash 0_quant_normal_logs/batch_eval_quant_normal_fixbsz.sh
  bash 0_quant_long_logs/batch_eval_quant_long_fixbsz.sh
  bash 0_sparse_normal_logs/batch_eval_sparse_normal_fixbsz.sh
  bash 0_sparse_long_logs/batch_eval_sparse_long_fixbsz.sh
```


### Step 2.2: Run various KV cache compression algorithms with fixed sequence length
```bash
  bash 0_quant_normal_logs/batch_eval_quant_normal_fixlen.sh
  bash 0_quant_long_logs/batch_eval_quant_long_fixbsz.sh
  bash 0_sparse_normal_logs/batch_eval_sparse_normal_fixlen.sh
  bash 0_sparse_long_logs/batch_eval_sparse_long_fixbsz.sh
```

### Step 2.3: Plotting Figures 
```bash
  # Figure 1 (e) & (i)
  python 0_plot/plot_normal_fixbsz.py
  # Figure 1 (f) & (j)
  python 0_plot/plot_normal_fixseqlen.py 
  # Figure 1 (g) & (k)
  python 0_plot/plot_long_fixbsz.py 
  # Figure 1 (h) & (i)
  python 0_plot/plot_long_fixseqlen.py 
```



## Other experiments 
We prepare length analysis and negative analysis as well as our tools for your reference. This part demands significant GPU resources, thus we directly provide cached results in our prepared environment. 
###  Length Analysis 
```bash
  cd benchmark_len/
  python -u plot_kde_shift_dist.py # Figure 3 
```


###  Negative Sample Analysis 
```bash
  cd benchmark_neg/
  python -u 0_ratio_vs_no_negative.py # Figure 5 
```

### A Suite of Tools 
We provide throughput predictor, length predictor, and negative sample evaluator in ``1_thr_predictor``, ``2_length_predictor``, and ``3_neg_samples`` respectively. 
