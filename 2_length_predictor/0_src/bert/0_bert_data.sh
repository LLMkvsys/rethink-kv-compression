
model_name="allenai/longformer-base-4096" # google-t5/t5-base

for lr in 1e-3
do 
     python -u 0_src/bert/length/samples_regression_sharegpt.py
done