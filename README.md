# The Signature of Counting
This project investigates and characterizes counting behavior in LLMs. File structure: 

├── data 
├── notebooks
├── plots
├── README.md
├── results
├── slurm_jobs
└── src

## Oveview
We perform experiments to characterize how LLMs  

## Steps to Reproduce
1. Set up environment 
    ```$ pip install .```

2. Generate synthetic data for experiments (random hashes).
    ```
        $ python src/generate_hashes.py
            --count 100
            --length 32
            --save_path data/hashes.c32.l100.jsonl
    ```
