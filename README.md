# The Signature of Counting
This project investigates and characterizes counting behavior in LLMs. 

## Oveview
LLMs are known to have deficits in their string matching and counting abilities. This repo studies their failure modes systematically under different settings, using strings which have minimal presence within the training corpora (random hashes). 

## Hypothesis Testing
To study the nature of counting errors that arise in LLMs, we design our experimental framework to test for statistically significant differences in model counting errors. Each setting tests a different dimension of counting, and we would expect the model to produce errors that are fairly close to each other.

- Counting within the input space (how many) vs output space (Give-N)
- Counting homogeneous vs non-homeogenous lists
- Counting the target string within natural language contexts vs counting within random token contexts: requires access to model's vocabulary

## Project structure: 
```text
    ├── data 
    ├── notebooks
    ├── plots
    ├── README.md
    ├── results
    ├── slurm_jobs
    └── src
```

## Steps to Reproduce Experiments
1. Set up environment 
    ```
        $ pip install .
    ```

2. Generate synthetic data for experiments (random hashes).
    ```
        $ python src/generate_hashes.py
            --count 100
            --length 32
            --save_path data/hashes.c32.l100.jsonl
    ```

## Results / Examples 

## Citation
```bibtex
@article{yourpaper2026,
  title={...}
}