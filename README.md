## Introduction
> - BS168 is a genomic language and conditional diffusion framework developed for programmable synthetic promoter design in *Bacillus subtilis* strain 168. Built upon the GSR‑fine‑tuned DNABERT‑2 model, we integrated conditional diffusion generation, GSR scoring and strength tuning into a closed‑loop workflow, and used it to design promoters at target strengths in strain 168. The GSR‑DNABERT‑2 model, fine‑tuned on 9,964 experimentally measured promoters, estimates expression strength from input DNA sequences. A conditional diffusion model generates de‑novo 95‑bp candidate sequences, which are further refined by GSR‑guided step‑wise substitutions to approach desired strength bins.
> - We describe bs168-model in our paper[XXX](https://doi.org/XXXX)

## Important links:  
- [Paper:XXXX](https://arxiv.org/abs/XXXXXXX)
- [Huggingface: model weights](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- [Dataset: The BS168 model is trained in two stages: regression prediction fine‑tuning and conditional diffusion generation, based on 9,964 experimentally measured *Bacillus subtilis* promoter sequences ](https://github.com/EmperorUlasses/bs168-model-paper)

## Installation with conda
```bash
# create conda python environment
conda create -n BS168Env python=3.11 -y
conda activate BS168Env

# install dependencies from requirements.txt
pip install -r requirements.txt
```

## Applications

### pretiction step by step

#### step 1
Download model weights from huggingface
```bash
git clone https://huggingface.co/zhihan1996/DNABERT-2-117M Model
```

#### step 2‑1 Predict promoter strength from input DNA sequence
Predict promoter strength using local GSR‑DNABERT‑2 weights
```bash
python BS168.py --model_dir=./results_new/results_RNA --mode=predict --promoter_seq="XXXX" --out_dir='./output'
```

#### step 2‑2 Conditional generation of promoters with target strength
Generate de‑novo 95‑bp promoters for given target strength bin
```bash
python BS168.py --model_dir=./results_new/results_RNA --mode=generate --target_bin=3 --frame=0 --out_dir='./output'
```


# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{BS1682026,
  author  = {XXX,XXX,XXX},
  journal = {XXX},
  title   = {Deep learning-guided programmable design of Bacillus subtilis promoters with prescribed strength},
  year    = {2026},
  volume  = {XX},
  URL={https://doi.org/10.XXXX/XXXXX},       
	DOI={10.XXXX/XXXXX},      
	ISSN={XXXX-XXXX}
}
```
