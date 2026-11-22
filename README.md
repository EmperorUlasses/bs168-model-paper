## Introduction
> - BS168 is a genomic language and conditional diffusion framework developed for programmable synthetic promoter design in *Bacillus subtilis* strain 168. Built on DNABERT-2 pre-trained on gene start regions (GSR) and a conditional diffusion architecture, our closed-loop workflow integrates strength prediction, in silico dissection of sequence features, and conditional sequence generation. We combined a conditional diffusion model trained on 9,964 measured promoters with GSR scoring to generate novel 95-bp promoters toward target strengths. Generated sequences are distinct from the training library (median 16 bp difference), span the full strength range, and include candidates predicted to exceed the strongest library promoter. Validated in *B. subtilis* 168, these designed promoters enable high surfactin production in kitchen waste fermentation. BS168 reduces experimental workload for custom-strength promoter engineering and serves as a core computational tool for scalable metabolic engineering in *B. subtilis*.
> - We describe bs168-model in our paper[XXX](https://doi.org/XXXX)

## Important links:  
- [Paper:XXXX](https://arxiv.org/abs/XXXXXXX)
- [Huggingface: model weights](https://huggingface.co/lileica/GEO-SeqGuider)
- [Dataset: The GEO-SeqGuider model is trained in two stages: genome pre training and specific task fine-tuning, based on the dataset ]( https://doi.org/XXXX/zenodo.XXX)

## Installation with conda
```bash
    conda env create -f requirements.yml
    conda activate GEOSeqGuiderEnv
```

## Applications

### pretiction step by step

#### step 1
Download model weights from huggingface
```bash
git clone https://huggingface.co/XXX/XXX Model
```

#### step 2
Predict with local model weights
```bash
    python GEOSeqGuider.py --model=localModel/foldx --promoter="XXXX" --sgRNA='XXXX' --pam='XXXX' --out_dir='XXXX'
```


# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{XXX,
  author  = {XXX,XXX,XXX},
  journal = {CNS},
  title   = {XXXXXXXXXXX},
  year    = {2026},
  volume  = {14},
  URL={https://www.XXXX},       
	DOI={XXX.2026.XXX},      
	ISSN={XXXX-XXXX}
}
```
