# Sparse Convolutional Denoising Autoencoder
Ported from https://github.com/work-hard-play-harder/SCDA

---

In order to use with the dataset provided in the original repo, generate the MAFs file using `maf_from_one_hot.py`. For datasets from the 1K Genomes Project, use `maf_from_vcf.py` to generate the MAFs file and `extract_vcf_sublist.py` to convert the vcf to the format expected by the model (one line per sample haplotype and classes starting from 1).

Run `python SCDA -h` to see more options for hyperparameters and model configuration.
