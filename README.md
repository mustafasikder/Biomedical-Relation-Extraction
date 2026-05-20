# Do Syntactic Features Help Biomedical Relation Extraction?

Code for the paper **"Do Syntactic Features Help Biomedical Relation Extraction? An Empirical Study of Verb Token and Dependency Graph Augmentation"**, accepted at BioNLP 2026 @ ACL.

## Overview

We evaluate two syntactic augmentation strategies on top of BiomedBERT for biomedical relation extraction:
- **Entity marker pooling** (baseline)
- **Verb token augmentation** — concatenates the ROOT verb hidden state to entity representations
- **Dependency GCN** — refines entity representations via graph convolution over the dependency parse

Experiments are conducted on ChemProt, DDI, and AIMed using three random seeds.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

We use the following datasets:
- [ChemProt](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/)
- [DDI Corpus](https://github.com/isegura/DDICorpus)
- [AIMed](http://mars.cs.utu.fi/PPICorpora/)

Place processed data under `data/chemprot/`, `data/ddi/`, and `data/aimed/`.

## Code

*Scripts will be uploaded shortly.*

## Citation

```bibtex
@inproceedings{sikder2026syntactic,
  title     = {Do Syntactic Features Help Biomedical Relation Extraction?
               An Empirical Study of Verb Token and Dependency Graph Augmentation},
  author    = {Sikder, Mustafa Kamal and Kwegyir-Afful, Ernest},
  booktitle = {Proceedings of the BioNLP Workshop at ACL 2026},
  year      = {2026}
}
```
