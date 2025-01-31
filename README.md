
<div align="center">

# ESMAdam: a plug-and-play all-purpose protein ensemble generator

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a><br>
[![bioarXiv](https://www.biorxiv.org/sites/default/files/biorxiv_article.jpg)](https://www.biorxiv.org/content/10.1101/2025.01.19.633818v1)

</div>




This repository contains the official implementation of the paper 
"ESMAdam: a plug-and-play all-purpose protein ensemble generator." 
ESMAdam serves as a versatile and efficient framework for general-purpose protein conformation ensemble generation. 
Using the ESMFold protein language model ESMFold and ADAM stochastic optimization in the continuous protein embedding space, 
ESMAdam addresses a wide range of ensemble generation tasks. 

<p align="center">
<img src="assets/frame_plot_2-1.png" width="450"/>
</p>

The model supports almost any protein ensemble generation task. In this repository we demonstrate a few shown in the paper, including
ensemble generation with experiment constraint, CG-to-all-atom backmapping, protein-complex alternative binding mode discovery,
and heterogeneous 3D structure reconstruction from cryo-EM images. Users are encouraged to replace the target function with your own!

## Installation
We recommend following ESM official github link for the installation instruction: https://github.com/facebookresearch/esm. Otherwises,
use the environment provided in this repository. (ESM installation is a bit tricky due to its use of openfold so I highly recommend building
dependency from there)
```sh
git clone https://github.com/Harrydirk41/ESMAdam.git
cd ESMAdam

# Create conda environment.
conda env create -f ESMAdam.yml
conda activate ESMAdam

# Support import as a package.
pip install -e .
```








## Citation

```
@article{yu2025esmadam,
  title={ESMAdam: a plug-and-play all-purpose protein ensemble generator},
  author={Yu, Zongxin and Liu, Yikai and Lin, Guang and Jiang, Wen and Chen, Ming},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}

```

