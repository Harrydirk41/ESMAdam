
<div align="center">

# ESMAdam: a plug-and-play all-purpose protein ensemble generator

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
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

The current model supports guiding the sampling process with end-to-end distance, radius of gyration, helix percent per residue.
Additional, we provide preliminary models guiding with cryo-EM 2D density images:
- End-to-end distance
- Radius of gyration
- Helix percent per residue with distance operator
- RMSD w.r.t the folded structure
- Cryo-EM 2D density images (preliminary)
- Helix percent per residue with RMSD operator (in progress)
- Beta percent per residue with RMSD operator (in progress)








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

