# multiLID - Unfolding local growth rate estimates for (almost) perfect adversarial detection

This repository provides the official Python implementation of  "Unfolding local growth rate estimates for (almost) perfect adversarial detection".

The multiLID is an improvment of the original [LID](https://github.com/adverML/lid_adversarial_subspace_detection). This paper is an improvement of the LID [1], which was just meant trying to describe the adversarial subspaces.
Later in [2], the LID is even meant not be mathematically effective enough to detect adversarial examples. We did some changes and could improve the detection.


## Installation

```
pip install -r requirements.txt
cd submodules; git submodule add -b multilid https://github.com/adverML/auto-attack.git autoattack
```


### Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/pdf/2212.06776.pdf):

```
@article{multilid,
  title={Unfolding local growth rate estimates for (almost) perfect adversarial detection},
  author={Lorenz, Peter and Keuper, Margret and Keuper, Janis},
  journal={arXiv preprint arXiv:2212.06776},
  year={2023}
}
```


### References

[1] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Song, D., Houle, M. E., & Bailey, J. (2018). Characterizing adversarial subspaces using local intrinsic dimensionality. Paper presented at 6th International Conference on Learning Representations, ICLR 2018, Vancouver, Canada.

[2] Athalye, A., Carlini, N., & Wagner, D. (2018, July). Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In ICML (pp. 274-283). PMLR.
