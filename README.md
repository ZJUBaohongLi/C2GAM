# Coupled Counterfactual Generative Adversarial Model (C<sup>2</sup>GAM)

Source code for [A Generative Approach for Treatment Effect Estimation under Collider Bias: From an Out-of-Distribution Perspective](https://proceedings.mlr.press/v235/li24al.html).

## Get Started

1. To install the necessary packages, run the following command-line code.
```
pip install -r requirements.txt
```

2. Run the demo (experiments on IHDP) in `main.py`.

## Useful Links

- `wasserstein_distance.py` is adapted from [SinkhornAutoDiff](https://github.com/gpeyre/SinkhornAutoDiff).

## Citation

```
@InProceedings{pmlr-v235-li24al,
  title={A Generative Approach for Treatment Effect Estimation under Collider Bias: From an Out-of-Distribution Perspective},
  author={Li, Baohong and Li, Haoxuan and Wu, Anpeng and Zhu, Minqin and Peng, Shiyuan and Cao, Qingyu and Kuang, Kun},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={28132--28145},
  year={2024},
  organization={PMLR}
}
```