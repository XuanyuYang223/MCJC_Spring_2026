# Toy Mamba / Selective SSM Examples

This repository contains small educational examples inspired by **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**.

The goal is to build intuition for state space models and selective updates, not to reproduce the full Mamba architecture or its optimized CUDA implementation.

Paper Link: https://arxiv.org/abs/2312.00752

## Files

* `toy_mamba_ssm_example.ipynb`
  A NumPy-based toy example showing basic SSM recurrence, hidden-state updates, and simple selective/gated behavior.

* `Toy_example_2.ipynb`
  A PyTorch toy text-classification example using a simplified selective SSM-style layer.

* `Paper Copy.pdf`
  Reference paper: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.

* `Presentation Slide.pdf`
  Presentation slides summarizing the main ideas of SSMs, selective SSMs, and Mamba.

## Main Ideas

The notebooks demonstrate:

* how an SSM stores past information in a hidden state
* how fixed SSM dynamics differ from selective/input-dependent updates
* how simple gates can control when the model remembers or updates information
* how a simplified selective SSM can be used in a tiny PyTorch sequence classifier

## Installation

```bash
pip install numpy matplotlib torch notebook
```

## Run

Open the notebooks with Jupyter:

```bash
jupyter notebook
```

Recommended order:

1. `toy_mamba_ssm_example.ipynb`
2. `Toy_example_2.ipynb`

## Limitations

This is a toy educational implementation. It does not include the full Mamba model, selective scan CUDA kernels, large-scale training, or pretrained checkpoints.

## Reference

Gu, A., & Dao, T.
*Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*

## Author

Xuanyu Yang
