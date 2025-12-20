**Transformer-Based Modeling of Neuronal Activity**

Thesis Code Repository

**Overview**

This repository contains the code developed for my thesis, which focuses on modeling neuronal activity using a Transformer-based deep learning architecture. The goal of this work is to predict neuron behavior, specifically soma membrane potential and spike generation from high-dimensional dendritic input signals.

The model is designed to capture both local spatial structure across dendritic channels and temporal dependencies over time, while remaining causal and biologically motivated.

**Model Description**

At each time step, dendritic inputs are treated as tokens and processed through causal convolutional layers to extract local spatial features. These features are projected into a latent representation and passed through a stack of Transformer-like blocks with local-causal self-attention.

Temporal information is encoded using Rotary Positional Embeddings (RoPE) applied within the attention mechanism, enabling efficient modeling of long sequences while preserving temporal order.

To reflect the different nature of neuronal outputs, the architecture splits into two prediction heads as you can see in the attached image:
**
Soma voltage head:** An LSTM layer followed by a linear projection, used to model smooth, continuous membrane potential dynamics.

**Spike head:** A separate linear layer for predicting discrete spike activity.

This separation improves stability, interpretability, and alignment with known neuronal dynamics.
<img width="604" height="620" alt="image" src="https://github.com/user-attachments/assets/3e1aeb22-531a-4dad-81d7-8985efd44663" />
