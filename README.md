# The Sparsely Gated Mixture of Experts Layer for PyTorch

This repository contains the implementation of the MoE layer described in the paper ["Outrageously Large Neural Networks"](https://arxiv.org/abs/1701.06538) for PyTorch. 

The code is based on the TensorFlow implementation that can be found [here](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py).


# Example

The file ```example.py``` contains an example illustrating how to train and evaluate the MoE layer with dummy inputs and targets.

This example was tested using torch v1.0.0 and Python v3.6.1 on CPU.
