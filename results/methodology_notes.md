# Methodology and claim checks

## Supported by the supplied notebook's code/output

- Dataset used by the executed code: MNIST via `torchvision.datasets.MNIST`.
- 20% random training subset; all 10,000 MNIST test images.
- Two-convolution PyTorch model; batch size 64; `range(5)` training loop.
- Nine optimizer variants; summary accuracies and elapsed times recorded in the notebook outputs.
- The custom optimizer performs top-k filtering, symmetric quantization simulation, and an error-feedback residual buffer.

## Not supported by the supplied notebook

- Fashion-MNIST data, seven epochs, Keras implementation/comparison.
- Layer freezing (all parameters remain trainable in the provided model/optimizer flow).
- STM32/firmware build, on-device training/inference, measured microcontroller RAM, latency, or energy results.
- A measured accuracy curve by epoch: the notebook's chart estimates accuracy from losses.

The supplied PDF report contains these additional claims, but no corresponding source code or device evidence was supplied in the notebook. Verify them and obtain permission from the coauthor before publishing the report or presenting those claims as completed work.

## Measurement limits

The notebook's dataset split, parameter initialization, and data order are not controlled with fixed seeds. The outputs represent one run. RSS change was sampled around process training and reported for only SGD and L-SGD; it should not be interpreted as the optimizer's memory footprint or an 83.75% reduction in model/system RAM. The canonical script adds fixed seeds and new measurements, but a clean rerun is still needed for updated results.
