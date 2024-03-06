# Flextrain

A teaching repository for diffusion based models and various training workflows.

## Setup

Using conda:

    conda create -n flextrain python=3.10
    conda activate flextrain

    git clone https://github.com/civodlu/flexdat.git
    pip install -e flexdat[all]
    git clone https://github.com/civodlu/flextrain.git
    pip install -e flextrain[all]


## Environment

Environment variables:

    DATASETS_ROOT: where to download the datasets
    EXPERIMENT_ROOT: where the experiment loggings are exported
    STARTED_WITHIN_VSCODE: `1` for debug mode
    OUTPUT_ARTEFACT_ROOT: where are exported trained models


## Notes

Notes:

*  Initialization is very important! Model will converge much more slowly if init is not right. 
    https://arxiv.org/pdf/1901.09321v2.pdf seems to be very useful


## Principles

Design principles:

* nn.Module have batch as input. Use `ModelBatchAdaptor` if needed
* loss should follow `loss_function_type`
* Lightning modules should be lightweight. Use `process_loss_outputs` to handle metrics/losses