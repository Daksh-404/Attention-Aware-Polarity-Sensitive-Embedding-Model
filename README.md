# Attention Aware Polarity Sensitive Embedding Model

## Dependencies
Tested with Python 3.8
Pytorch v0.5.0
## Dataset
FI dataset: [Google](https://drive.google.com/file/d/1pybbqRoh0xlW1ipu2NqsySHS_fxCRrTN/view?usp=sharing)

## Updates
I have updated the code for the given dataset. I have tried to make it as efficient and fast as possible by using GPUs and CPUs appropriately. The particular updates have been elucidated in the next section. The bugs seem to be system dependent somehow, and the code may throw some other errors on your system, hence, some debugging on your part, might still be required. I have tried and tested it on **Google Colab** with GPU and 12 GB RAM, and it seems to work fine with my new changes. The photo snippets of the outputs will be updated in this README in the coming updates. Cheers!

## Contributions:
* Designed and implemented the driver file for the given dataset.
* Designed and implemented the new dataset python file to take input,organize and pass the dataset.

## Optimizations and Corrections:
* Intermediate tensors filling up space in `forward()` of `ResNet()`, when iterated using
`._modules.items()`. Resolved using a wrapper of `Variable` around all of the repeating
variables
* Error in dimensions in the new loss function. I had to slice it up to two terms instead of 8
for polarity.
* Using dot function , `torch.matmul`, for multiplying two 2-D tensors
* Empty list was being passed in the dataset module. I added the required labels
* Had to convert some of the tensors to long before using their indexing in the new loss
function using `.long()`



## Original Authors:
1. https://github.com/adambielski
2. https://github.com/Xingxu1996

I thank them for the required coding modules.
