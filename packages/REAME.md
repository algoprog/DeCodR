These are two packages I, Julian, have been working on. They provide useful functionality for IR and might be helpful.

## Overview of encoding_indexing_package

It is useful when encoding a whole set of passages or documents. It also will index them using Faiss and provide the results of a search. In addition, it has built in metrics so that you can easily find validation and test metrics.

**Dependencies:**
 - Huggingface Transformers
 - PyTorch
 - tqdm
 - pytrec_eval
 - Numpy
 - Faiss
 - psutil

## Overview of ir_dataset_package

It has various datasets that allow data to be created for training and testing IR systems. It handles tokenization automatically and allows for lazy tokenization.

**Dependencies:**
 - Huggingface Transformers
 - PyTorch
 - Huggingface Datasets

## Installation
To install these packages, navigate into each package's directory and run the following command:

    python3 -m pip install -e .
This will install the package, from then on it can be used like any other package.
