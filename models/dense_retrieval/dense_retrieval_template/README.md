Dense retrieval template based on what I use for my research.

**Dependencies:** 
 - Hydra
 - Huggingface Transformers
 - Weights and Biases (For logging)
 - pytorch_metric_learning
 - ir_dataset_package (See packages directory in this repo)
 - encoding_indexing_package  (See packages directory in this repo)
 - Pytorch Lighting (Use only slightly for seed_everything function)

**args.py**

Has the arguments for training in data classes. Changing these will change the model and training parameters when `train.py` is called.

Arguments are broken into 3 main parts:

 - DataArguments - Which contain arguments about the data and how it is processed
 - ModelArguments - Contain options for the model
 - TrainingArguments - General training arguments including the name to save things under, number of epochs etc.


**train.py**

The main training loop for the model. It creates a data loader, initializes logging, creates the model and optimizer, trains the model, does validation runs, and saves checkpoints.

One part that might be confusing is:

    optimizer = hydra.utils.instantiate(
	    cfg.train_args.optimizer, model.parameters(), _convert_='partial',
    )
This is using the hydra package, which is used for the arguments and allows for easy to use configurations, to create the optimizer. Essentially it is using the information in the training config to create the optimizer using those values. This is really useful as it means any argument in the optimizer can be changed without passing that to the code manually. The `model.parameters()` is the same as for other PyTorch optimizers where the parameters need to be provided to the optimizer. These parameters can be accepted thanks to the ` _convert_='partial'` argument which allows additional arguments to be passed to the instantiated class.


**model.py**

Holds a basic model that returns the CLS token from a huggingface model.


**utils.py**

Hold utilities, right now I just have a class `EmbeddingMemoryQueue` which allows embeddings to be stored from previous batches and reused to increase the number of negative examples.


**validation.py**

Has a basic validator class which allows us to easily validate our model while it is training. Note that the `post_processing_fn` and `processing_before_indexing_fn` may need to be changed depending on the model output. I have changed the `default_processing_fn` and `processing_before_indexing_fn` functions further down in the validation file to hopefully work with the basic model, although I haven't tested them.


