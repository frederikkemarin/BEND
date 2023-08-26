Adding new embedders
====================


BEND features Embedder classes for a variety of available DNA LMs. DNA LMs are very heterogenous in terms of their
model loading and inference APIs. To add a new DNA LM to BEND, you need to implement a new Embedder class in  ``bend.utils.embedders.py`` that

1. Inherits from the `Embedder` class in ``bend.utils.embedders.BaseEmbedder``
2. Implements the `load_model` method
3. Implements the `embed` method 
4. If necessary, add model source files to the ``bend.models`` directory. This is only necessary if the model implementation cannot be pip installed and imported as a python module directly.


Implementing the ``load_model`` method
======================================

The ``load_model`` method should load the pretrained model and tokenizer, and store them as attributes of the class.
Additionally, it should ensure that the model is in eval mode and move the model to ``device``, which is a global variable defined in ``bend.utils.embedders.py``.
If there are other configurations that need to be set for the model, they should be set here as well, and if necessary be part of the ``load_model`` method's signature.


Implementing the ``embed`` method
=================================

The ``embed`` method should take a list of DNA sequences as input and return a list of embeddings for each sequence. The input sequences are provided as a list of strings, where each string is a DNA sequence. The output embeddings should be a list of numpy arrays, where each numpy array is the embedding for the corresponding input sequence.

In broad terms, most existing DNA LMs require the following steps to embed a single sequence:

1. Tokenize the sequence using the tokenizer.
2. Chunk the tokenized sequence into segments of length ``max_seq_len``. Most models have a maximum sequence length that they can process, and sequences longer than this length need to be chunked into smaller segments.
3. Embed each chunk using the model.
4. Optional: Remove any special token positions that were added to the sequence during tokenization from the embedding. This is controlled by the ``remove_special_tokens`` argument.
5. Optional: If the LM's tokenizer merges multiple input nucleotides into a single token, upsample the embedding to match the original input sequence length. This is controlled by the ``upsample_embeddings`` argument.
6. Concate the embeddings for each chunk into a single embedding for the entire sequence.
7. Convert torch embeddings to numpy arrays.

Each of these steps needs to be adjusted to the specific DNA LM. For example, some models do not require chunking, and some models do not require upsampling. The ``embed`` method should implement these steps in the correct order for the specific DNA LM. It is important that ``embed`` can handle sequences of any length, regardless of the DNA LM's inherent limits.

