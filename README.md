There are three data files in the repository:
1. bibtex_train.embeddings
2. bibtex_test.embeddings
3. bibtex_low_dimension.embeddings

`bibtex_train.embeddings` is the training dataset file whereas `bibtex_test.embeddings` is test dataset file.

`bibtex_low_dimension.embeddings` is the output file. This file contains the embedding (or low dimensional output) of each data point present in **train** and **test** files.

To run the project follow these steps:
1. Download the whole folder
2. In command line, run `python app.py`

Once the code is done executing, `bibtex_low_dimension.embeddings` will be created.