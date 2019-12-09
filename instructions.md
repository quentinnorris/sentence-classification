# CNN-Sentence-Classifier
Work is based off of: https://github.com/shagunsodhani/CNN-Sentence-Classifier

Simplified implementation of "Convolutional Neural Networks for Sentence Classification" paper
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.971868.svg)](https://doi.org/10.5281/zenodo.971868)

## General Usage

* Install [Keras](https://keras.io/#installation)
* Repository contains "Movie reviews with one sentence per review" (Pang and Lee, 2005) dataset in `sample_dataset`.
* Alternatively, to use some other dataset, make two files
    * `input.txt` where each line is a sentence to be classified
    * `label.txt` where each line is the label for corresponding line in `input.txt`
* Make `model` folder by running `mkdir model`
* Refer [this](http://nlp.stanford.edu/projects/glove/) to train or download Glove embeddings and [this](https://code.google.com/archive/p/word2vec/) for Word2Vec embeddings.
* Run `python3 app/train.py --data_dir=path_to_folder_containing_input.txt_and_label.txt --embedding_file_path=path_to_embedding_vectors_file --model_name=name_of_model_from_the_paper`
* For example, if data is in `data` folder, embedding file is `vectors.txt` and model is `cnn_static`, run `python3 app/train.py --data_dir=data --embedding_file_path=vectors.txt --model_name=cnn_static`
* To define your own model, pass `model_name` as `self`, define your model in [app/model/model.py](app/model/model.py) and invoke from `model_selector` function (in [model.py](app/model/model.py)).
* All supported arguments can be seen in [here](app/utils/argumentparser.py)

## Project Specific Usage

In order to run EFFECTIVELY, type the following command

python3 app/train.py --data_dir=data --embedding_file_path=vectors.txt --model_name=cnn-rand

To change the datasets that you are running the model on, you go to filereader.py and uncomment the other sets: note that MR, subj, trec all require additional preprocessing - I get a decoding utf-8 error everytime we utilize one of those

Current embedding file path is vectors.txt which is the same as glove.6B.100d.txt (maybe we can look into changing this and seeing what kind of results we get?) works for now 

This project is setup to work with Keras2 (not keras1 - if you are using keras1 we have changed the Merge command to concatenate )

## References

* [Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014)](https://arxiv.org/abs/1408.5882)
* [Summary of paper](https://gist.github.com/shagunsodhani/9ae6d2364c278c97b1b2f4ec53255c56)
