# Synthetic Texts (clickbaits) Generation using Different Variations of VAE
Code for paper **"5 Sources of Clickbaits You Should Know! Using Synthetic Clickbaits to Improve Prediction and Distinguish between Bot-Generated and Human-Written Headlines"**

This code is heavily based on, and thanks to, `https://github.com/kefirski/pytorch_RVAE`

* Need to train word embedding first on your own dataset by using `train_word_embeddings.py` script.
* Please refer to `startTrain.sh` and `startSample.sh` for training and inference scripts.
* `utils.py` includes code for extracting NLP features for detecting clickbaits in the paper.
