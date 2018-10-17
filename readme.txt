Assignment 4
Hidden Markov Models

HMM based approach to POS tagging using Viterbi algorithm.

Training Corpus : BERP Corpus
	|
	 `--- training/hmm-training.txt

Trained Model (pkl)
	|
	 `--- training/pickled_model

Python file for training model : pos-tagger-trainer.py
Python file for POS decoding : pos-tagger.py

Details of Implementation:

1) Extract individual counts for tag bigram and tag word likelihoods in the BERP corpus.
2) For the unknown words in the likelihood table, we change all words with single occurrence to UNK and use this for all unknwon words.
3) The tag bigram is add one smoothed.
4) Using the above, Viterbi is implemented on the HMM.