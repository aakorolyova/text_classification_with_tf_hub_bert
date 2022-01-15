# text_classification_with_tf_hub_bert
This repository contains my experiments and learnings on binary, multi-class, multi-label text classification using BERT model via tensorflow_hub.
As I was learning how to use BERT with tf_hub, I used this excellent repo a lot: https://github.com/strongio/keras-bert/blob/master/keras-bert.py - this code is used as a basis for my experinemts. I extended it for multi-class and multi-label classification, and added pre-processing functions for tasks taking a pair of text as input.

For my binary classification exercise, I used IMDB movie review dataset (https://ai.stanford.edu/~amaas/data/sentiment/).
Some of the datasets used here are from various hackathons and competitions:
- Predict The News Category Hackathon (https://machinehack.com/hackathon/predict_the_news_category_hackathon/overview) - multi-class classification
- uHack Sentiments 2.0: Decode Code Words (https://machinehack.com/hackathon/uhack_sentiments_20_decode_code_words/overview) - multi-label classification
- Science nlp classification (https://www.kaggle.com/c/nlpsci/leaderboard#score) - multi-class classification with a pair of texts (title and abstract of an article) as input.
