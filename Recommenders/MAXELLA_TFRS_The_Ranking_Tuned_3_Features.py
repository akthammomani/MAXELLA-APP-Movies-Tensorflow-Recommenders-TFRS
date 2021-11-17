#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Recommenders (TFRS) - Second Stage: Ranking


######################################################################
##                   TensorFlow Recommenders (TFRS)                 ##
##           TFRS - Tuning Ranking (Second Stage: Ranking)          ##    
##                      Created by: Aktham Momani                   ##
###################################################################### 


# Import the necessary Libararies: 

import os
import pprint
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Text
from keras.constraints import maxnorm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from typing import Dict, Text
import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

plt.style.use('ggplot')


# Ratings data.
ratings = tfds.load("movielens/1m-ratings", split="train", shuffle_files=True)


#The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
#View the data from ratings dataset:
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)


type(ratings)

# Next, As you can see below, in the map function, the ratings are considered explicit feedback since we can tell roughly how much the users like the movies based on the rating numbers.

#Let's select the necessary attributes:

ratings = ratings.map(lambda x: {
                                 "movie_title": x["movie_title"],
                                 "user_id": x["user_id"],
                                 "user_rating": x["user_rating"]
                                })


len(ratings)


# let's use a random split, putting 75% of the ratings in the train set, and 25% in the test set:
# Assign a seed=42 for consistency of results and reproducibility:
seed = 42
l = len(ratings)

tf.random.set_seed(seed)
shuffled = ratings.shuffle(l, seed=seed, reshuffle_each_iteration=False)

#Save 75% of the data for training and 25% for testing:
train_ = int(0.75 * l)
test_ = int(0.25 * l)

train = shuffled.take(train_)
test = shuffled.skip(train_).take(test_)



# Now, let's find out how many uniques users/movies:
movie_titles = ratings.batch(l).map(lambda x: x["movie_title"])
user_ids = ratings.batch(l).map(lambda x: x["user_id"])

#Movies uniques:
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))

#users unique
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# take a look at the movies:
unique_movie_titles[:10]



#Movies uniques
len_films = len(unique_movie_titles)
print(len_films) 


#users unique
len_users = len(unique_user_ids)
print(len_users) 


class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
                               tf.keras.layers.experimental.preprocessing.StringLookup(
                               vocabulary=unique_user_ids, mask_token=None),
                               tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                                  ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_movie_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                                                   ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
                                          # Learn multiple dense layers.
                                          tf.keras.layers.Dense(256, activation="relu"),
                                          tf.keras.layers.Dense(64, activation="relu"),
                                          # Make rating predictions in the final layer.
                                          tf.keras.layers.Dense(1)
                                          ])

    def call(self, inputs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


# This model takes user ids and movie titles, and outputs a predicted rating:


RankingModel()((["42"], ["7th Voyage of Sinbad, The (1958)"]))


task = tfrs.tasks.Ranking(
                         loss = tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.RootMeanSquaredError()]
                         )



class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                                                              loss = tf.keras.losses.MeanSquaredError(),
                                                               metrics=[tf.keras.metrics.RootMeanSquaredError()]
                                                              )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features["user_id"], features["movie_title"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)



#Let's first instantiate the model.
model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001



# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()



# Then, let's train the model:
history_train = model.fit(cached_train, validation_data = cached_test, epochs=3)



plt.subplots(figsize = (16,6))
plt.plot(history_train.history['total_loss'],color='red', alpha=0.8, label='loss-Train')
plt.plot(history_train.history['val_total_loss'],color='green', alpha=0.8, label='loss-Test')
plt.title("Total Loss over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()




plt.subplots(figsize = (16,6))
plt.plot(history_train.history['root_mean_squared_error'],color='red', alpha=0.8, label='RMSE-Train')
plt.plot(history_train.history['val_root_mean_squared_error'],color='green', alpha=0.8, label='RMSE-Test')
plt.title("RMSE over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#Evaluate the model
model.evaluate(cached_test, return_dict=True)



class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
        embedding_dimension = 64

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
                               tf.keras.layers.experimental.preprocessing.StringLookup(
                               vocabulary=unique_user_ids, mask_token=None),
                               tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                                  ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_movie_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                                                   ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
                                          # Learn multiple dense layers.
                                          tf.keras.layers.Dense(256, activation="relu"),
                                          tf.keras.layers.Dense(64, activation="relu"),
                                          # Make rating predictions in the final layer.
                                          tf.keras.layers.Dense(1)
                                          ])

    def call(self, inputs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))



RankingModel()((["42"], ["7th Voyage of Sinbad, The (1958)"]))


task = tfrs.tasks.Ranking(
                         loss = tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.RootMeanSquaredError()]
                         )



class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                                                              loss = tf.keras.losses.MeanSquaredError(),
                                                               metrics=[tf.keras.metrics.RootMeanSquaredError()]
                                                              )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features["user_id"], features["movie_title"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)



#Let's first instantiate the model.
model_1 = MovielensModel()
model_1.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001

# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()


# Then, let's train the model:
history_train_1 = model_1.fit(cached_train, validation_data = cached_test, epochs=32)


plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['total_loss'],color='red', alpha=0.8, label='loss-Train')
plt.plot(history_train_1.history['val_total_loss'],color='green', alpha=0.8, label='loss-Test')
plt.title("Total Loss over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['root_mean_squared_error'],color='red', alpha=0.8, label='RMSE-Train')
plt.plot(history_train_1.history['val_root_mean_squared_error'],color='green', alpha=0.8, label='RMSE-Test')
plt.title("RMSE over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#Evaluate the model
model_1.evaluate(cached_test, return_dict=True)


class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
        embedding_dimension = 64

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
                               tf.keras.layers.experimental.preprocessing.StringLookup(
                               vocabulary=unique_user_ids, mask_token=None),
                               tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                                  ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_movie_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                                                   ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
                                          # Learn multiple dense layers.
                                          tf.keras.layers.Dense(128, activation="relu"),
                                          tf.keras.layers.Dense(128, activation="relu"),
                                          tf.keras.layers.Dense(64, activation="relu"),
                                          # Make rating predictions in the final layer.
                                          tf.keras.layers.Dense(1)
                                          ])

    def call(self, inputs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))



RankingModel()((["42"], ["7th Voyage of Sinbad, The (1958)"]))



task = tfrs.tasks.Ranking(
                         loss = tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.RootMeanSquaredError()]
                         )



class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                                                              loss = tf.keras.losses.MeanSquaredError(),
                                                               metrics=[tf.keras.metrics.RootMeanSquaredError()]
                                                              )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features["user_id"], features["movie_title"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)



#Let's first instantiate the model.
model_2 = MovielensModel()
model_2.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001



# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()




# Then, let's train the model:
history_train_2 = model_2.fit(cached_train, validation_data = cached_test, epochs=32)



plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['total_loss'],color='red', alpha=0.8, label='loss-Train')
plt.plot(history_train_2.history['val_total_loss'],color='green', alpha=0.8, label='loss-Test')
plt.title("Total Loss over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['root_mean_squared_error'],color='red', alpha=0.8, label='RMSE-Train')
plt.plot(history_train_2.history['val_root_mean_squared_error'],color='green', alpha=0.8, label='RMSE-Test')
plt.title("RMSE over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#Evaluate the model
model_2.evaluate(cached_test, return_dict=True)



class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
        embedding_dimension = 64

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
                               tf.keras.layers.experimental.preprocessing.StringLookup(
                               vocabulary=unique_user_ids, mask_token=None),
                               tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                                  ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_movie_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                                                   ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
                                          # Learn multiple dense layers.
                                          tf.keras.layers.Dense(128, activation="relu"),
                                          tf.keras.layers.Dense(128, activation="relu"),
                                          tf.keras.layers.Dense(128, activation="relu"),
                                          tf.keras.layers.Dense(64, activation="relu"),
                                          # Make rating predictions in the final layer.
                                          tf.keras.layers.Dense(1)
                                          ])

    def call(self, inputs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))



RankingModel()((["42"], ["7th Voyage of Sinbad, The (1958)"]))



task = tfrs.tasks.Ranking(
                         loss = tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.RootMeanSquaredError()]
                         )


class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                                                              loss = tf.keras.losses.MeanSquaredError(),
                                                               metrics=[tf.keras.metrics.RootMeanSquaredError()]
                                                              )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features["user_id"], features["movie_title"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)




#Let's first instantiate the model.
model_3 = MovielensModel()
model_3.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001



# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()


# Then, let's train the model:
history_train_3 = model_3.fit(cached_train, validation_data = cached_test, epochs=32)


plt.subplots(figsize = (16,6))
plt.plot(history_train_3.history['total_loss'],color='red', alpha=0.8, label='loss-Train')
plt.plot(history_train_3.history['val_total_loss'],color='green', alpha=0.8, label='loss-Test')
plt.title("Total Loss over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


plt.subplots(figsize = (16,6))
plt.plot(history_train_3.history['root_mean_squared_error'],color='red', alpha=0.8, label='RMSE-Train')
plt.plot(history_train_3.history['val_root_mean_squared_error'],color='green', alpha=0.8, label='RMSE-Test')
plt.title("RMSE over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#Evaluate the model
model_3.evaluate(cached_test, return_dict=True)


# Deeper Network: Adding weight Constraint (Maximum Norm) and Dropout
# 
# **4 Dense Layers + embedding_dimension = 64 + epochs = 32**
# 
#  * Let's add weight constraint, which imposes a hard rule on the weights, max norm that forces the vector norm of the weights to be below a value=3. Once exceeded all weights in the node are made small enough to meet the constraint.
#  * Let's add Dropout: Generally, using a small dropout value of 20%-50% of neurons with 20% providing a good starting point. A probability too low has minimal effect and a value too high results in under-learning by the network.


class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
        embedding_dimension = 64

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
                               tf.keras.layers.experimental.preprocessing.StringLookup(
                               vocabulary=unique_user_ids, mask_token=None),
                               tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                                  ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_movie_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                                                   ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
                                          # Learn multiple dense layers.

                                          tf.keras.layers.Dense(128, kernel_constraint=maxnorm(3), activation="relu"),
                                          layers.Dropout(0.3),
                                          tf.keras.layers.Dense(128, kernel_constraint=maxnorm(3), activation="relu"),
                                          layers.Dropout(0.3),
                                          tf.keras.layers.Dense(128, kernel_constraint=maxnorm(3), activation="relu"),
                                          layers.Dropout(0.3),
                                          tf.keras.layers.Dense(64, kernel_constraint=maxnorm(3), activation="relu"),
                                          layers.Dropout(0.3),
                                          # Make rating predictions in the final layer.
                                          tf.keras.layers.Dense(1)
                                          ])

    def call(self, inputs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))




RankingModel()((["42"], ["7th Voyage of Sinbad, The (1958)"]))




task = tfrs.tasks.Ranking(
                         loss = tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.RootMeanSquaredError()]
                         )




class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                                                              loss = tf.keras.losses.MeanSquaredError(),
                                                               metrics=[tf.keras.metrics.RootMeanSquaredError()]
                                                              )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model((features["user_id"], features["movie_title"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)




#Let's first instantiate the model.
model_4 = MovielensModel()
model_4.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) #defauly learning_rate=0.001




# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()


# Then, let's train the model:
history_train_4 = model_4.fit(cached_train, validation_data = cached_test, epochs=32)



plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['total_loss'],color='red', alpha=0.8, label='loss-Train')
plt.plot(history_train_4.history['val_total_loss'],color='green', alpha=0.8, label='loss-Test')
plt.title("Total Loss over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['root_mean_squared_error'],color='red', alpha=0.8, label='RMSE-Train')
plt.plot(history_train_4.history['val_root_mean_squared_error'],color='green', alpha=0.8, label='RMSE-Test')
plt.title("RMSE over epochs", fontsize=14)
#plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#Evaluate the model
model_4.evaluate(cached_test, return_dict=True)


test_ratings ={}

for m in test.take(5):
    test_ratings[m["movie_title"].numpy()] = RankingModel()((["42"], [m["movie_title"]]))
    
for m in sorted(test_ratings, key=test_ratings.get, reverse=True):
    print(m)



