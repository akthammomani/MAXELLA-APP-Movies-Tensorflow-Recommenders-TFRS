#!/usr/bin/env python
# coding: utf-8

######################################################################
##                   TensorFlow Recommenders (TFRS)                 ##
##           TFRS - Tuning Retrieval (The Two Towers Model)         ##
##                      Created by: Aktham Momani                   ##
##                            Aug 05 2021                           ##
###################################################################### 
# 
# 
# 
# ## Contents:
# 
#  1. Run baseline First Stage: Retrieval 
#  2. Increase embedding_dimension from 32 to 64
#  3. Reduce Adagrad learning_rate:
#   * 0.01
#   * 0.001 (Default)  
#  4. epochs = 32 
#  5. Tuning Summary


# Import the necessary Libararies: 

import os
import pprint
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Text
import pandas as pd
import numpy as np

import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

plt.style.use('ggplot')



# Ratings data.
ratings = tfds.load("movielens/1m-ratings", split="train", shuffle_files=True)
# Features of all the available movies.
movies = tfds.load("movielens/1m-movies", split="train", shuffle_files=True)


#The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:
#View the data from ratings dataset:
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)



#The movies dataset contains the movie id, movie title, and data on what genres it belongs to. Note that the genres are encoded with integer labels:
#View the data from movies dataset:
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)



#Let's select the necessary attributes:

ratings = ratings.map(lambda x: {
                                 "movie_title": x["movie_title"],
                                 "user_id": x["user_id"],
                                })

movies = movies.map(lambda x: x["movie_title"])



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
movie_titles = movies.batch(l)
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


#  1. Run baseline First Stage: Retrieval 


#Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
embedding_dimension = 32




#We define the embedding on the user side, we must transform the user ids into a vector representation:
#we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer:
user_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  #User_ids vocabulary: list of unique integers that represents each user_id
                                  vocabulary=unique_user_ids, mask_token=None),
                                  # We add an additional embedding to account for unknown tokens.
                                  tf.keras.layers.Embedding(len(unique_user_ids) + 1, 
                                  embedding_dimension)#embedding layer with a vector size of 32
                                ])


# We now define the embedding of the movie portion 
movie_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  vocabulary=unique_movie_titles, mask_token=None),
                                  tf.keras.layers.Embedding(len(unique_movie_titles) + 1,
                                  embedding_dimension) #embedding layer with a vector size of 32
                                 ])



#We define the desired metrics : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(
                                     candidates=movies.batch(128).map(movie_model)
                                     )



#The Retrieval task is defined according to the FactorizedTopK metrics:
task = tfrs.tasks.Retrieval(
                            metrics=metrics
                           )


class MovielensModel(tfrs.Model):
    
    def __init__(self, user_model, movie_model):
        super().__init__()
        #The Two Towers: Movie and user Models:
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)



class NoBaseClassMovielensModel(tf.keras.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:

            # Loss computation.
            user_embeddings = self.user_model(features["user_id"])
            positive_movie_embeddings = self.movie_model(features["movie_title"])
            loss = self.task(user_embeddings, positive_movie_embeddings)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Loss computation.
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        loss = self.task(user_embeddings, positive_movie_embeddings)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


#Let's first instantiate the model.
model = MovielensModel(user_model, movie_model)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) # learning_rate default=0.001


# Then shuffle, batch, and cache the training and evaluation data:
# Segment the batches so that the model runs 13 training batches (2^13) and 11 test batches (2^11) per epoch, 
# while having a batch size which is a multiple of 2^n.
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()



# Then, let's train the model:
history_train = model.fit(cached_train, validation_data = cached_test, epochs=5)



plt.subplots(figsize = (16,6))
plt.plot(history_train.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()



#Evaluate the Base model
model.evaluate(cached_test, return_dict=True)


# 2. Increase embedding_dimension from 32 ==> 64



#Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting:
embedding_dimension = 64



#We define the embedding on the user side, we must transform the user ids into a vector representation:
#we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer:
user_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  #User_ids vocabulary: list of unique integers that represents each user_id
                                  vocabulary=unique_user_ids, mask_token=None),
                                  # We add an additional embedding to account for unknown tokens.
                                  tf.keras.layers.Embedding(len(unique_user_ids) + 1, 
                                  embedding_dimension)#embedding layer with a vector size of 64
                                ])


# So, basically we want an embedding layer for however many user ids we have where each one is represented by a vector of 64 floating point values that 
# basically represents in that 64-dimensional space how similar users 



# We now define the embedding of the movie portion 
movie_model = tf.keras.Sequential([
                                  tf.keras.layers.experimental.preprocessing.StringLookup(
                                  vocabulary=unique_movie_titles, mask_token=None),
                                  tf.keras.layers.Embedding(len(unique_movie_titles) + 1,
                                  embedding_dimension) #embedding layer with a vector size of 64
                                 ])



#We define the desired metrics : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(
                                     candidates=movies.batch(128).map(movie_model)
                                     )



#The Retrieval task is defined according to the FactorizedTopK metrics:
task = tfrs.tasks.Retrieval(
                            metrics=metrics
                           )



#Let's first instantiate the model.
model_1 = MovielensModel(user_model, movie_model)

model_1.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) # learning_rate default=0.001



# Then, let's train the model:
history_train_1 = model_1.fit(cached_train, validation_data = cached_test, epochs=5)



plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_1.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_1.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()



#Evaluate the Base model
model_1.evaluate(cached_test, return_dict=True)


# 3. Reduce Adagrad learning_rate:
# 3.1 learning_rate = 0.01


#Let's first instantiate the model.
model_2 = MovielensModel(user_model, movie_model)

model_2.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01)) # learning_rate default=0.001



# Then, let's train the model:
history_train_2 = model_2.fit(cached_train, validation_data = cached_test, epochs=5)



plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_2.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_2.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()




#Evaluate the Base model
model_2.evaluate(cached_test, return_dict=True)


# 3.2 learning_rate = 0.001 (Default)


#Let's first instantiate the model.
model_3 = MovielensModel(user_model, movie_model)

model_3.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)) # learning_rate default=0.001


# Then, let's train the model:
history_train_3 = model_3.fit(cached_train, validation_data = cached_test, epochs=5)



plt.subplots(figsize = (16,6))
plt.plot(history_train_3.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_3.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_3.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()


#Evaluate the Base model
model_3.evaluate(cached_test, return_dict=True)


# 4. epochs = 32


#Let's first instantiate the model.
model_4 = MovielensModel(user_model, movie_model)

model_4.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)) # learning_rate default=0.001


# Then, let's train the model:
history_train_4 = model_4.fit(cached_train, validation_data = cached_test, epochs=32)




plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()


plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_4.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs - Top 100", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['factorized_top_k/top_50_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_4.history['val_factorized_top_k/top_50_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs - Top 50", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()



plt.subplots(figsize = (16,6))
plt.plot(history_train_4.history['factorized_top_k/top_10_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train_4.history['val_factorized_top_k/top_10_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs - Top 10", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()



#Evaluate the Base model
model_4.evaluate(cached_test, return_dict=True)


# 5. Tuning Summary
# 
# As we can see below, we managed to improve accuracy and reduce loss by: 
# 
#  1. Increase embedding_dimension from 32-> 64
#  2. keeping learning_rate 0.1
# 
# As a result, loss was reduced from 903.56 (Baseline) to 846.05 and top_10_accuracy was improved from 3.2% to 4.3%.
# 


# Recommend the 5 best movies for user 42:

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model_4.user_model)

# recommends movies out of the entire movies dataset.
index.index(movies.batch(100).map(model_4.movie_model), movies)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :5]}")


# 6. Model serving


# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model")

  # Save the index.
    index.save(path)

  # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.keras.models.load_model(path)

  # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded(["42"])
    
    print(f"Recommendations: {titles[0][:5]}")



from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))





