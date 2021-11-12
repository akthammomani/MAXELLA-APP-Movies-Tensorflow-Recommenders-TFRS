# Machine Learning Modeling: TensorFlow Recommenders (TFRS)

<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/67468718/126877962-1c3737b7-69bb-40f4-a92f-7652d52240ac.JPG">
</p>

## Introduction

Here comes the really fun step: modeling! For this step, we'll be:
 * Feature importance using Deep and Cross Network (DCN-v2)
 * Training multiple TensorFlow Recommenders.
 * Apply hyperparameters tuning where applicable to ensure every algorithm will result in best prediction possible.
 * Finally, evaluate these Models.

## 1. Features Importance Using Deep & Cross Network (DCN-V2)

Deep and cross network, short for DCN, came out of Google Research, and is designed to learn explicit and bounded-degree cross features effectively:
 * Large and sparse feature space is extremely hard to train.
 * Oftentimes, we needed to do a lot of manual feature engineering, including designing cross features, which is very challenging and less effective.
 * Whilst possible to use additional neural networks under such circumstances, it's not the most efficient approach.
 Deep and cross network (DCN) is specifically designed to tackle all above challenges. 

**Feature Cross**

Let's say we're building a recommender system to sell a blender to customers. Then our customers' past purchase history, such as purchased bananas and purchased cooking books, or geographic features are single features. If one has purchased both bananas and cooking books, then this customer will be more likely to click on the recommended blender. The combination of purchased bananas and the purchased cooking books is referred to as feature cross, which provides additional interaction information beyond the individual features.

**Cross Network**

In real world recommendation systems, we often have large and sparse feature space. So, identifying effective feature processes in this setting would often require manual feature engineering or exhaustive search, which is highly inefficient. To tackle this issue, Google Research team has proposed Deep and Cross Network, DCN.
It starts with an input layer, typically an embedding layer, followed by a cross network containing multiple cross layers that models explicitly feature interactions, and then combines with a deep network that models implicit feature interactions. The deep network is just a traditional multilayer construction. But the core of DCN is really the cross network. It explicitly applies feature crossing at each layer. And the highest polynomial degree increases with layer depth. The figure here shows the deep and cross layer in the mathematical form.

**Deep & Cross Network**

There are a couple of ways to combine the cross network and the deep network:
 * Stack the deep network on top of the cross network.
 * Place deep & cross networks in parallel.

**Model Structure**

We first train a DCN model with a stacked structure, that is, the inputs are fed to a cross network followed by a deep network.
 
<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/67468718/141532309-64ed02a9-8539-4443-89d8-ca71db690cb3.JPG">
</p>

**DCN-v2: Features Importance**

<p align="center">
  <img width="500" height="400" src="https://user-images.githubusercontent.com/67468718/141532468-0272f2a7-5242-4dff-b756-22397424e09c.JPG">
</p>

One of the nice things about DCN is that we can visualize the weights from the cross network and see if it has successfully learned the important feature process.
As shown above, the stronger the interaction between two features is. In this case, the feature cross of user ID and movie ID, director, star are of great importance.


## 2. First Stage: Retrieval (The Two Towers Model)

Real-world recommender systems are often composed of two stages:

 * The retrieval stage (Selects recommendation candidates): is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.

 * The ranking stage (Selects the best candidates and rank them): takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

Retrieval models are often composed of two sub-models:

The retrieval model embeds user ID's and movie ID's of rated movies into embedding layers of the same dimension:

 * A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.
 * A candidate model computing the candidate representation (an equally-sized vector) using the candidate features.
 
As shown below, the two models are multiplied to create a query-candidate affinity scores for each rating during training. If the affinity score for the rating is higher than other for other candidates, then we can consider the model is a good one!

<p align="center">
  <img width="500" height="400" src="https://user-images.githubusercontent.com/67468718/141532926-3dab0bd3-2f8f-4a68-9b32-1461fdca0693.JPG">
</p>

**Embedding layer Magic**

As shown above, we might think of the embedding layers as just a way of encoding right a way of forcing the categorical data into some sort of a standard format that can be easily fed into a neural network and usually that's how it's used but embedding layers are more than that! The way they're working under the hood is every unique id is being mapped to a vector of n dimensions maybe it's 32 dimensions or 64 dimensions, but it's going to be like a vector of 32 floating point values and we can think of this as a position in a 32-dimensional space that represents the similarity between one user id and another or between one movie id and another so by using embedding layers in this way we're kind of getting around that whole problem of data sparsity and sparse vectors and at the same time, we're getting a measure of similarity out of the deal as well so it's a very simple and straightforward way of getting a recommendation candidates and then we could just brute force sort them all and get our top k recommendations.


The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.

In this Model, we built and trained such a two-tower model using the Movielens dataset (1m Dataset):
 * Get our data and split it into a training and test set.
 * Implement a retrieval model.
 * Fit and evaluate it.

**Tuning Summary**

As we can see below, we managed to improve accuracy and reduce loss by:
 * Increase embedding_dimension from 32-> 64
 * keeping learning_rate 0.1

As a result, loss was reduced from 903.56 (Baseline) to 846.05 and top_10_accuracy was improved from 3.2% to 4.3%.

![2_tower](https://user-images.githubusercontent.com/67468718/141533117-466991f0-bf30-4a4e-bd8f-c8bdb2bbf81d.JPG)

## 3. Second Stage: Ranking

The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

**Tuning Summary**

As shown below, we managed to reduce RMSE and loss from 93% to 86.89% and 81% to 64% respectively using below: 
 * Increased embedding_dimension from 32 ==> 64.
 * Increased epochs from 3 ==> 32
 * Increase Dense Layers from 2 ==> 4
 * Adding Dropout + Adding Max Norm

![ranking](https://user-images.githubusercontent.com/67468718/141533444-21ec4ccd-a272-4d87-a7a8-460b25bb7375.JPG)

## 4. Multi-Task Model (Joint Model): The Two Tower + The Ranking Models

In the TFRS Modeling - Two-Tower we built a retrieval system using movie watches as positive interaction signals.
In many applications, however, there are multiple rich sources of feedback to draw upon. For example, an e-commerce site may record user visits to product pages (abundant, but relatively low signal), image clicks, adding to cart, and, finally, purchases. It may even record post-purchase signals such as reviews and returns.
Integrating all these different forms of feedback is critical to building systems that users love to use, and that do not optimize for any one metric at the expense of overall performance.
In addition, building a joint model for multiple tasks may produce better results than building a number of task-specific models. This is especially true where some data is abundant (for example, clicks), and some data is sparse (purchases, returns, manual reviews). In those scenarios, a joint model may be able to use representations learned from the abundant task to improve its predictions on the sparse task via a phenomenon known as transfer learning. For example, this paper shows that a model predicting explicit user ratings from sparse user surveys can be substantially improved by adding an auxiliary task that uses abundant click log data.
In this Model, we built a multi-objective recommender for Movielens, using both implicit (movie watches) and explicit signals (ratings).

Due to the important and the high possibility of this model:
 * We’ll focusing in using the important features as shown from DCN-v2 (Figure 25): "user_occupation_text", "user_gender", "director", "star", "bucketized_user_age",
 * Normalize all Numerical Features: timestamps, user_age, user_gender.
 * Reconfigured all Movie Lens Classes (TensorFlow Recommenders) to accommodate the new embedding design due to new features, having deeper neural networks and adding regularization to help overfitting:
   * class UserModel
   * class QueryModel
   * class MovieModel
   * class MovieModel
   * class CandidateModel
   * class MovielensModel


**Tuning Summary**

As shown below, Joint Model failed to beat baseline Joint model:

![multi task](https://user-images.githubusercontent.com/67468718/141533627-d706c4f9-bc8f-421d-ae65-6ed315fe0835.JPG)
