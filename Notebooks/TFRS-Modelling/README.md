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

## Features Importance Using Deep & Cross Network (DCN-V2)

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
  <img width="500" height="500" src="https://user-images.githubusercontent.com/67468718/141532309-64ed02a9-8539-4443-89d8-ca71db690cb3.JPG">
</p>

![Personal_dataset_features_importance](https://user-images.githubusercontent.com/67468718/141532468-0272f2a7-5242-4dff-b756-22397424e09c.JPG)




