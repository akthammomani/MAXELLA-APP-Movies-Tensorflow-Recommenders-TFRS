# Features Importance Using Deep & Cross Network (DCN-V2)


![logo_small](https://user-images.githubusercontent.com/67468718/127299425-533f0a3c-c965-42a5-886a-5272170d9e0e.JPG)



## 1. Introduction: Deep & Cross Network (DCN-V2)

**Deep and cross network, short for DCN**, came out of Google Research, and is designed to learn explicit and bounded-degree cross features effectively:
 * large and sparse feature space is extremely hard to train.
 * Oftentimes, we needed to do a lot of manual feature engineering, including designing cross features, which is very challenging and less effective.
 * Whilst possible to use additional neural networks under such circumstances, it's not the most efficient approach.
 
***Deep and cross network (DCN) is specifically designed to tackle all above challenges.***


### 1.1 Feature Cross

Let's say we're building a recommender system to sell a blender to customers. Then our customers' past purchase history, such as purchased bananas and purchased cooking books, or geographic features are single features. If one has purchased both bananas and cooking books, then this customer will be more likely to click on the recommended blender. The combination of purchased bananas and the purchased cooking books is referred to as feature cross, which provides additional interaction information beyond the individual features. You can keep adding more cross features to even higher degrees:

![feature_cross](https://user-images.githubusercontent.com/67468718/135774082-c46d2f22-ea20-451c-af1a-ee4453f69176.JPG)

### 1.2 Cross Network

In real world recommendation systems, we often have large and sparse feature space. So identifying effective feature processes in this setting would often require manual feature engineering or exhaustive search, which is highly inefficient. To tackle this issue, ***Google Research team has proposed Deep and Cross Network, DCN.***

It starts with an input layer, typically an embedding layer, followed by a cross network containing multiple cross layers that models explicitly feature interactions, and then combines with a deep network that models implicit feature interactions. The deep network is just a traditional multilayer construction. But the core of DCN is really the cross network. It explicitly applies feature crossing at each layer. And the highest polynomial degree increases with layer depth. The figure here shows the deep and cross layer in the mathematical form.

![dcn](https://user-images.githubusercontent.com/67468718/135774206-d017c326-2568-49ce-ab13-98b696e6de84.JPG)

### 1.3 Deep & Cross Network Architecture

There are a couple of ways to combine the cross network and the deep networ:
 * Stack the deep network on top of the cross network.
 * Place deep & cross networs in parallel.
 
![DCN_structure](https://user-images.githubusercontent.com/67468718/135774992-b26eabcf-bd9e-40c2-b702-abb1cb193ff7.JPG)

## 2. Model Construction: Deep & Cross Network (DCN-V2):

The model architecture we will be building starts with an embedding layer, which is fed into a cross network followed by a deep network. The embedding dimension is set to 32 for all the features. You could also use different embedding sizes for different features.

### 2.1 DCN (Stacked) 

We first train a DCN model with a stacked structure, that is, the inputs are fed to a cross network followed by a deep network.

![DCN_stacked](https://user-images.githubusercontent.com/67468718/135777802-cacec165-ed8c-4951-85ee-f3fe1f421b4d.JPG)

```
Features Importance using Personal Dataset:
```
![Personal_dataset_features_importance](https://user-images.githubusercontent.com/67468718/140880610-d32918b7-ba3e-44ac-be20-d274608aae14.JPG)

```
Features Importance using TensorFlow Dataset:
```
![tensorflow_dataset_features_importance](https://user-images.githubusercontent.com/67468718/140880611-359efdea-cef6-424b-8a22-3ffbbeab48f7.JPG)

### 2.2 Low-rank DCN 

To reduce the training and serving cost, we leverage low-rank techniques to approximate the DCN weight matrices. The rank is passed in through argument projection_dim; a smaller projection_dim results in a lower cost. Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost. In practice, we've observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.

![low-rank-dcn](https://user-images.githubusercontent.com/67468718/137226644-e5bebcf9-2648-407c-94a5-108435076570.JPG)


### 2.3 DNN (Cross-Layer=False)

We train a same-sized DNN model as a reference.


## 3. Modeling Summary

As we can see above from the features importance both of the new features added to the dataset : "director" and "star" are very importance with user_id.

```
DCN            RMSE mean: 1.0354, stdv: 0.0225
DCN (low-rank) RMSE mean: 1.0098, stdv: 0.0207
DNN            RMSE mean: 0.9535, stdv: 0.0244
```



