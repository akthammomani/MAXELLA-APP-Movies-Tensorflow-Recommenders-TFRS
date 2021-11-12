# MAXELLA App
*Movies Recommendation using Tensorflow Recommenders (TFRS)*

<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/67468718/126877962-1c3737b7-69bb-40f4-a92f-7652d52240ac.JPG">
</p>

## 1. Introduction 

Over the last few decades, with the advent of YouTube, Amazon, Netflix and many other Web services, recommendation platforms are becoming much more part of our lives from e commerce, suggesting the customers articles that could be of interest.
In a very general way, Recommendation systems are algorithms program to present related things to users with items being movies to watch, books to read, products to buy or anything else, depending on the industry. Recommendation systems are very important in some industries because they can produce a large amount of money if they're effective, or if they are a way of standing out dramatically from competitors. The aim of the recommendation framework is to produce relevant suggestions for the collection of users of objects or products that may be of interest to them. Suggestions for Amazon books or Netflix shows are all really world examples of how industry leading systems work. The architecture of such recommendation engines depends on the domain and basic characteristics of the available data.

There are mainly Three types of recommendation engines:
 * Collaborative filtering.
 * Content based Filtering.
 * Hybrid (Combination of Collaborative and Content based Filtering).

<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/67468718/141490540-cdf1e250-0282-4c5d-8496-f9fd3e3b8217.JPG">
</p>
![recommender_types](https://user-images.githubusercontent.com/67468718/141490540-cdf1e250-0282-4c5d-8496-f9fd3e3b8217.JPG)

## Dataset: 

**Movie Lens** contains a set of movie ratings from the MovieLens website, a movie recommendation service. This dataset was collected and maintained by [GroupLens](https://grouplens.org/) , a research group at the University of Minnesota. There are 5 versions included: "25m", "latest-small", "100k", "1m", "20m". In all datasets, the movies data and ratings data are joined on "movieId". The 25m dataset, latest-small dataset, and 20m dataset contain only movie data and rating data. The 1m dataset and 100k dataset contain demographic data in addition to movie and rating data.

**movie_lens/1m** can be treated in two ways:

  * It can be interpreted as expressing which movies the users watched (and rated), and which they did not. This is a form of *implicit feedback*, where users' watches tell us which things they prefer to see and which they'd rather not see (This means that every movie a user watched is a positive example, and every movie they have not seen is an implicit negative example).
  * It can also be seen as expressesing how much the users liked the movies they did watch. This is a form of *explicit feedback*: given that a user watched a movie, we can tell roughly how much they liked by looking at the rating they have given.



**(1) [movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings):**
 * Config description: This dataset contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens. Ratings are in whole-star increments. This dataset contains demographic data of users in addition to data on movies and ratings.
 * This dataset is the largest dataset that includes demographic data from movie_lens.
 * "user_gender": gender of the user who made the rating; a true value corresponds to male
 * "bucketized_user_age": bucketized age values of the user who made the rating, the values and the corresponding ranges are:
   * 1: "Under 18"
   * 18: "18-24"
   * 25: "25-34"
   * 35: "35-44"
   * 45: "45-49"
   * 50: "50-55"
   * 56: "56+"
 * "movie_genres": The Genres of the movies are classified into 21 different classes as below:
   *  0: Action
   * 1: Adventure
   * 2: Animation
   * 3: Children
   * 4: Comedy
   * 5: Crime
   * 6: Documentary
   * 7: Drama
   * 8: Fantasy
   * 9: Film-Noir
   * 10: Horror
   * 11: IMAX
   * 12: Musical
   * 13: Mystery
   * 14: Romance
   * 15: Sci-Fi
   * 16: Thriller
   * 17: Unknown
   * 18: War
   * 19: Western
   * 20: no genres listed
 * "user_occupation_label": the occupation of the user who made the rating represented by an integer-encoded label; labels are preprocessed to be consistent across different versions
 * "user_occupation_text": the occupation of the user who made the rating in the original string; different versions can have different set of raw text labels
 * "user_zip_code": the zip code of the user who made the rating.
 * Download size: 5.64 MiB
 * Dataset size: 308.42 MiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): No
 * Features:
 ```
 FeaturesDict({
               'bucketized_user_age': tf.float32,
               'movie_genres': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=21)),
               'movie_id': tf.string,
               'movie_title': tf.string,
               'raw_user_age': tf.float32,
               'timestamp': tf.int64,
               'user_gender': tf.bool,
               'user_id': tf.string,
               'user_occupation_label': ClassLabel(shape=(), dtype=tf.int64, num_classes=22),
               'user_occupation_text': tf.string,
               'user_rating': tf.float32,
               'user_zip_code': tf.string,
             })
 ```
 Example:
|bucketized_user_age	|movie_genres|	movie_id|	movie_title|	raw_user_age|	timestamp|	user_gender|	user_id	|user_occupation_label|	user_occupation_text	|user_rating	|user_zip_code|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|45.0	|7 (Drama)|b'357'	|b"One Flew Over the Cuckoo's Nest (1975)"	|46.0	|879024327	|True	|b'138'	|4 (doctor/health care)	|b'doctor'	|4.0|	b'53211'|


**(2) [movie_lens/1m-movies](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-movies):**

 * Config description: This dataset contains data of approximately 3,900 movies rated in the 1m dataset.
 * Download size: 5.64 MiB
 * Dataset size: 351.12 KiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): Yes
 * Features:
```
FeaturesDict({
              'movie_genres': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=21)),
              'movie_id': tf.string,
              'movie_title': tf.string,
            })
```
Example:
|movie_genres	|movie_id	|movie_title|
|:-----:|:-----:|:-----:|
|4 (Comedy) |b'1681'	|b'You So Crazy (1994)'|	
|0 (Action)|b'838'	|b'In the Line of Duty 2 (1987)'|


**Citation:**
```
@article{10.1145/2827872,
author = {Harper, F. Maxwell and Konstan, Joseph A.},
title = {The MovieLens Datasets: History and Context},
year = {2015},
issue_date = {January 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {4},
issn = {2160-6455},
url = {https://doi.org/10.1145/2827872},
doi = {10.1145/2827872},
journal = {ACM Trans. Interact. Intell. Syst.},
month = dec,
articleno = {19},
numpages = {19},
keywords = {Datasets, recommendations, ratings, MovieLens}
           }
```

## TensorFlow Recommenders Installation:

For free error **TensorFlow Recommenders (TFRS)** installation, we've created a fresh conda enviroment before proceeding with **TensorFlow Recommenders (TFRS)** installation and reviewed all packages versions to make sure they're matching **[TensorFlow Recommenders (TFRS) Requirements](https://www.tensorflow.org/install)** to avoid any conflicts:

```
# TensorFlow is tested and supported on Windows 7 or later 64-bit systems: (Python 3.6–3.8)
# Create a fresh conda enviroment: Note Python 3.6–3.8
conda create --name tfrs python=3.7

# Browse Existing Conda environments:
conda env lis

# Activate the new environment to use it:
conda activate tfrs

# Requires the latest pip:
pip install --upgrade pip

# Current stable release for CPU and GPU:
pip install tensorflow

# Install tensorflow-recommenders (tfrs):
pip install tensorflow-recommenders

# Make sure we get latest TFRS Datasets:
pip install --upgrade tensorflow-datasets

pip install ipywidgets
```

## Why choosing **TensorFlow Recommenders (TFRS)**?

 * TensorFlow Recommenders (TFRS) is a library for building recommender system models.
 * It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.
 * It's built on Keras and aims to have a gentle learning curve while still giving you the flexibility to build complex models.

TFRS makes it possible to:
 * Build and evaluate flexible recommendation retrieval models.
 * Freely incorporate item, user, and [context information](https://www.tensorflow.org/recommenders/examples/featurization) into recommendation models.
 * Train [multi-task models](https://www.tensorflow.org/recommenders/examples/multitask/) that jointly optimize multiple recommendation objectives.
 
TFRS is open source and available on **[Github](https://github.com/tensorflow/recommenders)**.

To learn more, see the [tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) on how to build a movie recommender system, or check the API docs for the [API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs) reference.
