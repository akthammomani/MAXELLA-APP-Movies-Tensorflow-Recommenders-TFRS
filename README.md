# MAXELLA App for Movies Recommendation using Tensorflow Recommenders (TFRS)

<p align="center">
  <img width="500" height="500" src="https://user-images.githubusercontent.com/67468718/126877962-1c3737b7-69bb-40f4-a92f-7652d52240ac.JPG">
</p>

## Dataset: 

**Movie Lens** contains a set of movie ratings from the MovieLens website, a movie recommendation service. This dataset was collected and maintained by [GroupLens](https://grouplens.org/) , a research group at the University of Minnesota. There are 5 versions included: "25m", "latest-small", "100k", "1m", "20m". In all datasets, the movies data and ratings data are joined on "movieId". The 25m dataset, latest-small dataset, and 20m dataset contain only movie data and rating data. The 1m dataset and 100k dataset contain demographic data in addition to movie and rating data: 

 * "25m": This is the latest stable version of the MovieLens dataset. It is recommended for research purposes.
 * "latest-small": This is a small subset of the latest version of the MovieLens dataset. It is changed and updated over time by GroupLens.
 * "100k": This is the oldest version of the MovieLens datasets. It is a small dataset with demographic data.
 * "1m": This is the largest MovieLens dataset that contains demographic data.
 * "20m": This is one of the most used MovieLens datasets in academic papers along with the 1m dataset.

**For this project will focus mainly in **100K** dataset**: 100k-ratings & 100k-movies

The "100k-ratings" versions in addition include the following demographic features.

 * "user_gender": gender of the user who made the rating; a true value corresponds to male
 * "bucketized_user_age": bucketized age values of the user who made the rating, the values and the corresponding ranges are:
 
   * 1: "Under 18"
   * 18: "18-24"
   * 25: "25-34"
   * 35: "35-44"
   * 45: "45-49"
   * 50: "50-55"
   * 56: "56+"
   
 * "user_occupation_label": the occupation of the user who made the rating represented by an integer-encoded label; labels are preprocessed to be consistent across different versions
 * "user_occupation_text": the occupation of the user who made the rating in the original string; different versions can have different set of raw text labels
 * "user_zip_code": the zip code of the user who made the rating.

In addition, the "100k-ratings" dataset would also have a feature "raw_user_age" which is the exact ages of the users who made the rating

```
Datasets with the "-movies" suffix contain only "movie_id", "movie_title", and "movie_genres" features.

 * Homepage: https://grouplens.org/datasets/movielens/
 * Source code: tfds.structured.MovieLens
 * Versions:
   * 0.1.0 (default): No release notes.
 * Supervised keys (See as_supervised doc): None
 * Figure (tfds.show_examples): Not supported.
```
**(1) movie_lens/100k-ratings:**
 * Config description: This dataset contains 100,000 ratings from 943 users on 1,682 movies. This dataset is the oldest version of the MovieLens dataset.
Each user has rated at least 20 movies. Ratings are in whole-star increments. This dataset contains demographic data of users in addition to data on movies and ratings.
 * Download size: 4.70 MiB
 * Dataset size: 32.41 MiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): Yes
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

**(2) movie_lens/100k-movies:**

 * Config description: This dataset contains data of 1,682 movies rated in the 100k dataset.
 * Download size: 4.70 MiB
 * Dataset size: 150.35 KiB
 * Auto-cached ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)): Yes
 * Features:
```
FeaturesDict({
    'movie_genres': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=21)),
    'movie_id': tf.string,
    'movie_title': tf.string,
})
```



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

