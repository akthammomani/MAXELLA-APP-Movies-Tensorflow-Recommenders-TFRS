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
  <img width="500" height="350" src="https://user-images.githubusercontent.com/67468718/141490540-cdf1e250-0282-4c5d-8496-f9fd3e3b8217.JPG">
</p>


## 2. Problem Statement 

This Project is all about how to successfully formulate a recommendation engine, the difference between implicit and explicit feedback and how to build a movie recommendation system with TensorFlow and TFRS. 


**Context**

Google/YouTube is all about connecting people to the movies/videos they love. To help customers find those movies, they developed world-class movie recommendation system called TensorFlow Recommender (TFRS). Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Google/YouTube uses those predictions to make personal videos recommendations based on each user’s unique tastes. 

**Criteria Of Success**

Be apple to successfully build a Movie Recommendation engine with the highest possible retrieval accuracy (Predicting Movies) AND with the lowest Loss/RMSE (Ranking Movies)

**Constraints**

TensorFlow Recommender (TFRS) is a brand new package where there’s very low used cases, so for the success of this project there will be plenty of research involved to successfully complete this project.



## 3. Dataset: 

Because of the richness of the metadata in Tensorflow Movie Lens dataset, we have decided to choose 1 million Movie lens from TensorFlow to be our main dataset for this project. Also, we used both datasets from [Movielens website](https://grouplens.org/datasets/movielens/): movies metadata & credits.

**Movie Lens** contains a set of movie ratings from the MovieLens website, a movie recommendation service. This dataset was collected and maintained by [GroupLens](https://grouplens.org/) , a research group at the University of Minnesota. There are 5 versions included: "25m", "latest-small", "100k", "1m", "20m". In all datasets, the movies data and ratings data are joined on "movieId". The 25m dataset, latest-small dataset, and 20m dataset contain only movie data and rating data. The 1m dataset and 100k dataset contain demographic data in addition to movie and rating data.

**movie_lens/1m** can be treated in two ways:

  * It can be interpreted as expressing which movies the users watched (and rated), and which they did not. This is a form of *implicit feedback*, where users' watches tell us which things they prefer to see and which they'd rather not see (This means that every movie a user watched is a positive example, and every movie they have not seen is an implicit negative example).
  * It can also be seen as expressesing how much the users liked the movies they did watch. This is a form of *explicit feedback*: given that a user watched a movie, we can tell roughly how much they liked by looking at the rating they have given.



**(1) [movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings):**
 * Config description: This dataset contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens. Ratings are in whole-star increments. This dataset contains demographic data of users in addition to data on movies and ratings.
 * This dataset is the largest dataset that includes demographic data from movie_lens.
 * "user_gender": gender of the user who made the rating; a true value corresponds to male
 * "bucketized_user_age": bucketized age values of the user who made the rating, the values and the corresponding ranges are:
 * "movie_genres": The Genres of the movies are classified into 21 different classes as below:
 * "user_occupation_label": the occupation of the user who made the rating represented by an integer-encoded label; labels are preprocessed to be consistent across different versions
 * "user_occupation_text": the occupation of the user who made the rating in the original string; different versions can have different set of raw text labels
 * "user_zip_code": the zip code of the user who made the rating.
 * "release_date": This is the movie release date, in unix epoch (UTC - units of seconds) (int64).
 * "director": This is the director of the movie.
 * "start": This is the main star of the movie.
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
               'timestamp': tf.int64,
               'user_gender': tf.bool,
               'user_id': tf.string,
               'user_occupation_label': ClassLabel(shape=(), dtype=tf.int64, num_classes=22),
               'user_occupation_text': tf.string,
               'user_rating': tf.float32,
               'user_zip_code': tf.string,
               'release_date': tf.int64,
               'director': tf.string,
               'star': tf.string,
             })
 ```

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

## 4. TensorFlow Recommenders Installation:

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

**Why choosing TensorFlow Recommenders (TFRS)?**

 * TensorFlow Recommenders (TFRS) is a library for building recommender system models.
 * It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.
 * It's built on Keras and aims to have a gentle learning curve while still giving you the flexibility to build complex models.

TFRS makes it possible to:
 * Build and evaluate flexible recommendation retrieval models.
 * Freely incorporate item, user, and [context information](https://www.tensorflow.org/recommenders/examples/featurization) into recommendation models.
 * Train [multi-task models](https://www.tensorflow.org/recommenders/examples/multitask/) that jointly optimize multiple recommendation objectives.
 
TFRS is open source and available on **[Github](https://github.com/tensorflow/recommenders)**.

To learn more, see the [tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) on how to build a movie recommender system, or check the API docs for the [API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs) reference.

## 5. [Data Wrangling & Exploratory Data Analysis EDA](https://github.com/akthammomani/MAXELLA-APP-Movies-Tensorflow-Recommenders-TFRS/tree/main/Notebooks/Data_Wrangling_Exploratory_Data_Analysis_EDA)

The Data wrangling step focuses on collecting our data, organizing it, and making sure it's well defined. For our project we have collected below datasets to have a good foundation so we can build a Deep Learning model with the best performance possible:
 * [movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings)
 * [movie_lens/1m-movies](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-movies)
 * [movies_metadata.csv](https://grouplens.org/datasets/movielens/)
 * [credits.csv](https://grouplens.org/datasets/movielens/)

We have 4 Datasets to support this project as shown above, so we'll focus in below:
 * Convert Tensorflow datasets (1m-ratings and 1m-movies) from Tensorflow (ensorflow_datasets.core.as_dataframe.StyledDataFrame) to pandas DataFrame (pandas.core.frame.DataFrame) for easy data wrangling (Some Pandas method doesn’t work well with StyledDataFrame from TensorFlow)
 * Fix any wrong values in user_zip_code (Any zipcode >5 characters).   
 * Fixing "movie_genres": let's make sure that genres are the format of a list for easy access. Merging and concatenation will be needed.
 * Fixing "user_occupation_label": one category label is missing "10" causing 'K-12 student' & 'college/grad student' to be labeled as "17" so here, we'll assign "10" to 'K-12 student'. 
 * Add 5 more features to the original Dataset: 'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and movie_lens_movie_id --> Will get these features using 2 datasets from Movielens website: movies_metadata.csv & credits.csv
 * Fix existing wrong movie title (or in some cases misspelled).
 * Remove all special characters or letter accents from Movie titles, cast and director.
 * Add movie id which is matching the original movie id in the movie lens original dataset (for some reason the movie id from Tensorflow dataset is not matching).
 * Fix duplicates movie_title with same movie_id.
 * After fixing above items, we converted Pandas DataFrame to Tensorflow dataset:
   * From 'cast' features, let's drop all secondary casting and keep only the star of the movie and let's call the feature "star".
   * Let's make sure to keep only the important columns. 
   * Change the data types of the important features to fit with Tensorflow-Recommender TFRS Library.
   * Keep in mind tfds currently does not support float64 so we'll be using int64 or float32 depends on the data.
   * Wrap the pandas DataFrame into tf.data.Dataset object using tf.data.Dataset.from_tensor_slices (To check other options - [here](https://www.srijan.net/resources/blog/building-a-high-performance-data-pipeline-with-tensorflow#gs.f33srf))
   * Movie Lens dataset transformation after Data Wrangling:
![dataset_head](https://user-images.githubusercontent.com/67468718/141497590-35895da5-f162-4df6-acbd-6f25ae35562d.JPG)
   * We used Folium Library to map zipcode per user_rating to give us more understanding of the users behaviour vs location:
     * Folium Library, due to its impressive capability of interactive visualization. More details about folium and how to install check [here](https://pypi.org/project/folium/)).
     * Let's create Two Dataframes --> One: Avg user rating per Zip Code and Two: Takes user occupants per Zipcodes.
     * For folium to work without error, zipcode needs to be in a string format. 
     * Try to limit the number of zipcodes to avoid any memory errors, in our case we'll be taking only CA Zipcodes which are available in the Movie lens 1M Dataset.
     * Also, we'll be using external databases:
       * To get the coordinate of the zipcodes --> This is needed for to highlight the zipcodes as labels in the map.
       * To get a geoJSON file which has coordinates of zipcode --> This is needed to map the zip codes boundaries.
       ![folium_rating](https://user-images.githubusercontent.com/67468718/141501541-e2ae7e21-118d-4fff-bec4-49c78aafa29d.JPG)
   * We used Abstract Syntax Tree (AST) to help us in list values in the dataset (movie_genres), extract cast and directors from credits.csv by transforming existing list values feature from a list of strings to a list of list because Pandas read the lists (e.g., genres) as a string which is a problem because we cannot even loop the lists to count the genres
     ![ast](https://user-images.githubusercontent.com/67468718/141503160-40b7dd5a-4fe5-42ac-921f-b9ee7f6e849b.JPG)

## 6. [Features Importance Using Deep & Cross Network (DCN-v2)](https://github.com/akthammomani/MAXELLA-APP-Movies-Tensorflow-Recommenders-TFRS/tree/main/Notebooks/TFRS-Features-Importance)
