
# TFRS Features Engineering

### 1. Introduction

In this notebook, we'll try to improve our cleaned Dataframe which was orginally from Tensorflow Dataset: [movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings), so our focus will be mainly in: 

 * Fixing **"movie_genres"**: let's make sure that genres are the format of a list for easy access.
 * Fixing **"user_occupation_label"**: one category label is missing "10" causing 'K-12 student' & 'college/grad student' to be labled as "17" so here, we'll assign "10" to 'K-12 student'.
 * Add 5 more features to the original Dataset: **'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and movie_lens_movie_id** --> Will get these features using 2 datasets from [Movielens website](https://grouplens.org/datasets/movielens/):
   * **movies_metadata.csv**
   * **credits.csv**

 * Fix existing wrong movie title (or in some cases misspelled).
 * Let's remove all special characters or letter accents from Movie titles, cast and director.
 * Add movie id which is matching the orginal movie id in the movie lens original dataset (for some reason the movie id from tensorflow dataset is not matching).
 * Fix duplicates movie_title with same movie_id.
 * After fixing above items, let's convert Pandas dataframe to tensforflow dataset.
