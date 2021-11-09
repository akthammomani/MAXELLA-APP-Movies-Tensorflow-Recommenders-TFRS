
# TFRS Features Engineering

In this notebook, we'll try to improve our cleaned Dataframe which was orginally from Tensorflow Dataset: [movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings), so our focus will be mainly in: 

 * Fixing **"movie_genres"**: let's make sure that genres are the format of a list for easy access.
 * Fixing **"user_occupation_label"**: one category label is missing "10" causing 'K-12 student' & 'college/grad student' to be labled as "17" so here, we'll assign "10" to 'K-12 student'.
 * Add 5 more features to the original Dataset: **'cast', 'director', 'cast_size', 'crew_size', 'imdb_id', 'release_date' and movie_lens_movie_id** --> Will get these features using 2 datasets from [Movielens website](https://grouplens.org/datasets/movielens/):
   * **movies_metadata.csv**
   * **credits.csv**
   
      **1. Crew:** From the crew, we will only pick the director as our feature since the others don't contribute that much to the feel of the movie.
      
      **2. Cast:** Choosing Cast is a little more tricky. Lesser known actors and minor roles do not really affect people's opinion of a movie. Therefore, we must only select the major characters and their respective actors. Arbitrarily we will choose the top 4 actors that appear in the credits list.

 * Fix existing wrong movie title (or in some cases misspelled).
 * Let's remove all special characters or letter accents from Movie titles, cast and director.
 * Add movie id which is matching the orginal movie id in the movie lens original dataset (for some reason the movie id from tensorflow dataset is not matching).
 * Fix duplicates movie_title with same movie_id.
 * After fixing above items, let's convert Pandas dataframe to tensforflow dataset:
 
   * From 'cast' features, let's drop all secondary casting and keep only the star of the movie and let's call the feature "star" 
   * Let's make sure to keep only the important columns. 
   * Change the data types of the important features to fit with Tensorflow-Recommender TFRS Library.
   * Keep in mind **tfds** currently does not support **float64** so we'll be using **int64 or  float32** depends on the data.
   * We'll wrap the **pandas dataframe** into **tf.data.Dataset** object using **tf.data.Dataset.from_tensor_slices** (To check other options - [here](https://www.srijan.net/resources/blog/building-a-high-performance-data-pipeline-with-tensorflow#gs.f33srf))
