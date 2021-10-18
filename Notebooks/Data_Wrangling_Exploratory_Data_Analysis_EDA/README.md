
# MAXELLA APP - TFRS 
# Data Wrangling & Exploratory Data Analysis EDA

<p align="center">
  <img width="700" height="500" src="https://user-images.githubusercontent.com/67468718/137802953-e8be7ade-21d5-4861-b9af-d657e6d2bd4e.JPG">
</p>

## 1. Introduction: 

**The Data wrangling step** focuses on collecting or converting our data, organizing it, and making sure it's well defined. For our project we will be using  ***movie_lens/1m dataset*** from TensorFlow  because it's a unique dataset with plenty of Metadata that's needed for this project, so let' focus in below items:

 * Clean NANs (If any), duplicate values (If any), wrong values and removing insignificant columns.
 * Remove any special characters.
 * Renaming Column labels.
 * Correct datatypes.
 
In the other hand, **Exploratory Data Analysis EDA Step**, let's focus in the following: 
 
 * To get familiar with the features in our dataset.
 * Generally understand the core characteristics of our cleaned dataset.
 * Explore the data relationships of all the features and understand how the features compare to the response variable.
 * Let's be creative and think about interesting figures and all the plots that can be created to help deepen our understanding of the data.

## 2. Dataset: 

**Movie Lens** contains a set of movie ratings from the MovieLens website, a movie recommendation service. This dataset was collected and maintained by [GroupLens](https://grouplens.org/) , a research group at the University of Minnesota. There are 5 versions included: "25m", "latest-small", "100k", "1m", "20m". In all datasets, the movies data and ratings data are joined on "movieId". The 25m dataset, latest-small dataset, and 20m dataset contain only movie data and rating data. The 1m dataset and 100k dataset contain demographic data in addition to movie and rating data.

**movie_lens/1m** can be treated in two ways:

  * It can be interpreted as expressesing which movies the users watched (and rated), and which they did not. This is a form of *implicit feedback*, where users' watches tell us which things they prefer to see and which they'd rather not see (This means that every movie a user watched is a positive example, and every movie they have not seen is an implicit negative example).
  * It can also be seen as expressesing how much the users liked the movies they did watch. This is a form of *explicit feedback*: given that a user watched a movie, we can tell roughly how much they liked by looking at the rating they have given.



**[movie_lens/1m-ratings](https://www.tensorflow.org/datasets/catalog/movie_lens#movie_lens1m-ratings):**
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
   * 0: Action
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
**Example:**

|bucketized_user_age	|movie_genres|	movie_id|	movie_title|	raw_user_age|	timestamp|	user_gender|	user_id	|user_occupation_label|	user_occupation_text	|user_rating	|user_zip_code|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|45.0	|7 (Drama)|b'357'	|b"One Flew Over the Cuckoo's Nest (1975)"	|46.0	|879024327	|True	|b'138'	|4 (doctor/health care)	|b'doctor'	|4.0|	b'53211'|


## 3. Data Wrnagling Objectives:

 * Let's change user_gender from boolian Female or Male: True --> Male, False --> Female
 * Let's remove the symbols: (b), (') and (").
 * Let's drop columns: user_occupation_label and movie_genres.
 * Let's change "timestamp" which is in the unix epoch (units of seconds) to datetime64[ns]
 * Let's fix any wrong values in user_zip_code (Any zipcode >5 characters)


## 4. Exploratory Data Analysis (EDA)

 * To get familiar with the features in our dataset.
 * Generally understand the core characteristics of our cleaned dataset.
 * Explore the data relationships of all the features and understand how the features compare to the response variable.
 * Let's be creative and think about interesting figures and all the plots that can be created to help deepen our understanding of the data.
 * Let's create one feature that give us the year when the movie was released will call it "movie_release_year".

**Alright, let's start the fun part, let's extract insights from the dataset by asking very useful questions:**

**(1) What is the preferable month of the year to rate/watch movies?**

As shown below, looks like Summer & Holidays Months are the highest, which make sense!!!

![movies_month](https://user-images.githubusercontent.com/67468718/137811621-77527920-5995-49fa-8027-964726e277f9.JPG)
