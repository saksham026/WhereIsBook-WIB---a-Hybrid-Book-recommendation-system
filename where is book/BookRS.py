import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation


#Read userid-songid-listen_count triplets
#This step might take time to download data from external sources
triplets_file = 'ratings_cleaned.csv'
books_metadata_file = 'books.csv'

book_df_1 = pandas.read_csv(triplets_file)
#song_df_1.columns = ['user_id', 'book_id', 'rating']

#Read song  metadata
book_df_2 =  pandas.read_csv(books_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
book_df = pandas.merge(book_df_1, book_df_2.drop_duplicates(['book_id']), on="book_id", how="left") 



book_df.head()


len(book_df)


book_df = book_df.head(10000)

#Merge song title and artist_name columns to make a merged column
book_df['book'] = book_df['title'].map(str) + " - " + book_df['authors']




book_grouped = book_df.groupby(['book']).agg({'rating': 'count'}).reset_index()
grouped_sum = book_grouped['rating'].sum()
book_grouped['percentage']  = book_grouped['rating'].div(grouped_sum)*100
book_grouped.sort_values(['rating', 'book'], ascending = [0,1])



users = book_df['user_id'].unique()

len(users)

###Fill in the code here
books = book_df['book'].unique()
len(books)


train_data, test_data = train_test_split(book_df, test_size = 0.20, random_state=0)
print(train_data.head(5))



pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'book')


user_id = users[5]
pm.recommend(user_id)


###Fill in the code here
user_id = users[8]
pm.recommend(user_id)


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'book')


#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)


user_id = users[7]
#Fill in the code here
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
pred=is_model.recommend(user_id)
print(pred)

print("---------------------$$$$$$$$$$$$$$$$$$$$$********************")
is_model.get_similar_items(['The Hobbit - J.R.R. Tolkien'])



