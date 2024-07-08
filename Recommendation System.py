import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('rating.csv')

# Check for any negative or unrealistic values in the ratings
ratings = ratings[(ratings['rating'] >= 0) & (ratings['rating'] <= 5)]

# Merge the datasets on movieId
data = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0 (or use other imputation techniques)
user_item_matrix.fillna(0, inplace=True)

# Check the size of the user-item matrix
print("User-item matrix shape:", user_item_matrix.shape)
# Calculate cosine similarity between users
user_similarity = cosine_similarity ( user_item_matrix )

# Create a DataFrame for user similarity
user_similarity_df = pd.DataFrame ( user_similarity , index = user_item_matrix.index ,
                                    columns = user_item_matrix.index )


def get_recommendations(user_id , num_recommendations=5):
    # Get the similarity scores for the given user
    user_similarity_scores = user_similarity_df[user_id]

    # Get the ratings provided by the given user
    user_ratings = user_item_matrix.loc[user_id]

    # Calculate the weighted average of ratings from similar users
    weighted_ratings = user_item_matrix.T.dot ( user_similarity_scores )

    # Normalize the ratings by the sum of similarities
    sum_of_similarities = user_similarity_scores.sum ( )
    if sum_of_similarities == 0:
        sum_of_similarities = 1  # Avoid division by zero
    weighted_average_ratings = weighted_ratings / sum_of_similarities

    # Create a DataFrame for recommendations
    recommendations = pd.DataFrame ( weighted_average_ratings , columns = ['weighted_average_rating'] )

    # Exclude movies already rated by the user
    recommendations = recommendations[~recommendations.index.isin ( user_ratings[user_ratings > 0].index )]

    # Sort by weighted average rating
    recommendations = recommendations.sort_values ( 'weighted_average_rating' , ascending = False )

    # Return the top N recommendations
    return recommendations.head ( num_recommendations )


# Get recommendations for a specific user
user_id = int(input("Enter your userID (1-91) : "))
recommendations = get_recommendations ( user_id )
print ( f"Recommendations for user {user_id}:\n" , recommendations )
