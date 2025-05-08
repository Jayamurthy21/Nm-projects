import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge ratings with movie genres
data = pd.merge(ratings, movies, on='movie_id')

# Get user ID from name
def get_user_id(name):
    row = users[users['name'].str.lower() == name.lower()]
    return row.iloc[0]['user_id'] if not row.empty else None

# Build user profile from ratings
def build_user_profile(user_id):
    user_data = data[data['user_id'] == user_id]
    if user_data.empty:
        print(f"No ratings found for user ID {user_id}")
        return pd.Series()
    genre_matrix = user_data['genres'].str.get_dummies(sep='|')
    weights = user_data['rating'].values.reshape(-1, 1)
    profile = genre_matrix.mul(weights.flatten(), axis=0).sum()
    return profile

# Recommend movies for a matched pair
def recommend_for_match(user1_id, user2_id, top_n=3):
    profile1 = build_user_profile(user1_id)
    profile2 = build_user_profile(user2_id)
    
    if profile1.empty or profile2.empty:
        return pd.DataFrame()
    
    combined_profile = (profile1 + profile2) / 2
    movie_genres = movies['genres'].str.get_dummies(sep='|')
    
    similarity_scores = cosine_similarity([combined_profile], movie_genres)[0]
    movies['similarity'] = similarity_scores
    
    already_watched = ratings[(ratings['user_id'] == user1_id) | (ratings['user_id'] == user2_id)]['movie_id']
    recommended = movies[~movies['movie_id'].isin(already_watched)].sort_values(by='similarity', ascending=False)
    
    return recommended[['title', 'genres', 'similarity']].head(top_n)

# Main program
if __name__ == "__main__":
    name1 = input("Enter name of first user: ")
    name2 = input("Enter name of second user: ")
    
    id1 = get_user_id(name1)
    id2 = get_user_id(name2)
    
    if id1 is None or id2 is None:
        print("One or both names not found in user data.")
    else:
        recommendations = recommend_for_match(id1, id2)
        if recommendations.empty:
            print("No movie recommendations available.")
        else:
            print(f"\nTop Movie Recommendations for {name1} and {name2}:")
            print(recommendations.to_string(index=False))
