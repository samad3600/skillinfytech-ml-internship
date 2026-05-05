import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Drop null
movies.dropna(inplace=True)

# Convert to string
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Simple tags
movies['tags'] = movies['overview']

# Convert list to string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Save
pickle.dump(movies, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

print("✅ Model ready!")