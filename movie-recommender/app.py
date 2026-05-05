import streamlit as st
import pickle
import requests

# Load data
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# TMDB API
API_KEY = "YOUR_API_KEY_HERE"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else None

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return recommended, posters

# UI
st.set_page_config(page_title="🎬 CineAI", layout="wide")

st.markdown("# 🎬 CineAI – Smart Movie Recommender")
st.write("### Discover movies you'll love instantly")

selected_movie = st.selectbox("🎥 Choose a movie", movies['title'].values)

if st.button("🚀 Recommend"):
    names, posters = recommend(selected_movie)

    st.subheader("✨ Recommended for you")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i])
            st.write(names[i])