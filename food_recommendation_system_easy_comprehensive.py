import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Set the background image using CSS
image_url = "https://cdn.vox-cdn.com/uploads/chorus_image/image/73039055/Valle_KimberlyMotos__1_of_47__websize__1_.0.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 style="font-size:35px;">Most Popular Restaurants Based on Reviews and Ratings</h1>', unsafe_allow_html=True)

# Select box for state location
place = st.selectbox("Select the state location:", 
                     ['KL', 'Ipoh', 'JB', 'Kuching', 'Langkawi', 'Melaka', 'Miri', 'Penang', 'Petaling Jaya', 'Shah Alam'])

# Import data
google_review_data = pd.read_csv('GoogleReview_data_cleaned.csv')
tripadvisor_data = pd.read_csv('TripAdvisor_data_cleaned.csv')

# Data Preprocessing
google_review_data.dropna(axis=0, how='any', inplace=True)
tripadvisor_data.dropna(axis=0, how='any', inplace=True)
google_review_data.drop_duplicates(inplace=True, keep=False)
tripadvisor_data.drop_duplicates(inplace=True, keep=False)

if 'Number of Reviews' not in google_review_data.columns:
    google_review_data['Number of Reviews'] = google_review_data['Review'].apply(lambda x: len(x.split()))  # Example assumption
if 'Number of Reviews' not in tripadvisor_data.columns:
    tripadvisor_data['Number of Reviews'] = tripadvisor_data['Review'].apply(lambda x: len(x.split()))  # Example assumption

combined_data = pd.merge(google_review_data, tripadvisor_data, on=['Restaurant', 'Location'], how='inner')
combined_data = combined_data.drop_duplicates(subset=['Restaurant'], keep='first')
combined_data['Combined Rating'] = (combined_data['Rating_x'] + combined_data['Rating_y']) / 2
combined_data['Total Reviews'] = combined_data['Number of Reviews_x'] + combined_data['Number of Reviews_y']

place_df = combined_data[combined_data['Location'].str.lower().str.contains(place.lower())]
sorted_data = place_df.sort_values(by=['Total Reviews', 'Combined Rating'], ascending=[False, False])
sorted_data.reset_index(drop=True, inplace=True)

popular_restaurants = sorted_data[['Restaurant', 'Location', 'Total Reviews', 'Combined Rating']].head(10)
popular_restaurants = popular_restaurants.rename(columns={
    'Restaurant': 'Name',
    'Total Reviews': 'Number of Reviews',
    'Combined Rating': 'Average Rating'
})

st.dataframe(popular_restaurants.style.format({
    'Number of Reviews': '{:.0f}',
    'Average Rating': '{:.1f}'
}))

# Food Recommendation System
df = pd.read_csv('/content/1662574418893344.csv')

# Clean text data
def text_cleaning(text):
    return "".join([char for char in text if char not in string.punctuation])

df['Describe'] = df['Describe'].apply(text_cleaning)

# Content Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Describe'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    food_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[food_indices]

# Advanced Content Based Filtering
def create_soup(x):
    return x['C_Type'] + " " + x['Veg_Non'] + " " + x['Describe']

df['soup'] = df.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Collaborative Filtering
rating = pd.read_csv('/content/ratings.csv')
rating = rating[:511]  # Removing last row with no values

food_rating = rating.groupby(by='Food_ID').count()['Rating'].reset_index().rename(columns={'Rating': 'Rating_count'})
user_rating = rating.groupby(by='User_ID').count()['Rating'].reset_index().rename(columns={'Rating': 'Rating_count'})
rating_matrix = rating.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)
csr_rating_matrix = csr_matrix(rating_matrix.values)

recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)

def Get_Recommendations(title):
    user = df[df['Name'] == title]
    user_index = np.where(rating_matrix.index == int(user['Food_ID']))[0][0]
    user_ratings = rating_matrix.iloc[user_index]
    reshaped = user_ratings.values.reshape(1, -1)
    distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)
    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})
    result = pd.merge(nearest_neighbors, df, on='Food_ID', how='left')
    return result.head()

# You can include a Streamlit input widget to get a title from the user and display recommendations
title = st.text_input("Enter a food name for recommendations:")
if title:
    recommendations = Get_Recommendations(title)
    st.dataframe(recommendations)
