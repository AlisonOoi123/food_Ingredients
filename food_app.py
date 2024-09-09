import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the HTML template for the front end with custom styles
html_temp = """
    <style>
    body {
        background-color: #F6F6F6;  /* Set the background color to gold */
        font-family: Arial, sans-serif;  /* Set the font family */
    }
    h1 {
        color: #1E56A0;  /* Set the h1 (title) color to dark gray */
    }
    </style>
    <div style="background-color:#F6F6F6;padding:13px;text-align:center;">
    <h1 style="color: #163172;">Food Recommendation System</h1>
    </div>
"""

# Set the page configuration
st.set_page_config(
    page_title="Food Recommendation System",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon=None,
)

# Display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)

# Add a picture of food
# Center-align the image using st.image
st.image("./AI061118-16_xlarge.jpg", use_column_width=True)


# Add a caption
st.caption('by Valeria Filippou')

user_input = st.text_input("Enter ingredients separated by commas:")

df = pd.read_csv('./Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

df = df[['Title', 'Cleaned_Ingredients']]

def recommend_dishes(data, user_input):
  # Preprocess user input
  user_input = user_input.lower()

  # Calculate the number of matching ingredients
  vectorizer = CountVectorizer()
  ingredients_matrix = vectorizer.fit_transform(data['Cleaned_Ingredients'])

  user_vector = vectorizer.transform([user_input])

  similarities = cosine_similarity(user_vector, ingredients_matrix)

  # Find dishes with at least `threshold` matching ingredients
  matching_dishes = [(index, row) for index, row in enumerate(similarities[0]) if row >= 0.3]

  recommended_dishes = data.iloc[[index for index, _ in matching_dishes]]

  return recommended_dishes[['Title', 'Cleaned_Ingredients']]


if st.button("Recommend"):
  if user_input:
    recommended_dishes = recommend_dishes(df, user_input)
    st.subheader("Recommended Dishes:")

    if not recommended_dishes.empty:
      # Create a dictionary to store whether the ingredients expander is open for each dish
      expanders_open = {}
    
      for idx, row in recommended_dishes.iterrows():
        title = row['Title']
        cleaned_ingredients = row['Cleaned_Ingredients']
    
        # Create an expander for each dish
        with st.expander(f"{title}", expanded=expanders_open.get(title, False)):
    
          # st.markdown(cleaned_ingredients)
          # Split the ingredients string at the comma
          ingredients_list = [ingredient.lstrip("'") for ingredient in cleaned_ingredients.split("', ")]
    
          # Remove "for serving" from each ingredient
          ingredients_list = [ingredient.replace('for serving', '') for ingredient in ingredients_list]
    
    
          # Check if the first ingredient starts with "[" and remove it
          if ingredients_list[0].startswith("['"):
            ingredients_list[0] = ingredients_list[0][2:]
    
          # Check if the last ingredient ends with ']'
          if ingredients_list[-1].endswith("']"):
            ingredients_list[-1] = ingredients_list[-1][:-2]
    
    
          st.markdown('\n'.join([f"- {ingredient}" for ingredient in ingredients_list]))
    else:
      st.write("No recommended dishes found. Please try a different combination of ingredients.")
              

  else:
    st.warning("Please enter ingredients to get recommendations.")

st.sidebar.header("About This App")
st.sidebar.info("Welcome to the Food Recommendation System! This web app suggests dishes based on the ingredients you provide.")
st.sidebar.info("The more ingredients you specify, the more accurate the recommendations will be.")
st.sidebar.info("To get your recommended dishes, simply enter the ingredients you'd like to use and click on the 'Recommend' button.")
st.sidebar.info("Example 1: fish, oil, potato, yoghurt, salt, pepper, zucchini")
st.sidebar.info("Example 2: chicken, salt, rice, tomato, lettuce, pepper, cucumber")
st.sidebar.info("Be creative!")
st.sidebar.info("We hope you find a delicious dish to enjoy. Don't forget to rate this app!")

feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Thank you for rating the app!")
