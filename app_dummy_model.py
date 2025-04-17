import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit_star_rating import st_star_rating

# Dummy recommendation algorithm class with predict function
class DummyAlgo:
    # Dummy algorithm class takes any number of input parameters, but doesn't use them
    def __init__(self, **kwargs):
        pass

    def fit(self, trainset):
        """Simulate training process (no actual training)"""
        pass  # No operation for dummy algorithm

    def predict(self, user_id, content_id):
        """Generate a dummy rating prediction for given user and content IDs"""
        return np.random.uniform(1, 5)  # Random rating between 1 and 5


# Load Data
def load_data():
    user_df = pd.read_csv('user_data_students.csv')       # Load user data
    content_df = pd.read_csv('content_data_students.csv')  # Load content data
    ratings_df = pd.read_csv('ratings_data.csv')           # Load ratings data
    return user_df, content_df, ratings_df

# Generate recommendations
def get_recommendations(user_id, algo, content_df, n=5):
    recommendations = []
    for _, row in content_df.iterrows():
        content_id = row['Content ID']
        title = row['Name']
        predicted_rating = algo.predict(user_id, content_id)
        recommendations.append((title, predicted_rating))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(recommendations, columns=['Name', 'Predicted Rating'])

# Convert numeric rating to stars
def convert_to_stars(rating):
    num_stars = int(round(rating))
    return '⭐' * num_stars

# Load data and initialize model
user_df, content_df, ratings_df = load_data()
num_users = user_df['User ID'].nunique() + 1
num_items = content_df['Content ID'].nunique() + 1
algo = DummyAlgo(num_users=num_users, num_items=num_items)  # Change the algorithm here.
algo.fit(ratings_df)  # Finetune the algorithm on uptodate data.

# Streamlit App
st.set_page_config(page_title="Personal Learning Platform", page_icon="🎓")
st.sidebar.image("./wt-logo-black.svg", use_container_width=True)  # Placeholder image for the logo
st.sidebar.header("Select Your Profile")

# User Selection in Sidebar
user_df['Full Name'] = user_df['First Name'] + " " + user_df['Last Name']
user_names = user_df['Full Name'].tolist()
user_names.insert(0, None)
selected_user_name = st.sidebar.selectbox("Please select your name to proceed:", user_names, index=0)

# Main Title
st.title("WT Personal Learning Platform 🎓🎓")

# Main Content with Tabs
if selected_user_name:
    selected_user_info = user_df[user_df['Full Name'] == selected_user_name].iloc[0]
    user_id = selected_user_info['User ID']
    user_ratings = ratings_df[ratings_df['User ID'] == user_id]

    tab1, tab2, tab3 = st.tabs(["Recommendations", "Rate Content", "Profile Information"])

    with tab1:
        st.write("Based on your profile, here are some learning materials we think you'll love!")

        top_n_recommendations = get_recommendations(user_id, algo, content_df, n=5)

        # Display the top recommendation in a highlighted format
        if not top_n_recommendations.empty:
            top_recommendation = top_n_recommendations.iloc[0]
            st.markdown(
                f"<div style='padding: 20px; border: 3px solid #FFD700; background-color: #FFF8DC; border-radius: 10px; margin-bottom: 20px;'>"
                f"<h2>🌟 {top_recommendation['Name']}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Display the rest of the recommendations with card layout
        for i, row in top_n_recommendations.iloc[1:].iterrows():
            with st.container():
                st.markdown(
                    f"<div style='padding: 10px; border: 1px solid #E0E0E0; background-color: #F9F9F9; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<h4>🔹 {row['Name']}</h4>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Button to retrain the model and recalculate recommendations
        if st.button("Recalculate my recommendations ⭐"):
            algo.fit(ratings_df)  # Retrain the algorithm
            st.success("Model retrained successfully! Generating new recommendations...")
            top_n_recommendations = get_recommendations(user_id, algo, content_df, n=5)

    with tab2:
        st.write("Select a content item to rate:")

        # List of content names
        content_names = content_df['Name'].tolist()
        selected_content_name = st.selectbox("Select content:", content_names)

        if selected_content_name:
            # Retrieve the content ID
            content_id = content_df[content_df['Name'] == selected_content_name]['Content ID'].values[0]

            # Display the star rating widget
            rating = st_star_rating(
                label="Your Rating",
                maxValue=5,
                defaultValue=0,
                key=f"rating_{content_id}"
            )

            if st.button("Submit Rating"):
                if rating > 0:
                    # Save the rating to the ratings_df DataFrame
                    new_rating = pd.DataFrame({
                        'User ID': [user_id],
                        'Content ID': [content_id],
                        'Rating': [rating]
                    })
                    ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
                    st.success(f"Thank you! You've rated '{selected_content_name}' with {rating} stars.")
                else:
                    st.warning("Please select a rating before submitting.")

    with tab3:
        st.markdown(
            f"<div style='padding: 20px; border: 1px solid #E0E0E0; background-color: #F9F9F9; border-radius: 10px; margin-bottom: 20px;'>"
            f"<h3>{selected_user_info['Full Name']}</h3>"
            f"<p><strong>Background:</strong> {selected_user_info['Study Background']}</p>"
            f"<p><strong>Age:</strong> {selected_user_info['Age']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("Your Ratings History"):
            if not user_ratings.empty:
                st.write("Here are the ratings you've given to learning materials:")
                user_ratings = ratings_df[ratings_df['User ID'] == user_id]
                user_ratings = user_ratings.merge(content_df, on='Content ID')
                user_ratings['Rating (Stars)'] = user_ratings['Rating'].apply(convert_to_stars)
                st.dataframe(user_ratings[['Name', 'Rating', 'Rating (Stars)']])
            else:
                st.write("You haven't rated any content yet.")