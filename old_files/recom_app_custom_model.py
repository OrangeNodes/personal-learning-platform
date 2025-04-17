import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from streamlit_star_rating import st_star_rating

# Dummy recommendation algorithm class with predict function
class DummyAlgo:
    def fit(self, trainset):
        """Simulate training process (no actual training)"""
        pass  # No operation for dummy algorithm

    def predict(self, user_id, content_id):
        """Generate a dummy rating prediction for given user and content IDs"""
        return np.random.uniform(1, 5)  # Random rating between 1 and 5

# Add a new class here with a better algorithm.
import numpy as np
import matplotlib.pyplot as plt

class MatrixFactorization:
    def __init__(self, num_users, num_items, latent_dim=10, learning_rate=0.01, regularization=0.01, randomness_scale=0.1):
        """
        Initialize the Matrix Factorization model.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items/content.
            latent_dim (int): Number of latent factors (dimensionality of embeddings).
            learning_rate (float): Step size for gradient updates.
            regularization (float): Coefficient for L2 regularization to prevent overfitting.
            randomness_scale (float): Scale of random initialization for embeddings.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.randomness_scale = randomness_scale

        # Try to load the model, if not found, initialize new embeddings
        try:
            print("Loading existing model...")
            self.load_model('matrix_factorization_model.npz')
        except:
            print("Model not found. Initializing new embeddings...")
            self.user_factors = np.random.normal(scale=self.randomness_scale, size=(num_users, latent_dim))
            self.item_factors = np.random.normal(scale=self.randomness_scale, size=(num_items, latent_dim))

        # To track training and test losses over epochs
        self.train_losses = []
        self.test_losses = []
        self.train_mae = []  # To track training MAE
        self.test_mae = []   # To track test MAE

    def fit(self, train_data):
        """
        Train the model using stochastic gradient descent.

        Args:
            train_data (DataFrame): Training dataset with columns ['User ID', 'Content ID', 'Rating'].
        """
        self.train(train_data=train_data, test_data=None, epochs=10)

    def train(self, train_data, test_data, epochs):
        """
        Train the model using stochastic gradient descent.

        Args:
            train_data (DataFrame): Training dataset with columns ['User ID', 'Content ID', 'Rating'].
            test_data (DataFrame): Test dataset with the same structure as train_data.
            epochs (int): Number of training epochs.
        """
        # Track which user IDs and item IDs were seen during training
        self.seen_user_ids = set(train_data['User ID'])
        self.seen_item_ids = set(train_data['Content ID'])

        for epoch in range(epochs):
            train_loss = 0
            train_abs_error = 0  # Accumulate absolute errors for MAE

            # Iterate over each user-item-rating triplet in the training data
            for user, item, rating in zip(train_data['User ID'], train_data['Content ID'], train_data['Rating']):
                # Predict the rating as the dot product of user and item embeddings
                pred = np.dot(self.user_factors[user], self.item_factors[item])
                error = rating - pred  # Calculate prediction error

                # Update user and item embeddings using gradient descent with regularization
                self.user_factors[user] += self.learning_rate * (
                    error * self.item_factors[item] - self.regularization * self.user_factors[user]
                )
                self.item_factors[item] += self.learning_rate * (
                    error * self.user_factors[user] - self.regularization * self.item_factors[item]
                )

                # Accumulate squared error and absolute error
                train_loss += error**2 + self.regularization * (
                    np.sum(self.user_factors[user]**2) + np.sum(self.item_factors[item]**2)
                )
                train_abs_error += abs(error)

            # Compute average training loss and MAE for the epoch
            train_loss /= len(train_data)
            train_abs_error /= len(train_data)
            self.train_losses.append(train_loss)
            self.train_mae.append(train_abs_error)

            if test_data is None:
                # Print progress
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_abs_error:.4f}")
                continue
            else:
                # Evaluate the model on the test data
                test_loss = 0
                test_abs_error = 0  # Accumulate absolute errors for MAE
                for user, item, rating in zip(test_data['User ID'], test_data['Content ID'], test_data['Rating']):
                    pred = np.dot(self.user_factors[user], self.item_factors[item])
                    error = rating - pred
                    test_loss += error**2
                    test_abs_error += abs(error)

                # Compute average test loss and MAE for the epoch
                test_loss /= len(test_data)
                test_abs_error /= len(test_data)
                self.test_losses.append(test_loss)
                self.test_mae.append(test_abs_error)

                # Print progress
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train MAE: {train_abs_error:.4f}, Test MAE: {test_abs_error:.4f}")


    def export_model(self, file_path):
        """
        Export the trained user and item embeddings to a file.

        Args:
            file_path (str): Path to save the model (e.g., 'model.npz').
        """
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'latent_dim': self.latent_dim
        }
        np.savez(file_path, **model_data)
        print(f"Model exported to {file_path}")

    def load_model(self, file_path):
        """
        Load user and item embeddings from a saved file.

        Args:
            file_path (str): Path to the saved model file (e.g., 'model.npz').
        """
        model_data = np.load(file_path)
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.latent_dim = model_data['latent_dim']
        print(f"Model loaded from {file_path}")

    def predict(self, user_id, content_id):
        """
        Predict the rating for a specific user and content.

        Args:
            user_id (int): ID of the user.
            content_id (int): ID of the content.

        Returns:
            float: Predicted rating.

        Raises:
            ValueError: If the user or content was not seen during training.
        """
        # user_id = user_id + 1  # Adjust for 0-based indexing
        # content_id = content_id + 1  # Adjust for 0-based indexing
        if user_id not in self.seen_user_ids:
            raise ValueError(f"User ID {user_id} was not seen during training.")
        if content_id not in self.seen_item_ids:
            raise ValueError(f"Content ID {content_id} was not seen during training.")

        # Compute the predicted rating as the dot product of the user and item embeddings
        pred = np.dot(self.user_factors[user_id], self.item_factors[content_id])
        return pred

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
algo = MatrixFactorization(num_users=num_users, num_items=num_items)  # Change the algorithm here.
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