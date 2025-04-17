import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import matplotlib.pyplot as plt

# Load the data
def load_data():
    ratings_data = pd.read_csv('data/ratings_data.csv')
    return ratings_data

# Train and plot the performance of SVD model over epochs
def train_and_plot_svd(ratings_df, n_epochs=20):
    reader = Reader(rating_scale=(1, 5))
    ratings_data_surprise = Dataset.load_from_df(ratings_df[['User ID', 'Content ID', 'Rating']], reader)
    trainset, testset = train_test_split(ratings_data_surprise, test_size=0.2, random_state=42)
    
    epoch_rmse = []
    
    # Train the model for each epoch and record RMSE
    for epoch in range(1, n_epochs + 1):
        algo = SVD(n_epochs=epoch, random_state=42, n_factors=5, lr_all=0.001)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse_value = rmse(predictions, verbose=False)
        epoch_rmse.append(rmse_value)
        print(f"Epoch {epoch}: RMSE = {rmse_value}")
    
    # Plotting RMSE over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), epoch_rmse, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over Number of Epochs')
    plt.grid(True)
    plt.show()

# Main script
if __name__ == "__main__":
    # Load the ratings data
    ratings_df = load_data()
    
    # Train the model and plot RMSE over epochs
    train_and_plot_svd(ratings_df, n_epochs=1000)
