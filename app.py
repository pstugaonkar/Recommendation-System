from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved models and preprocessing objects
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
pca = joblib.load('pca_model.pkl')

# Load the movie data (ensure it matches the data you used for training)
data = pd.read_csv('LargeMovieDataset.csv')  # Replace with your actual data path

# Preprocess the data: Apply KMeans prediction to the scaled data
processed_data = data[['Rating', 'Popularity']]  # Include other features if necessary
scaled_data = scaler.transform(processed_data)
data['Cluster'] = kmeans.predict(scaled_data)  # Ensure 'Cluster' column exists

# Recommendation function
def recommend_movies(movie_name, data, num_recommendations=5):
    # Find the cluster of the given movie
    movie_cluster = data.loc[data['Title'] == movie_name, 'Cluster'].values[0]
    
    # Get movies from the same cluster
    similar_movies = data[data['Cluster'] == movie_cluster]['Title']
    
    # Exclude the input movie from recommendations
    recommendations = similar_movies[similar_movies != movie_name].head(num_recommendations)
    
    return recommendations.tolist()

@app.route('/')
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    movie_name = request.args.get('movie_name', default=None, type=str)
    if not movie_name:
        return jsonify({"error": "Movie name is required!"}), 400
    recommendations = recommend_movies(movie_name, data)
    if not recommendations:
        return jsonify({"error": "Movie not found!"}), 404
    return jsonify({"movie_name": movie_name, "recommendations": recommendations})

@app.route('/predict', methods=['POST'])
def predict_cluster():
    data_input = request.json
    genre = data_input['Genre']
    rating = data_input['Rating']
    popularity = data_input['Popularity']
    
    genre_encoded = encoder.transform([[genre]]).toarray()
    genre_columns = encoder.get_feature_names_out(['Genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)
    
    input_data = pd.DataFrame([[rating, popularity]], columns=['Rating', 'Popularity'])
    processed_data = pd.concat([genre_df, input_data], axis=1)
    
    scaled_input = scaler.transform(processed_data)
    cluster = kmeans.predict(scaled_input)
    
    return jsonify({"Predicted Cluster": int(cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)