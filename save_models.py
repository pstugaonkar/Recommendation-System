import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

# Load your dataset
data = pd.read_csv('LargeMovieDataset.csv')  # Replace with the correct dataset path

# Preprocess the data (this should be similar to what you did before)
encoder = OneHotEncoder()
genre_encoded = encoder.fit_transform(data[['Genre']]).toarray()

scaler = StandardScaler()
processed_data = data[['Rating', 'Popularity']]  # Include other features if necessary
scaled_data = scaler.fit_transform(processed_data)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.predict(scaled_data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Save the models and preprocessing objects
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(pca, 'pca_model.pkl')

print("Models and preprocessing steps saved successfully!")
