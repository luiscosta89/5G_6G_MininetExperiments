import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

cluster_stations = {}

# Assuming data is read from the CSV file containing the station positions
data = pd.read_csv('data_dst.csv')

# Aggregate the positions by station (taking the mean position as an example)
aggregated_data = data.groupby('station_name').mean().reset_index()

# Prepare the features and labels
X_train = aggregated_data[['x', 'y', 'z']].values  # Input features
station_names = aggregated_data['station_name'].values  # Station names (used later for labeling)

# Assuming that we split the data for training and testing (can be adjusted if needed)
X_test = X_train.copy()

# Train the XGBoost model to predict the final position
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 1,
    'subsample': 1,
}

dtrain = xgb.DMatrix(X_train, label=X_train)
dtest = xgb.DMatrix(X_test)

model = xgb.train(params, dtrain, num_boost_round=100)

# Predict final positions for the test data
preds = model.predict(dtest)

# Create dictionaries for actual and predicted positions
actual_positions = {station: position for station, position in zip(station_names, X_test)}
predicted_positions = {station: position for station, position in zip(station_names, preds)}

def compare_results():
    # Print the comparison of actual and predicted positions
    print("Comparison of Actual vs Predicted Positions:")
    for station in station_names:
        actual = actual_positions[station]
        predicted = predicted_positions[station]
        print(f"Station: {station}")
        print(f"  Actual Position: {actual}")
        print(f"  Predicted Position: [{predicted[0]:.8f}, {predicted[1]:.8f}, {predicted[2]:.8f}]")
        print("-" * 50)

def plot_results():
    # Plot actual vs predicted positions (final positions only)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Actual positions
    actual_x, actual_y, actual_z = X_test[:, 0], X_test[:, 1], X_test[:, 2]
    ax.scatter(actual_x, actual_y, actual_z, color='blue', label='Actual Positions')

    # Predicted positions
    predicted_x, predicted_y, predicted_z = preds[:, 0], preds[:, 1], preds[:, 2]
    ax.scatter(predicted_x, predicted_y, predicted_z, color='red', label='Predicted Positions', marker='^')

    # Label each point with the station name
    for i, station in enumerate(station_names):
        ax.text(actual_x[i], actual_y[i], actual_z[i], station, color='blue')
        ax.text(predicted_x[i], predicted_y[i], predicted_z[i], station, color='red')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Station Final Positions: Actual vs Predicted')
    ax.legend()

    plt.show()

# Step 1: Convert predicted_positions dict to a list of positions
positions_array = np.array(list(predicted_positions.values()))

# Step 2: Calculate the optimal number of clusters using the Elbow Method
def find_optimal_k(positions_array):
    sse = []
    k_values = range(1, 10)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(positions_array)
        sse.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center
    
    # Use KneeLocator to find the elbow point
    kn = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    
    # Plot the Elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return optimal_k

# Step 3: Perform KMeans clustering using the optimal k found
optimal_k = find_optimal_k(positions_array)
if optimal_k is None:
    optimal_k = 3  # Fallback to 3 if no elbow found

print(f"Optimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=0)
cluster_labels = kmeans.fit_predict(positions_array)

# Step 4: Plot the clusters and show which station belongs to which cluster
def plot_clusters(predicted_positions, cluster_labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for clusters
    colors = plt.cm.get_cmap("Set1", optimal_k)

    # Plot each station with its cluster
    for i, (station, position) in enumerate(predicted_positions.items()):
        ax.scatter(position[0], position[1], position[2], color=colors(cluster_labels[i]), s=100, label=station)
        ax.text(position[0], position[1], position[2], station, fontsize=9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Cluster Plot of {len(predicted_positions)} Stations')
    plt.show()

plot_clusters(predicted_positions, cluster_labels)

# Step 5: Print which stations belong to which clusters
cluster_station_mapping = {i: [] for i in range(optimal_k)}
for i, label in enumerate(cluster_labels):
    station_name = list(predicted_positions.keys())[i]
    cluster_station_mapping[label].append(station_name)

for cluster, stations in cluster_station_mapping.items():
    print(f"Cluster {cluster + 1}: {', '.join(stations)}")
    new_data = [(cluster + 1, stations)]
    cluster_stations.update(new_data)

# Step 6: Save cluster info into a JSON file
with open('clusters_dst.json', 'w') as json_file:
    json.dump(cluster_stations, json_file)

#print(cluster_stations)
