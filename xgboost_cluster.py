import pandas as pd
import numpy as np
from xgboost import DMatrix, train
from sklearn.cluster import KMeans
from kneed import KneeLocator
import json
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load the CSV data
# -----------------------------
csv_file = "mobile_positions.csv"  # Replace with your CSV path
data = pd.read_csv(csv_file)

# Assume CSV columns: Station_name, x, y, z
station_names = data['Station_name'].unique()
X_train = data[['x', 'y', 'z']].values

# -----------------------------
# Step 2: Train XGBoost model
# -----------------------------
dtrain = DMatrix(X_train, label=X_train)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 1,
    'subsample': 1
}

model = train(params, dtrain, num_boost_round=100)

# -----------------------------
# Step 3: Predict final positions
# -----------------------------
dtest = DMatrix(X_train)
preds = model.predict(dtest)

# Aggregate predictions to get one final position per station
predicted_positions = {}
for station in station_names:
    # Get all predicted rows corresponding to this station
    station_indices = data.index[data['Station_name'] == station].tolist()
    station_preds = preds[station_indices]
    # Average to get a single final predicted position
    final_pos = np.mean(station_preds, axis=0)
    predicted_positions[station] = final_pos

# Convert to array for clustering
positions_array = np.array(list(predicted_positions.values()))

# -----------------------------
# Step 4: Find optimal number of clusters using Knee Point
# -----------------------------
def find_optimal_k(positions_array):
    sse = []
    k_values = range(1, min(10, len(positions_array)+1))  # avoid k > n
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(positions_array)
        sse.append(kmeans.inertia_)
    kn = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    # Plot the elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method / Knee Point for Optimal k')
    plt.grid(True)
    plt.show()
    if optimal_k is None:
        optimal_k = 3
    return optimal_k

optimal_k = find_optimal_k(positions_array)
print(f"Optimal number of clusters: {optimal_k}")

# -----------------------------
# Step 5: Perform KMeans clustering
# -----------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
cluster_labels = kmeans.fit_predict(positions_array)

# -----------------------------
# Step 6: Generate JSON with cluster assignments
# -----------------------------
cluster_dict = {}
for i in range(optimal_k):
    stations_in_cluster = [station for j, station in enumerate(predicted_positions.keys()) if cluster_labels[j] == i]
    cluster_dict[f"{i+1}"] = stations_in_cluster

# Save JSON file
with open("clusters.json", "w") as f:
    json.dump(cluster_dict, f, indent=4)

print("Clusters saved in clusters.json")
print(json.dumps(cluster_dict, indent=4))

# -----------------------------
# Optional Step 7: Plot 3D clusters
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.get_cmap("Set1", optimal_k)

for i, (station, pos) in enumerate(predicted_positions.items()):
    ax.scatter(pos[0], pos[1], pos[2], color=colors(cluster_labels[i]), s=100)
    ax.text(pos[0], pos[1], pos[2], station, fontsize=9)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Predicted Station Positions and Clusters')
plt.show()