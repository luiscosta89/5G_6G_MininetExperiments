import csv
import random

# Number of stations
num_stations = 25

# File name
filename = "mobile_positions.csv"

# Generate random positions (you can adjust the range if needed)
positions = [
    [f"sta{i+1}", round(random.uniform(0, 100), 2), round(random.uniform(0, 100), 2), round(random.uniform(0, 50), 2)]
    for i in range(num_stations)
]

# Write to CSV
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Station_name", "x", "y", "z"])
    writer.writerows(positions)

print(f"CSV file '{filename}' generated successfully!")
