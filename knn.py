import numpy as np
import matplotlib.pyplot as plt

# set colors for different classes
colors = {0: 'r',
          1: 'g',
          2: 'b'}

# create some random data
train_data = []
for i in range(100):
    train_data.append(np.array([np.random.normal(0, 2), np.random.normal(1, 2), 0]))
    train_data.append(np.array([np.random.normal(1, 5), np.random.normal(-2, 1), 1]))
    train_data.append(np.array([np.random.normal(4, 1), np.random.normal(3, 1), 2]))

# plot it
plt.figure()
for i in range(len(train_data)):
    point = train_data[i]
    plt.scatter(point[0], point[1], c=colors[point[2]])
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# create some random test data
test_data = []
for i in range(500):
    test_data.append(np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), -1]))

# set number of nearest neighbors
k = 3

def euclidean_distance(point_1, point_2):
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

for i in range(len(test_data)):
    # get euclidean distance with each training datapoint
    distances = [ euclidean_distance(test_data[i][:2], train_data[j][:2]) for j in range(len(train_data)) ]

    # sort from lowest to highest
    sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])

    # get k nearest neighbors
    neighbors = sorted_distances[:k]

    # get the most common class
    class_votes = {}
    for neighbor in neighbors:
        neighbor_class = train_data[neighbor[0]][2]

        if neighbor_class not in class_votes.keys():
            class_votes[neighbor_class] = 1
        else:
            class_votes[neighbor_class] += 1

    # get top class
    sorted_class_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    top_class = sorted_class_votes[0][0]

    test_data[i][2] = top_class

# plot results
plt.figure()
for i in range(len(train_data)):
    point = train_data[i]
    plt.scatter(point[0], point[1], c=colors[point[2]], alpha=0.1)
for i in range(len(test_data)):
    point = test_data[i]
    plt.scatter(point[0], point[1], c=colors[point[2]], alpha=0.5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()