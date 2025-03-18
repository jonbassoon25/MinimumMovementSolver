'''
Problem: 
	Given 2 sets of random coordinates with the same number of items on a 2 dimensional plane 
	where one set is static and the other can be moved,
	determine the assignment of the movable coordinates so that they 
	are positioned on the static coordinates with the least possible movement

	Solution using a greedy policy: Time complexity ranging from O(N) to O(N!) depending on the initial state:
		use most optimal assignment for each movable point
		for group of shared assignments, don't change the assignment of the point that benifits the most from it's optimal assignment, 
			but change the rest to their second most optimal assignment. If one of these assignments conflits with a previous final assignment, 
			consider it a shared assignment on the next repitition
		repeat the preivous step until all assignments are made
'''

import random
import matplotlib.pyplot as plt
import numpy as np
import math


#Generates assignments for a given weight matrix using a greedy policy
def get_greedy_assignments(weight_matrix, rep_limit = 100):
	#Matrix rows are movable points
	#Matrix columns are static points
	if len(weight_matrix) != len(weight_matrix[0]):
		raise Exception("A non-uniform matrix was detected")
	
	#Get relative assignment values in the matrix
	
	for i in range(len(weight_matrix)):
		smallest = math.inf
		for value in weight_matrix[i]:
			if value < smallest:
				smallest = value
		for j in range(len(weight_matrix[i])):
			weight_matrix[i][j] -= smallest

	# assignments[moveable point index] = static point index (assignment index)
	assignments = [None for i in range(len(weight_matrix))]

	# shared_assignments[static point index (assignment index)] = [...movable point indicies]
	shared_assignments = [[] for i in range(len(weight_matrix))]

	#Calculate first optimal weights
	active_shared_assignment_indicies = set()
	for i in range(len(weight_matrix)): #for each movable index
		optimal_static_index = 0
		for j in range(1, len(weight_matrix)):
			if weight_matrix[i][j] < weight_matrix[i][optimal_static_index]:
				optimal_static_index = j
			
		assignments[i] = optimal_static_index
		shared_assignments[optimal_static_index].append(i)
		if len(shared_assignments[optimal_static_index]) > 1:
			active_shared_assignment_indicies.add(optimal_static_index)

	#Calculate final weights
	
	reps = 0
	while (len(active_shared_assignment_indicies) > 0 and reps < rep_limit):
		for shared_assignment_index in list(active_shared_assignment_indicies):
			active_shared_assignment_indicies.remove(shared_assignment_index)
			#Determine the best assignment based on the relative weights, looking for least additional weight
			conflicting_indicies = shared_assignments[shared_assignment_index]

			optimal_index = conflicting_indicies[0]
			for j in range(len(conflicting_indicies)):
				if weight_matrix[conflicting_indicies[j]][shared_assignment_index] < weight_matrix[optimal_index][shared_assignment_index]:
					optimal_index = conflicting_indicies[j]
			
			#Give the optimal index it's wanted assignment
			assignments[optimal_index] = shared_assignment_index
			shared_assignments[shared_assignment_index] = [optimal_index]

			#Give the other indicies the assignment index of the assignment 1 ordinal place less wanted than the optimal
			for j in range(len(conflicting_indicies)):
				if conflicting_indicies[j] == optimal_index:
					continue
				next_optimal_assignment_index = None
				no_index_found = True
				for k in range(len(weight_matrix)):
					if (weight_matrix[conflicting_indicies[j]][k] > weight_matrix[conflicting_indicies[j]][shared_assignment_index]):
						if next_optimal_assignment_index == None or (weight_matrix[conflicting_indicies[j]][k] < weight_matrix[conflicting_indicies[j]][next_optimal_assignment_index]):
							next_optimal_assignment_index = k
							no_index_found = False
				if no_index_found:
					raise Exception("No index was calculated.")
			
				#Assign the new assignment
				assignments[conflicting_indicies[j]] = next_optimal_assignment_index
				shared_assignments[next_optimal_assignment_index].append(conflicting_indicies[j])
				if len(shared_assignments[next_optimal_assignment_index]) > 1:
					active_shared_assignment_indicies.add(next_optimal_assignment_index)
		
		if (reps % 50) == 0:
			print(f">= {reps} reps completed")
		reps += 1

	if reps == rep_limit:
		print("Absolute assignment not found")
	
	return assignments



# Test vars
num_coordinates = 60
coordinate_range = (0, 100)
coordinate_precision = 4 #number of coordinates past the decimal for each generated coordinate
graph_coordinates = True

# Generate coordinates
coordinate_range = (coordinate_range[0] * 10**coordinate_precision, coordinate_range[1] * 10**coordinate_precision)
static_coordinates, movable_coordinates = tuple([[(random.randint(*coordinate_range) / 10**coordinate_precision, random.randint(*coordinate_range) / 10**coordinate_precision) for i in range(num_coordinates)] for j in range(2)])

# Save generated coordinates to file
with open("coordinates.txt", "w") as output_file:
	output_file.write(f"Static Coordinates:\n\t{'\n\t'.join([str(coordinate) for coordinate in static_coordinates])}\n\nMovable Coordinates:\n\t{'\n\t'.join([str(coordinate) for coordinate in movable_coordinates])}")

# Construct matrix for hungarian algorithm where rows are static points, columns are movable points, and values are the eucliean weights
weight_matrix = np.zeros((num_coordinates, num_coordinates), dtype="double")
for i in range(num_coordinates):
	for j in range(num_coordinates):
		x_distance = static_coordinates[j][0] - movable_coordinates[i][0]
		y_distance = static_coordinates[j][1] - movable_coordinates[i][1]
		weight_matrix[i][j] = round((x_distance**2 + y_distance**2)**0.5, coordinate_precision + 3)

# Display the weight matrix
assignments = get_greedy_assignments(weight_matrix)

# Graph generated coordinates
if graph_coordinates:
	static_x_values, static_y_values = [coordinate[0] for coordinate in static_coordinates], [coordinate[1] for coordinate in static_coordinates]
	movable_x_values, movable_y_values = [coordinate[0] for coordinate in movable_coordinates], [coordinate[1] for coordinate in movable_coordinates]
	
	#Label static coordinates
	offset = (coordinate_range[1] - coordinate_range[0]) / coordinate_range[1] / 8
	for i, x, y in zip(range(1, len(static_coordinates) + 1), static_x_values, static_y_values):
		plt.text(x + offset, y, str(i))
	#Label movable coordinates
	for i, x, y in zip(range(1, len(movable_coordinates) + 1), movable_x_values, movable_y_values):
		plt.text(x + offset, y, str(i))

	#Draw assignment lines
	for i in range(len(assignments)):
		plt.plot((static_x_values[assignments[i]], movable_x_values[i]), (static_y_values[assignments[i]], movable_y_values[i]))
	
	#Plot coordinates
	plt.scatter(static_x_values, static_y_values, c="black")
	plt.scatter(movable_x_values, movable_y_values, c="blue")
	plt.show()