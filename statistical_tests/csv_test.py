import numpy as np
import csv


col_headers = ["blank", "Classifier 1", "Classifier 2", "Classifier 3"]
row_headers = ["dataset1", "dataset2", "dataset3"]


better_data = [[1,2], ["all"], [5,6]]

data = [0.5, 0.2, 0.3]

row = [row_headers[0], data[0], data[1], data[2]]
print(row)

f = open('test.csv', 'w')
writer = csv.writer(f)
writer.writerow(row)
f.close()

