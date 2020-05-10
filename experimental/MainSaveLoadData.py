import tables
import numpy as np


# write some data to a file
filename = 'test.h5'
ROW_SIZE = 100
NUM_COLUMNS = 200

f = tables.open_file(filename, mode='w')
atom = tables.Float64Atom()

array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))

for idx in range(NUM_COLUMNS):
    x = np.random.rand(1, ROW_SIZE)
    array_c.append(x)
f.close()


# append to the dataset
f = tables.open_file(filename, mode='a')
f.root.data.append(x)
f.close()

# read a subset of the data
f = tables.open_file(filename, mode='r')
print(f.root.data[1:10, 2:20])
f.close()
