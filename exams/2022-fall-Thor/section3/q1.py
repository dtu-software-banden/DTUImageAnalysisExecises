import numpy as np


array = [
    [64, 94, 21, 19, 31],
    [38, 88, 30, 23, 92],
    [81, 55, 47, 17, 43],
    [53, 62, 23, 23, 18],
    [35, 59, 84, 44, 90]
]
output = [array[0]]
for y in range(1,5):
    line = []
    for x in range(len(array[y])):
        if x == 0:
            line.append(array[x][y] + min(output[y-1][x],output[y-1][x+1]))
        elif x == len(array[y]) - 1:
            line.append(array[x][y] + min(output[y-1][x],output[y-1][x-1]))
        else:
            line.append(array[x][y] + min(output[y-1][x-1],output[y-1][x],output[y-1][x+1]))

    output.append(line)

print(np.array(output))
# VIRKER IKKE !!!