import numpy as np
import matplotlib.pyplot as plt


def graph_map(f, center, bounds):
    samples = 20
    dx = 2 * bounds[0] / samples
    dy = 2 * bounds[1] / samples
    x, y = np.mgrid[slice(center[0]-bounds[0], center[0]+bounds[0]+dx, dx), 
                    slice(center[1]-bounds[1], center[1]+bounds[1]+dy, dy)]

    z = np.zeros(x.shape)

    for i in range(len(x)):
        for j in range(len(x[i])):
            z[i][j] = f((x[i][j], y[i][j]))

    z = ((z - np.min(z)) / (np.max(z) - np.min(z))) ** 0.4

    plt.contourf(x, y, z, levels=50)

    
def graph_path(path):
    plt.plot(path[:, 0], path[:, 1])
    #plt.scatter(path[:, 0], path[:, 1], s=6, c=range(len(path)), cmap='tab20')
    plt.scatter(path[-1, 0], path[-1, 1])
    plt.scatter(path[0, 0], path[0, 1], marker='x')

    
def graph_full(func_d, center, bounds, path, show=True):
    plt.figure(figsize=(12,12))

    graph_map(func_d['func'], center, bounds)
    graph_path(path)
    
    plt.title(func_d['title'])
    plt.xlim((center[0] - bounds[0], center[0] + bounds[0]))
    plt.ylim((center[1] - bounds[1], center[1] + bounds[1]))
    if show:
        plt.show()
