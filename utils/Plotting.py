from matplotlib.colors import ListedColormap
import numpy as np

pr_cmap_values = np.array([(1.0, 0, 0, i / 255) for i in range(256)])
pr_cmap = ListedColormap(pr_cmap_values)
