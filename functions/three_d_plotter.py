import matplotlib.pyplot as plt

def threedplotter(x_query, y):
    """
    3-D scatterplot.
    
    Parameters:
    - x_query: numpy array (n_samples, 2)
    - y: numpy array (n_samples,)
    
    Makes scatterplot of points with coordinates given by x_query and y
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = x_query[:,0]
    y_coords = x_query[:,1]
    z_coords = y

    scatter = ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis')
    ax.set_xlabel('X (Dimension 1)')
    ax.set_ylabel('X (Dimension 2)')
    ax.set_zlabel('Y (Target)')
    plt.colorbar(scatter, label='Valore di Y')

    plt.show()