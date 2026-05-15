import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,       # Non serve un'installazione LaTeX esterna
    "font.family": "serif",     # Usa un font con le grazie (stile Roman)
    "mathtext.fontset": "cm",   # Usa Computer Modern (il font standard di LaTeX)
})

def threedplotter(x_query, y, name=None):
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
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    plt.colorbar(scatter, label='$y$ value')

    if name is not None:
        plt.savefig(name)

    plt.show()