import matplotlib.pyplot as plt
import numpy as np

def plotXD( df
            , title=None
            , XYZ=None  # list of column indexes for X,Y,Z - default is [0,1,2]
            , colour_column='ClusterTruth'  # use this column to determine the colour of the point
            , size=50
            , alpha=0.8
            , project2D=False  # 2D graph if true
            , elev=None
            , azim=None
            , roll=None
            , save_to_file = None
           ):

    if XYZ is None:  # default to first three columns for 3D plots
        X = df.columns[0]
        Y = df.columns[1]
        Z = df.columns[2]
    else:  # use what has been provided
        X = df.columns[XYZ[0]]
        Y = df.columns[XYZ[1]]
        Z = df.columns[XYZ[2]]

    markers = ['o', 'X', 'P', '^', 's', 'd', 'v', '<', '>', 'h', '*', '.']
    palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'b']
    category_column = Z

    # cluster should be indicated by colour since this has priority over shape in vision
    if 'Colour' not in df.columns:
        if colour_column in df.columns:
            df['Colour'] = df[colour_column].map(lambda x: palette[x])
        else:
            raise ValueError(f'{colour_column} column does not exist')

    fig = plt.figure(figsize=(10, 10))

    # Set the figure background color to grey
    fig.patch.set_facecolor('white')  # Grey outer background

    if project2D:
        ax = fig.add_subplot()  # an axes object - 2D
    else:
        ax = fig.add_subplot(projection='3d')  # an Axes3D object
        ax.view_init(elev, azim)
        ax.view_init(elev=elev, azim=azim, roll=roll)  # change the viewing angle

        # Set the 3D plot background to white
        ax.set_facecolor('white')  # White background for the 3D plot

        # Set grid color
        grid_color = 'grey'
        ax.xaxis._axinfo['grid'].update(color=grid_color)
        ax.yaxis._axinfo['grid'].update(color=grid_color)
        ax.zaxis._axinfo['grid'].update(color=grid_color)
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # X-axis pane
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Y-axis pane
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Z-axis pane

    handles = []

    for i, category in enumerate(np.unique(df[category_column])):
        marker = markers[i]
        mask = df[category_column] == category

        if project2D:
            ax.scatter(df.loc[mask, X], df.loc[mask, Y]
                       , c=df.loc[mask, 'Colour']
                       , marker='o'
                       , alpha=alpha
                       , s=size
                       )
        else:
            ax.scatter(df.loc[mask, X], df.loc[mask, Y], df.loc[mask, Z]
                       , c=df.loc[mask, 'Colour']
                       , marker=marker
                       , alpha=alpha
                       , s=size
                       )

        # Create a legend entry
        handles.append(plt.Line2D([0], [0]
                                  , marker=marker
                                  , color='w'
                                  , label=f'Category {category:.0f}'
                                  , markerfacecolor='black'
                                  , markersize=10))

    ax.set_xlabel(X)
    ax.set_ylabel(Y)

    if not project2D:
        ax.set_zlabel(Z)
        ticks = np.arange(df[category_column].min(), df[category_column].max() + 1)
        ax.set_zticks(ticks)

        # Legend
        ax.legend(handles=handles, title="Category", bbox_to_anchor=(0.85, 0.95), loc='upper left')

    # Set the title with custom font properties
    # ax.set_title(title, fontsize=10, fontweight='bold', color='black', family='serif', pad=20)

    # Customize grid and background
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Save the plot as a PDF file
    if save_to_file is not None:
        plt.savefig(save_to_file, dpi=None, format="pdf", bbox_inches="tight")

    plt.close()

def likelihood_to_rgb(likelihoods):
    tol = 1e-8
    num_clusters = likelihoods.shape[1]
    colors_rgb = np.zeros((len(likelihoods), 3))
    for i in range(len(likelihoods)):
        if num_clusters >= 1:
            colors_rgb[i][0] = likelihoods[i][0] if likelihoods[i][0] > tol else 0.0  # Red component
        if num_clusters >= 2:
            colors_rgb[i][1] = likelihoods[i][1] if likelihoods[i][1] > tol else 0.0 # Green component
        if num_clusters >= 3:
            colors_rgb[i][2] = likelihoods[i][2] if likelihoods[i][2] > tol else 0.0 # Blue component
    return colors_rgb

