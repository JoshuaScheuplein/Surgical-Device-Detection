import numpy as np

from matplotlib.axes import Axes
from matplotlib import pyplot as plt


def plot_rays(origins_1, directions_1, origins_2, directions_2, anchor_points_lat=None, anchor_points_ap=None,
              intersection_points=None, distance_matrix=None, roi_range=60.0):
    """
    Plot rays from two sets in 3D space using lines.

    :param origins_1: (K, N, 3) array of origins for the first set of rays.
    :param directions_1: (K, N, 3) array of directions for the first set of rays.
    :param origins_2: (K, M, 3) array of origins for the second set of rays.
    :param directions_2: (K, M, 3) array of directions for the second set of rays.
    :param anchor_points_lat: (K, P, 3) array of closest 3D points on lateral rays.
    :param anchor_points_ap: (K, P, 3) array of closest 3D points on AP rays.
    :param intersection_points: (K, P, 3) array of 3D intersection points.
    :param distance_matrix: (K, P) array of distances between ray pairs.
    :param roi_range: float specifying the maximum extent of the displayed 3D ROI.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho') # forces orthographic (parallel) projection

    K = origins_1.shape[0]
    for k in range(K):

        # Plot the first set of rays (LAT)
        for origin, direction in zip(origins_1[k], directions_1[k]):
            direction *= (2 * np.linalg.norm(origin)) / np.linalg.norm(direction)
            end_point = origin + direction
            ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]],
                    color='#3182bd', label='LAT Rays')

        # Plot the second set of rays (AP)
        for origin, direction in zip(origins_2[k], directions_2[k]):
            direction *= (2 * np.linalg.norm(origin)) / np.linalg.norm(direction)
            end_point = origin + direction
            ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]],
                    color='#756bb1', label='AP Rays')
            
        # Plot anchor points and connecting distance lines
        if anchor_points_lat is not None and anchor_points_ap is not None:
            for p in range(anchor_points_lat.shape[1]):
                ax.plot([anchor_points_lat[k, p, 0], anchor_points_ap[k, p, 0]],
                        [anchor_points_lat[k, p, 1], anchor_points_ap[k, p, 1]],
                        [anchor_points_lat[k, p, 2], anchor_points_ap[k, p, 2]],
                        color='k', linestyle='--')
                ax.scatter(anchor_points_lat[k, p, 0], anchor_points_lat[k, p, 1], anchor_points_lat[k, p, 2],
                        c='#3182bd', s=15)
                ax.scatter(anchor_points_ap[k, p, 0], anchor_points_ap[k, p, 1], anchor_points_ap[k, p, 2],
                        c='#756bb1', s=15)

    # Plot intersection points
    vmin, vmax = 0.0, np.max(distance_matrix)
    triangulated_keypoints = intersection_points.reshape(-1, 3) # (K, P, 3) -> (N, 3)    
    sc = ax.scatter(triangulated_keypoints[:, 0], triangulated_keypoints[:, 1], triangulated_keypoints[:, 2],
                    c=distance_matrix.reshape(-1), cmap='RdYlGn_r', vmin=vmin, vmax=vmax, s=30)
    
    # Plot implant main axes
    if K == 2:
        for p in range(intersection_points.shape[1]):
            ax.plot([intersection_points[0, p, 0], intersection_points[1, p, 0]],
                    [intersection_points[0, p, 1], intersection_points[1, p, 1]],
                    [intersection_points[0, p, 2], intersection_points[1, p, 2]],
                    color='k', linestyle='-')
    
    # Create colorbar for distance matrix
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.15)
    cbar.set_label('Ray Intersection Distance [mm]', labelpad=15)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set ROI limits
    center = np.array([0, 0, 0])
    # center = np.mean(intersection_points, axis=0)
    ax.set_xlim(center[0] - roi_range, center[0] + roi_range)
    ax.set_ylim(center[1] - roi_range, center[1] + roi_range)
    ax.set_zlim(center[2] - roi_range, center[2] + roi_range)

    # Ensure that all axes of the 3D plot have equal scale
    # (-> avoid perspective distortions)
    set_axes_equal(ax)

    # Add legend (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()    # Collect all existing legend handles and labels
    by_label = dict(zip(labels, handles))               # Create a dictionary mapping from label -> handle
    ax.legend(by_label.values(), by_label.keys(),       # Pass only the unique (values, keys) pairs to the legend
              loc='upper center', ncols=2)

    plt.tight_layout()
    # plt.show()

    return


def set_axes_equal(ax: Axes):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc... This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence we call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return
