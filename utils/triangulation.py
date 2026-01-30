import logging

import numpy as np
from scipy.optimize import minimize


def compute_ray_from_2d(positions_2d: np.ndarray, view_index: int, source_points: np.ndarray, invKRs: np.ndarray):
    """
    Compute 3D rays from 2D positions using the projection matrix.

    Args:
        positions_2d: Array of detected 2D positions
        view index: index of the current view w.r.t. the full sequence of 3D scan projections
        source_points: 3D source locations (i.e., ray origins) for computation of ray directions
        invKRs: Inverse of projection matrices for computation of ray directions
    
    Returns:
        Array of 3D ray origins and directions
    """

    N = positions_2d.shape[0]

    # Compute inverse of the projection matrix
    assert source_points.shape[1] == (3) and invKRs.shape[1:] == (3, 3) # (num_views, 3) and (num_views, 3, 3)
    source_point, invKR = source_points[view_index], invKRs[view_index]

    # Convert 2D positions to homogeneous coordinates
    positions_2d_hom = np.hstack((positions_2d, np.ones((positions_2d.shape[0], 1)))) # (N, 2) -> (N, 3)

    # Compute ray directions in 3D space
    ray_origins = np.tile(source_point.squeeze(), (N, 1)) # repeat / duplicate
    ray_directions = np.dot(invKR.squeeze(), positions_2d_hom.T).T # ((3, 3) @ (3, N)).T = (N, 3)
    assert np.all(ray_origins.shape == ray_directions.shape) # (N, 3)

    return ray_origins, ray_directions # (N, 3), (N, 3)


def closest_distance_between_rays(ray1_origin, ray1_direction, ray2_origin, ray2_direction):
    """
    Compute the closest distance between two rays.

    :param ray1_origin: Origin of the first ray
    :param ray1_direction: Direction of the first ray
    :param ray2_origin: Origin of the second ray
    :param ray2_direction: Direction of the second ray
    :return: The closest distance, closest points, and center point between the two rays
    """

    # ensure directions are float arrays
    ray1_origin = ray1_origin.astype(np.float64)
    ray2_origin = ray2_origin.astype(np.float64)
    ray1_direction = ray1_direction.astype(np.float64)
    ray2_direction = ray2_direction.astype(np.float64)

    w0 = ray1_origin - ray2_origin
    a = np.dot(ray1_direction, ray1_direction)
    b = np.dot(ray1_direction, ray2_direction)
    c = np.dot(ray2_direction, ray2_direction)
    d = np.dot(ray1_direction, w0)
    e = np.dot(ray2_direction, w0)

    denom = a * c - b * b

    if abs(denom) < 1e-12:
        # nearly parallel; choose s=0
        s = 0.0
        t = (b * 0 - e) / c if c > 1e-12 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    closest_point_ray1 = ray1_origin + s * ray1_direction
    closest_point_ray2 = ray2_origin + t * ray2_direction
    center_point = closest_point_ray1 + 0.5 * (closest_point_ray2 - closest_point_ray1)

    return np.linalg.norm(closest_point_ray1 - closest_point_ray2), closest_point_ray1, closest_point_ray2, center_point


def find_intersecting_rays(positions_2d_lat: np.ndarray, positions_2d_ap: np.ndarray, view_index_lat: int, view_index_ap: int,
                           source_points: np.ndarray, invKRs: np.ndarray):
    """
    Find 3D positions of intersecting rays between two sets of 3D lines.

    Args:
        positions_2d_lat: Array in shape (N, 2) containing 2D positions for the first (lateral) set of rays
        positions_2d_ap: Array in shape (M, 2) containing 2D positions for the second (anterior-posterior) set of rays
        view_index_lat: view index of the first (lateral) view w.r.t. the full sequence of 3D scan projections
        view_index_ap: view index of the second (anterior-posterior) view w.r.t. the full sequence of 3D scan projections
        source_points: 3D source locations (i.e., ray origins) for computation of ray directions
        invKRs: Inverse of projection matrices for computation of ray directions

    Returns:
        Distance matrix and 3D intersection points for all tuples of rays
    """

    N, M = len(positions_2d_lat), len(positions_2d_ap)
    origins_lat, directions_lat = compute_ray_from_2d(positions_2d_lat, view_index_lat, source_points, invKRs)
    origins_ap, directions_ap = compute_ray_from_2d(positions_2d_ap, view_index_ap, source_points, invKRs)

    distance_matrix = np.zeros((N, M))
    anchor_points_lat = np.zeros((N, M, 3))
    anchor_points_ap = np.zeros((N, M, 3))
    intersection_points = np.zeros((N, M, 3))

    for i in range(N):
        for j in range(M):
            distance, anchor_point_lat, anchor_point_ap, intersection_point = closest_distance_between_rays(
                origins_lat[i], directions_lat[i],
                origins_ap[j], directions_ap[j]
            )
            distance_matrix[i, j] = distance
            anchor_points_lat[i, j] = anchor_point_lat
            anchor_points_ap[i, j] = anchor_point_ap
            intersection_points[i, j] = intersection_point

    plot_data = [origins_lat, directions_lat, origins_ap, directions_ap,
                 anchor_points_lat, anchor_points_ap]

    return distance_matrix, intersection_points, plot_data


def compute_3d_orientation(
    tip_3d, i, j, # Note: (i, j) = (lat_idx, ap_idx)
    origins_lat_tip_kps, directions_lat_tip_kps,
    origins_ap_tip_kps, directions_ap_tip_kps,
    origins_lat_dir_kps, directions_lat_dir_kps,
    origins_ap_dir_kps, directions_ap_dir_kps,
    method="auto",
    parallel_eps=1e-6,
    plane_angle_switch_deg=10.0,
    icr_opts=None
    ):
    """
    Compute a 3D orientation vector for a matched LAT/AP pair (i, j).

    Returns:
        orientation_3d: (3,)
        quality_metric: float (distance -> smaller = better)
        plane_angle_deg: float (debug)
    """

    # Compute plane normal vector for each view (source -> tip_kp cross source -> dir_kp)
    n1 = np.cross(directions_lat_tip_kps[i], directions_lat_dir_kps[i])
    n2 = np.cross(directions_ap_tip_kps[j], directions_ap_dir_kps[j])
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    # Compute the angle between the two plane normals
    if n1_norm > 1e-9 and n2_norm > 1e-9:
        n1u = n1 / n1_norm
        n2u = n2 / n2_norm
        dot = np.clip(abs(np.dot(n1u, n2u)), -1.0, 1.0)
        plane_angle = np.degrees(np.arccos(dot))
    else:
        # If one normal is degenerated, force the use of ICR method (planes ill-defined)
        n1u = n1 / (n1_norm + 1e-12)
        n2u = n2 / (n2_norm + 1e-12)
        plane_angle = 0.0

    # Decide which method should be used (either CP, ICR, or Auto)
    use_icr = (method.lower() == "icr") or \
              (method.lower() == "auto" and plane_angle < plane_angle_switch_deg)

    # Precompute midpoint (closest points) between direction rays (used both by CP fallback and sign decision)
    distance, c1, c2, mid_point = closest_distance_between_rays(
        origins_lat_dir_kps[i], directions_lat_dir_kps[i],
        origins_ap_dir_kps[j],  directions_ap_dir_kps[j]
    )

    ##########################################################
    # Cross-Product (CP) Method
    ##########################################################
    # In CP, the implant orientation is the line of intersection of the two planes.
    # This direction equals the cross product of the two plane normals.
    ##########################################################
    if not use_icr:
        axis_dir = np.cross(n1, n2)

        # Fallback: If planes are nearly parallel, approximate using the line between direction rays
        if np.linalg.norm(axis_dir) < parallel_eps:
            axis_dir = c2 - c1

        # Normalize direction vector to unit length
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

        # Ensure 3D axis points consistently away from 3D tip keypoint
        v_head = mid_point - tip_3d # Vector from triangulated 3D tip towards the 3D intersection point of direction rays
        if np.linalg.norm(v_head) > 1e-9:
            if np.dot(axis_dir, v_head) < 0: # If orientation points opposite to that direction, flip it
                axis_dir = -axis_dir
        else:
            # Ambiguous head location -> prefer axis that has positive dot with average ray directions
            avg_ray = directions_lat_dir_kps[i] / (np.linalg.norm(directions_lat_dir_kps[i]) + 1e-12) + \
                      directions_ap_dir_kps[j] / (np.linalg.norm(directions_ap_dir_kps[j]) + 1e-12)
            if np.dot(axis_dir, avg_ray) < 0: # If orientation points opposite to that direction, flip it
                axis_dir = -axis_dir

        return axis_dir, distance, plane_angle

    ##########################################################
    # Iterative-Closest-Ray (ICR) Method
    ##########################################################
    # ICR refines the orientation by minimizing the perpendicular distances
    # between the candidate implant axis and both direction rays.
    ##########################################################

    # Variable names according to the paper
    t = tip_3d
    s1 = origins_lat_dir_kps[i]; s2 = origins_ap_dir_kps[j]
    p1 = directions_lat_dir_kps[i]; p2 = directions_ap_dir_kps[j]

    # Function that maps inclination and azimuth angles (theta, phi) to an unit 3D vector d
    # (See https://en.wikipedia.org/wiki/Spherical_coordinate_system)
    def d_from_angles(angles):
        theta, phi = angles
        d = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        d /= np.linalg.norm(d) + 1e-12
        return d

    # Per-view optimization term:
    # term_n(d) = | ( d x p_n ) * ( t - s_n ) | / || d x p_n ||
    def icr_term(d_vec, p_n, s_n, t_vec):
        cross = np.cross(d_vec, p_n)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            # nearly parallel: return large penalty (discourage this d)
            return 1e6
        num = abs(np.dot(cross, (t_vec - s_n)))
        return num / norm_cross

    # Objective E(d) = term_1(d) + term_2(d)
    def icr_objective(angles):
        d_candidate = d_from_angles(angles)
        term1 = icr_term(d_candidate, p1, s1, t)
        term2 = icr_term(d_candidate, p2, s2, t)
        return term1 + term2

    # Initialization: Mekki's paper suggests using t - p as an initial estimate
    init_dir = t - mid_point
    init_dir = init_dir / (np.linalg.norm(init_dir) + 1e-12)

    # Convert to angles (See https://en.wikipedia.org/wiki/Spherical_coordinate_system)
    theta_0 = np.arccos(np.clip(init_dir[2], -1.0, 1.0))    # theta = arccos(z / r) (r = 1.0 because unit vector)
    phi_0 = np.arctan2(init_dir[1], init_dir[0])            # phi = arctan2(y / x)

    # Optimize using 'Nelder-Mead' method
    res = minimize(icr_objective, [theta_0, phi_0], method="Nelder-Mead",
                    options={'maxiter'   : icr_opts.get("maxiter", 200),
                             'xatol'     : icr_opts.get("tol", 1e-6),
                             'fatol'     : 1e-8})
    logging.info(f"Optimizer finished after {res.nit} iterations with success = '{res.success}':")
    logging.info(res.message)

    # Best 3D direction vector
    theta_opt, phi_opt = res.x
    d_opt = d_from_angles([theta_opt, phi_opt])

    # Ensure 3D axis points consistently away from 3D tip keypoint
    v_head = mid_point - t # Vector from triangulated 3D tip toward the 3D intersection point of direction rays
    if np.linalg.norm(v_head) > 1e-9:
        if np.dot(d_opt, v_head) < 0: # If orientation points opposite to that direction, flip it
            d_opt = -d_opt
    else:
        # Fallback sign rule: match average view ray
        avg_ray = p1 / (np.linalg.norm(p1) + 1e-12) + p2 / (np.linalg.norm(p2) + 1e-12)
        if np.dot(d_opt, avg_ray) < 0: # If orientation points opposite to that direction, flip it
            d_opt = -d_opt

    # Orientation quality metric: We store the objective value (smaller = better)
    objective = res.fun # Value of the objective function at x
    rms_dist = np.sqrt(objective / 2.0) # Convert objective to RMS distance (mm) for comparability to CP distance

    return d_opt, rms_dist, plane_angle


"""
Helper functions for performing triangulation refinement
(i.e., find additional 3D implants by re-matching potentially occluded 2D instances
from the first scout view with unmatched 2D predictions in the second scout view)
"""


def _angle_between_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    """ Return angle in degrees between vectors a and b (safe to zero-length). """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    cosang = np.dot(a, b) / (na * nb)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _project_line_to_view(tip3d: np.ndarray, dir3d: np.ndarray,
                          proj_mat_pfw: np.ndarray, length_mm: float = 50.0):
    """
    Project a 3D line / implant axis (defined by tip3d + dir3d) into 2D.

    Args:
        tip3d: (3,) tip point in world coordinates
        dir3d: (3,) unit direction vector
        proj_mat_pfw: 3×4 projection matrix (world -> pixel)
        length_mm: length along dir3d to sample the second point

    Returns:
        p1_uv, p2_uv: two 2D points (each shape (2,)) in (u, v) coordinates
    """
    p1 = tip3d
    p2 = tip3d + length_mm * dir3d
    points3d = np.stack((p1, p2), axis=0) # The axis parameter specifies the index of the new axis in the dimensions of the result. 
    assert points3d.shape == (2, 3)

    # homogeneous coordinates
    points3d_homo = np.pad(points3d, ((0, 0), (0, 1)), constant_values=1).T # (4, N)

    # projection from 3D volume to 2D image plane
    points2d_homo = proj_mat_pfw @ points3d_homo # (3, 4) @ (4, N) = (3, N)
    points2d = (points2d_homo.squeeze()[:2] / points2d_homo.squeeze()[-1]).T # (N, 2)
    
    return points2d[0], points2d[1]


def _is_duplicate_epipolar_implant(t1: np.ndarray, h1: np.ndarray, t2: np.ndarray, h2: np.ndarray,
                                   duplicate_distance_thresh_mm: float, duplicate_angle_thresh_deg: float) -> bool:
    """
    Determine whether implant (t2, h2) is effectively a duplicate of (t1, h1).
    Criteria: midpoint distance small.
    """
    mid1 = 0.5 * (t1 + h1)
    mid2 = 0.5 * (t2 + h2)
    mid_dist = np.linalg.norm(mid2 - mid1) 
    return mid_dist <= duplicate_distance_thresh_mm


def _is_duplicate_directional_implant(existing_tip: np.ndarray, existing_dir: np.ndarray,
                                      candidate_tip: np.ndarray, candidate_dir: np.ndarray,
                                      dist_thresh_mm: float, angle_thresh_deg: float) -> bool:
    """
    Determine whether a candidate tip + orientation is effectively a duplicate
    of an existing directional implant.

    Criteria:
      - Tip positions are closer than dist_thresh_mm
      - Orientation vectors differ by at most angle_thresh_deg
    """
    tip_dist = np.linalg.norm(existing_tip - candidate_tip)
    ang = _angle_between_vectors_deg(existing_dir, candidate_dir)
    return (tip_dist <= dist_thresh_mm) and (ang <= angle_thresh_deg)


"""
Helper functions for performing triangulation with uncalibrated instead of
calibrated system projection matrices.
"""


def _get_dicom_tag(dicom_data, tag):
    """ Return the raw value of a DICOM tag or None if missing. """
    elem = dicom_data.get(tag, None)
    return None if elem is None else elem.value


def _get_angle_array(angle_start: float, angle_increment: float, num_views: int, mode: str):
    assert mode in ["orbital", "angular"], f"Mode '{mode}' is not supported!"

    # default start angle if missing
    angle_start = 0.0 if angle_start is None else float(angle_start)

    # scalar increment
    if np.isscalar(angle_increment):
        angle_increment = float(angle_increment)
        print(f"Using [scalar] angle increment for computing [{mode}] angle array! ({angle_increment})")

        if mode == "orbital":
            # include the first frame at the start angle:
            # angles[n] = start + angle_increment * n for n=0...num_views-1
            angles = angle_start + angle_increment * np.arange(int(num_views), dtype=float)
        else:
            # angular angles are encoded as single float value in 'Positioner Secondary Angle Increment'
            angular_angle = angle_start + angle_increment
            angles = np.full(num_views, angular_angle)

    # multi-value increment    
    else:
        print(f"Using [multi-value] angle increment for computing [{mode}] angle array! ({angle_increment})")
        angles = angle_start + np.array(angle_increment, dtype=float)
    
    assert len(angles) == num_views, f"Expected {num_views} angle values but got {len(angles)}"

    return angles
