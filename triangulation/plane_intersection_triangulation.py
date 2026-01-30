import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.plotting import plot_rays
from utils.triangulation import compute_ray_from_2d, find_intersecting_rays, compute_3d_orientation
from utils.triangulation import _project_line_to_view, _angle_between_vectors_deg, _is_duplicate_directional_implant


#####################################################################################################
# thresholds
#####################################################################################################
parallel_eps                    = 1e-6  # for cross product magnitudes
plane_angle_switch_deg          = 10.0  # below this angle, switch to ICR when method='auto'

kp_2d_thresh_px                 = 30    # 30px * 0.3mm ≈ 10mm
axis_2d_thresh_deg              = 20.0  # above this angle, do not enforce refinement

duplicate_kp_thresh_mm          = 10.0
duplicate_angle_thresh_deg      = 10.0
#####################################################################################################


def plane_intersection_triangulation(lat_keypoints: np.ndarray, ap_keypoints: np.ndarray, lat_view_idx: int, ap_view_idx: int,
                                     projection_matrices: np.ndarray, source_points: np.ndarray, invKRs: np.ndarray,
                                     mu_3d: float, method: str = "auto", icr_opts: dict = None,
                                     enable_refinement: bool = False, debug: bool = False):
    """
    Triangulate implants where each detection provides:
      - tip keypoint (index 0) w.r.t. 2D projection images
      - direction keypoint (index 2), which indicates 2D orientation but not necessarily end-point

    Implements two methods (following Mekki et al. 2023):
      - 'cp'    : Cross-Product / plane-plane intersection (analytical, fast)
      - 'icr'   : Iterative Closest Ray (optimization-based, robust for nearly in-plane wires)
      - 'auto'  : choose CP by default, but switch to ICR if planes are near-parallel (low angle between normals)

    Args:
        lat_keypoints: (N, 4, 3) array containing 2D detections from the first (lateral) view; element[0,:2] = tip (x,y), element[2,:2] = direction point (x,y)
        ap_keypoints:  (M, 4, 3) array containing 2D detections from the second (AP) view; element[0,:2] = tip (x,y), element[2,:2] = direction point (x,y)
        lat_view_idx, ap_view_idx: indices of views in projection_matrices
        projection_matrices: Projection matrices in pixel from voxel mapping
        source_points: 3D source locations (i.e., ray origins) for computation of ray directions
        invKRs: Inverse of projection matrices for computation of ray directions
        mu_3d: distance threshold in mm for accepting triangulated tips
        method: 'auto' | 'cp' | 'icr'
        icr_opts: dict with optimizer settings (maxiter, tol)
        enable_refinement: whether unmatched predictions from one view should be checked for correspondence
                           with already matched (and potentially occluding) predictions from the other view

    Returns:
        tips_3d: (L, 3) array of triangulated tip positions
        orientations_3d: (L, 3) array of unit orientation vectors (pointing away from tip)
        filtered_distance_matrix: (L, 2): [:,0]=tip-pair distances used to accept pair (mm), [:,1]=quality metric for orientation
        (optional) debug_data: dict with intermediate arrays
    """

    # check if model has predicted any object instances at all
    if len(lat_keypoints) == 0 or len(ap_keypoints) == 0:
        return [], [], [], []

    if icr_opts is None:
        icr_opts = {"maxiter": 200, "tol": 1e-6}

    ################################################
    # Triangulation of 3D Tip Coordinates
    ################################################

    # Extract 2D tip and direction coords (x, y) and convert to (u, v) pixel convention
    tip_kps_lat_view = np.asarray([det[0, :2] for det in lat_keypoints])[:, ::-1]   # (N,2)
    dir_kps_lat_view = np.asarray([det[2, :2] for det in lat_keypoints])[:, ::-1]   # (N,2)
    tip_kps_ap_view = np.asarray([det[0, :2] for det in ap_keypoints])[:, ::-1]     # (M,2)
    dir_kps_ap_view = np.asarray([det[2, :2] for det in ap_keypoints])[:, ::-1]     # (M,2)

    # Triangulate 3D tip positions from ray intersections (-> backproject tip rays and compute closest points)
    dist_matrix_tips, intersection_points_tips, plot_data_tips = find_intersecting_rays(
        positions_2d_lat=tip_kps_lat_view, positions_2d_ap=tip_kps_ap_view,
        view_index_lat=lat_view_idx, view_index_ap=ap_view_idx,
        source_points=source_points, invKRs=invKRs
    )
    if debug:
        origins_lat_tips, directions_lat_tips, origins_ap_tips, directions_ap_tips, anchor_points_lat_tips, anchor_points_ap_tips = plot_data_tips
        plot_rays(np.expand_dims(origins_lat_tips, axis=0), np.expand_dims(directions_lat_tips, axis=0), np.expand_dims(origins_ap_tips, axis=0), np.expand_dims(directions_ap_tips, axis=0),
                  np.expand_dims(anchor_points_lat_tips.reshape(-1, 3), axis=0), np.expand_dims(anchor_points_ap_tips.reshape(-1, 3), axis=0),
                  np.expand_dims(intersection_points_tips.reshape(-1, 3), axis=0), np.expand_dims(dist_matrix_tips.reshape(-1), axis=0))

    # Globally optimal 'Hungarian' assignment
    max_cost_value = 1e6 # 'linear_sum_assignment' does not support np.inf values as cost
    C = np.full_like(dist_matrix_tips, fill_value=max_cost_value, dtype=float)
    valid_mask = dist_matrix_tips < mu_3d
    C[valid_mask] = dist_matrix_tips[valid_mask]
    row_ind, col_ind = linear_sum_assignment(C)

    # Keep only valid assignments (with finite cost)
    valid_pairs_mask = C[row_ind, col_ind] < max_cost_value
    row_valid = row_ind[valid_pairs_mask]
    col_valid = col_ind[valid_pairs_mask]
    matched_pairs = list(zip(row_valid, col_valid))

    if len(matched_pairs) == 0:
        logging.info("No tip matches found under the provided mu_3d threshold!")
        return [], [], [], []

    ################################################
    # Calculation of 3D Direction Vectors
    ################################################

    # Initialize outputs and helpers
    K = len(matched_pairs)
    tips_3d = np.zeros((K, 3))
    orientations_3d = np.zeros((K, 3))
    tip_pair_distances = np.full((K,), np.inf) # first column of filtered_distance_matrix
    orientation_quality = np.full((K,), np.inf) # second column of filtered_distance_matrix
    debug_data = {
        "plane_angles_deg"  : [],
    }

    # Compute all backprojected 3D rays for tips and direction points
    origins_lat_tip_kps, directions_lat_tip_kps = compute_ray_from_2d(tip_kps_lat_view, lat_view_idx, source_points=source_points, invKRs=invKRs)   # (N, 3)
    origins_ap_tip_kps, directions_ap_tip_kps = compute_ray_from_2d(tip_kps_ap_view, ap_view_idx, source_points=source_points, invKRs=invKRs)       # (M, 3)
    origins_lat_dir_kps, directions_lat_dir_kps = compute_ray_from_2d(dir_kps_lat_view, lat_view_idx, source_points=source_points, invKRs=invKRs)   # (N, 3)
    origins_ap_dir_kps, directions_ap_dir_kps = compute_ray_from_2d(dir_kps_ap_view, ap_view_idx, source_points=source_points, invKRs=invKRs)       # (M, 3)    

    # For each matched pair, compute 3D orientation vector using selected method
    for k, (i, j) in enumerate(matched_pairs):
        tip_3d = intersection_points_tips[i, j]

        # Store 3D tip location and intersection distance
        tips_3d[k] = tip_3d
        tip_pair_distances[k] = dist_matrix_tips[i, j]

        # Compute 3D axis orientaion vector
        dir_3d, quality_metric, plane_angle_deg = compute_3d_orientation(
            tip_3d, i, j,
            origins_lat_tip_kps, directions_lat_tip_kps,
            origins_ap_tip_kps, directions_ap_tip_kps,
            origins_lat_dir_kps, directions_lat_dir_kps,
            origins_ap_dir_kps, directions_ap_dir_kps,
            method, parallel_eps, plane_angle_switch_deg, icr_opts
        )

        # Store 3D axis orientaion vector and debugging infos
        orientations_3d[k] = dir_3d
        orientation_quality[k] = quality_metric # Orientation quality metric: distance between direction rays in mm (smaller = better)
        debug_data["plane_angles_deg"].append(plane_angle_deg) 

    # # (Optional:) Compute tip-to-axis distance for final filtering of ill-posed implant candidates
    # tip_to_axis_dist = np.zeros((K,))
    # for idx in range(K):
    #     tip = tips_3d[idx]
    #     axis_dir = orientations_3d[idx]
    #     i, j = matched_pairs[idx]
    #     _, _, _, mid_point = closest_distance_between_rays(origins_lat_dir_kps[i], directions_lat_dir_kps[i],
    #                                                        origins_ap_dir_kps[j], directions_ap_dir_kps[j])
    #     tproj = np.dot((tip - mid_point), axis_dir)
    #     q = mid_point + tproj * axis_dir
    #     tip_to_axis_dist[idx] = np.linalg.norm(tip - q)

    # debug_data.update({"tip_to_axis_dist" : tip_to_axis_dist})
    
    # # Filter by tip-to-axis distance and orientation quality
    # mask = np.logical_and(tip_to_axis_dist < mu_3d, orientation_quality < (3.0 * mu_3d))
    # if not np.any(mask):
    #     logging.info("No tip-direction pairs passed the tip-to-axis and orientation-quality filters!")
    #     return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), debug_data

    # tips_3d = tips_3d[mask]
    # orientations_3d = orientations_3d[mask]
    # tip_pair_distances = tip_pair_distances[mask]
    # orientation_quality = orientation_quality[mask]

    filtered_distance_matrix = np.stack((tip_pair_distances, orientation_quality), axis=1)

    # Final refinement step to retrieve additional 3D instances based on unmatched 2D detections from potentially occluded implants
    if enable_refinement:
        N_lat, M_ap = len(lat_keypoints), len(ap_keypoints)
        assert dist_matrix_tips.shape == (N_lat, M_ap)

        # (Optional:) used for 2D filtering criterion
        proj_mat_pfw_lat = projection_matrices[lat_view_idx]
        proj_mat_pfw_ap = projection_matrices[ap_view_idx]

        # We'll perform two passes:
        #  1) Try to match each unmatched AP detection against LAT detections
        #  2) Try to match each unmatched LAT detection against AP detections
        # This symmetric approach recovers candidates (regardless of which view suffered occlusion).

        # Pass 1: Try to match each unmatched AP detection against LAT detections
        logging.info("Matching each unmatched AP detection against LAT detections ...")
        matched_ap_mask = np.zeros(M_ap, dtype=bool)
        matched_ap_mask[col_valid] = True
        unmatched_ap = np.where(~matched_ap_mask)[0]
        for j in unmatched_ap:
            assert j not in col_valid

            for i in range(N_lat):
                if (i, j) in matched_pairs:
                    logging.info(f"AP detection {j} has already been matched with LAT detection {i} -> Skipping to next candidate pair in refinement step ...")
                    continue

                d_tip_3d = dist_matrix_tips[i, j]
                p_tip_3d = intersection_points_tips[i, j]

                # Skip pairing candidate if none of the two keypoint triangulations fulfills ray intersection criterion
                if mu_3d <= d_tip_3d:
                    logging.info(f"Pairing between AP detection {j} and LAT detection {i} exceeds maximum allowed 3D keypoint error: \
                                 mu_3d = {mu_3d} mm <= tip_dist_3d = {d_tip_3d:.2f} mm -> Skipping to next candidate pair in refinement step ..")
                    continue                

                # Triangulate axis for this pairing
                dir_3d, quality_metric, plane_angle_deg = compute_3d_orientation(
                    p_tip_3d, i, j, # Note: (i, j) = (lat_idx, ap_idx)
                    origins_lat_tip_kps, directions_lat_tip_kps,
                    origins_ap_tip_kps, directions_ap_tip_kps,
                    origins_lat_dir_kps, directions_lat_dir_kps,
                    origins_ap_dir_kps, directions_ap_dir_kps,
                    method, parallel_eps, plane_angle_switch_deg, icr_opts
                )                
            
                # # (Optional:) Consistency check between triangulated 3D result and 2D detection
                # unmatched_tip_2d = tip_kps_ap_view[j]
                # unmatched_dir_2d = dir_kps_ap_view[j]
                # proj_mat_unmatched_view = proj_mat_pfw_ap

                # # Check if projected 3D tip of matching candidate is close to unmatched 2D tip prediction
                # proj_tip_kp_uv, proj_dir_kp_uv = _project_line_to_view(p_tip_3d, dir_3d, proj_mat_unmatched_view)
                # tip_kp_distance_2d = np.linalg.norm(unmatched_tip_2d - proj_tip_kp_uv)
                # if kp_2d_thresh_px < tip_kp_distance_2d:
                #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: tip_kp_dist_2d = {tip_kp_distance_2d:.2f}px]")
                #     continue # too far, skip this candidate
                
                # # Check if projected 3D axis of matching candidate is close to unmatched 2D axis prediction
                # unmatched_axis_3d_to_2d = proj_tip_kp_uv - proj_dir_kp_uv
                # unmatched_axis_2d = unmatched_tip_2d - unmatched_dir_2d
                # axis_angle_2d = _angle_between_vectors_deg(unmatched_axis_3d_to_2d, unmatched_axis_2d)
                # if axis_2d_thresh_deg < axis_angle_2d:
                #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: ax_ang_2d = {axis_angle_2d:.2f}°")
                #     continue # too divergent, skip this candidate

                print(f"\t\tRefinement Step: Accepted previously unmatched pairing between AP detection {j} and LAT detection {i} \
                        (kp_quality_3d = {d_tip_3d:.2f} mm, ax_quality_3d = {quality_metric:.2f} mm)")

                # # (Optional:) Duplicate check with already triangulated implant instances
                # is_duplicate = False
                # for existing_tip_3d, existing_head_3d in zip(tips_3d, orientations_3d):
                #     if _is_duplicate_directional_implant(existing_tip_3d, existing_head_3d, p_tip_3d, dir_3d, duplicate_kp_thresh_mm, duplicate_angle_thresh_deg):
                #         is_duplicate = True # set flag if at least one duplication with already existing matches is detected
                # if is_duplicate:
                #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: Duplicate detected!")
                #     continue
                
                # Append additional triangulations
                tips_3d = np.vstack([tips_3d, p_tip_3d])
                orientations_3d = np.vstack([orientations_3d, dir_3d])
                filtered_distance_matrix = np.vstack([filtered_distance_matrix, np.array([d_tip_3d, quality_metric], dtype=float)])
                debug_data["plane_angles_deg"].append(plane_angle_deg)
                
                matched_pairs.append((i, j)) # avoid doubled triangulations in refinement step

        # Pass 2: Try to match each unmatched LAT detection against AP detections
        logging.info("Matching each unmatched LAT detection against AP detections ...")
        matched_lat_mask = np.zeros(N_lat, dtype=bool)
        matched_lat = [i for (i, j) in matched_pairs]
        matched_lat_mask[matched_lat] = True
        unmatched_lat = np.where(~matched_lat_mask)[0]
        for i in unmatched_lat:
            assert i not in row_valid

            for j in range(M_ap):
                if (i, j) in matched_pairs:
                    logging.info(f"LAT detection {i} has already been matched with AP detection {j} -> Skipping to next candidate pair in refinement step ...")
                    continue

                d_tip_3d = dist_matrix_tips[i, j]
                p_tip_3d = intersection_points_tips[i, j]

                # Skip pairing candidate if none of the two keypoint triangulations fulfills ray intersection criterion
                if mu_3d <= d_tip_3d:
                    logging.info(f"Pairing between LAT detection {i} and AP detection {j} exceeds maximum allowed 3D keypoint error: \
                                 mu_3d = {mu_3d} mm <= tip_dist_3d = {d_tip_3d:.2f} mm -> Skipping to next candidate pair in refinement step ..")
                    continue                

                # Triangulate axis for this pairing
                dir_3d, quality_metric, plane_angle_deg = compute_3d_orientation(
                    p_tip_3d, i, j, # Note: (i, j) = (lat_idx, ap_idx)
                    origins_lat_tip_kps, directions_lat_tip_kps,
                    origins_ap_tip_kps, directions_ap_tip_kps,
                    origins_lat_dir_kps, directions_lat_dir_kps,
                    origins_ap_dir_kps, directions_ap_dir_kps,
                    method, parallel_eps, plane_angle_switch_deg, icr_opts
                )                
            
                # # (Optional:) Consistency check between triangulated 3D result and 2D detection
                # unmatched_tip_2d = tip_kps_lat_view[i]
                # unmatched_dir_2d = dir_kps_lat_view[i]
                # proj_mat_unmatched_view = proj_mat_pfw_lat

                # # Check if projected 3D tip of matching candidate is close to unmatched 2D tip prediction
                # proj_tip_kp_uv, proj_dir_kp_uv = _project_line_to_view(p_tip_3d, dir_3d, proj_mat_unmatched_view)
                # tip_kp_distance_2d = np.linalg.norm(unmatched_tip_2d - proj_tip_kp_uv)
                # if kp_2d_thresh_px < tip_kp_distance_2d:
                #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: tip_kp_dist_2d = {tip_kp_distance_2d:.2f}px]")
                #     continue # too far, skip this candidate
                
                # # Check if projected 3D axis of matching candidate is close to unmatched 2D axis prediction
                # unmatched_axis_3d_to_2d = proj_tip_kp_uv - proj_dir_kp_uv
                # unmatched_axis_2d = unmatched_tip_2d - unmatched_dir_2d
                # axis_angle_2d = _angle_between_vectors_deg(unmatched_axis_3d_to_2d, unmatched_axis_2d)
                # if axis_2d_thresh_deg < axis_angle_2d:
                #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: ax_ang_2d = {axis_angle_2d:.2f}°")
                #     continue # too divergent, skip this candidate

                print(f"\t\tRefinement Step: Accepted previously unmatched pairing between LAT detection {i} and AP detection {j} \
                        (kp_quality_3d = {d_tip_3d:.2f} mm, ax_quality_3d = {quality_metric:.2f} mm)")

                # # (Optional:) Duplicate check with already triangulated implant instances
                # is_duplicate = False
                # for existing_tip_3d, existing_head_3d in zip(tips_3d, orientations_3d):
                #     if _is_duplicate_directional_implant(existing_tip_3d, existing_head_3d, p_tip_3d, dir_3d, duplicate_kp_thresh_mm, duplicate_angle_thresh_deg):
                #         is_duplicate = True # set flag if at least one duplication with already existing matches is detected
                # if is_duplicate:
                #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: Duplicate detected!")
                #     continue
                
                # Append additional triangulations
                tips_3d = np.vstack([tips_3d, p_tip_3d])
                orientations_3d = np.vstack([orientations_3d, dir_3d])
                filtered_distance_matrix = np.vstack([filtered_distance_matrix, np.array([d_tip_3d, quality_metric], dtype=float)])
                debug_data["plane_angles_deg"].append(plane_angle_deg)
                
                matched_pairs.append((i, j)) # avoid doubled triangulations in refinement step
    
    return tips_3d, orientations_3d, filtered_distance_matrix, debug_data
