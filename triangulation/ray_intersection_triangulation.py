import logging

import numpy as np

from scipy.optimize import linear_sum_assignment

from utils.plotting import plot_rays
from utils.triangulation import find_intersecting_rays
from utils.triangulation import compute_ray_from_2d, compute_3d_orientation                                         # only used for refinement step
from utils.triangulation import _project_line_to_view, _angle_between_vectors_deg, _is_duplicate_epipolar_implant   # only used for refinement step


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


def ray_intersection_triangulation(lat_keypoints: np.ndarray, ap_keypoints: np.ndarray, lat_view_idx: int, ap_view_idx: int,
                                   projection_matrices: np.ndarray, source_points: np.ndarray, invKRs: np.ndarray,
                                   mu_3d: float, min_implant_length: float, median_implant_length: float,
                                   enable_refinement=False, debug=False):
    """
    Triangulate 3D positions of epipolar consistent implant detections from two views.

    Args:
        lat_keypoints: Array in shape (N, 4, 3) containing 2D detections from the first (lateral) view
        ap_keypoints: Array in shape (M, 4, 3) containing 2D detections from the second (anterior-posterior) view
        lat_view_idx: view index of the first (lateral) view w.r.t. the full sequence of 3D scan projections
        ap_view_idx: view index of the second (anterior-posterior) view w.r.t. the full sequence of 3D scan projections
        projection_matrices: Projection matrices in pixel from voxel mapping
        source_points: 3D source locations (i.e., ray origins) for computation of ray directions
        invKRs: Inverse of projection matrices for computation of ray directions
        mu_3d: 3D ray distance threshold below which keypoints are considered intersecting
        min_implant_length: minimum length of implant in mm

    Returns:
        implants_3d: Array in shape (L, 2, 3) containing 3D positions of intersecting implant detections,
        second dimension is for tip and head positions (tips = implants_3d[:, 0], heads = implants_3d[:, 1])
    """

    # check if model has predicted any object instances at all
    if len(lat_keypoints) == 0 or len(ap_keypoints) == 0:
        return [], []

    # keypoints.shape = (N, 4, 3) -> [...][kp_tip, kp_tip_direction, kp_head, kp_head_direction][x, y, visibility]
    tips_lat_view = np.asarray([det[0, :2] for det in lat_keypoints])
    heads_lat_view = np.asarray([det[2, :2] for det in lat_keypoints])
    tips_ap_view = np.asarray([det[0, :2] for det in ap_keypoints])
    heads_ap_view = np.asarray([det[2, :2] for det in ap_keypoints])

    # (x, y) -> (u, v)
    tips_lat_view = tips_lat_view[:, ::-1]      # (N, 2)
    heads_lat_view = heads_lat_view[:, ::-1]    # (N, 2)
    tips_ap_view = tips_ap_view[:, ::-1]        # (M, 2)
    heads_ap_view = heads_ap_view[:, ::-1]      # (M, 2)

    # compute distance of backprojected rays for detected tips individually
    dist_matrix_tips, intersection_points_tips, plot_data_tips = find_intersecting_rays(
        positions_2d_lat=tips_lat_view, positions_2d_ap=tips_ap_view,
        view_index_lat=lat_view_idx, view_index_ap=ap_view_idx,
        source_points=source_points, invKRs=invKRs
    )
    if debug:
        origins_lat_tips, directions_lat_tips, origins_ap_tips, directions_ap_tips, anchor_points_lat_tips, anchor_points_ap_tips = plot_data_tips
        plot_rays(np.expand_dims(origins_lat_tips, axis=0), np.expand_dims(directions_lat_tips, axis=0), np.expand_dims(origins_ap_tips, axis=0), np.expand_dims(directions_ap_tips, axis=0),
                  np.expand_dims(anchor_points_lat_tips.reshape(-1, 3), axis=0), np.expand_dims(anchor_points_ap_tips.reshape(-1, 3), axis=0),
                  np.expand_dims(intersection_points_tips.reshape(-1, 3), axis=0), np.expand_dims(dist_matrix_tips.reshape(-1), axis=0))

    # compute distance of backprojected rays for detected heads individually
    dist_matrix_heads, intersection_points_heads, plot_data_heads = find_intersecting_rays(
        positions_2d_lat=heads_lat_view, positions_2d_ap=heads_ap_view,
        view_index_lat=lat_view_idx, view_index_ap=ap_view_idx,
        source_points=source_points, invKRs=invKRs
    )
    if debug:
        origins_lat_heads, directions_lat_heads, origins_ap_heads, directions_ap_heads, anchor_points_lat_heads, anchor_points_ap_heads = plot_data_heads
        plot_rays(np.expand_dims(origins_lat_heads, axis=0), np.expand_dims(directions_lat_heads, axis=0), np.expand_dims(origins_ap_heads, axis=0), np.expand_dims(directions_ap_heads, axis=0),
                  np.expand_dims(anchor_points_lat_heads.reshape(-1, 3), axis=0), np.expand_dims(anchor_points_ap_heads.reshape(-1, 3), axis=0),
                  np.expand_dims(intersection_points_heads.reshape(-1, 3), axis=0), np.expand_dims(dist_matrix_heads.reshape(-1), axis=0))
    
    # prepare cost matrix C of shape (N, M)
    max_cost_value = 1e6 # 'linear_sum_assignment' does not support np.inf values as cost
    C = np.full_like(dist_matrix_tips, fill_value=max_cost_value, dtype=float)

    # filter out implants that are not intersecting properly
    implants_intersect_valid = np.logical_and(dist_matrix_heads < mu_3d, dist_matrix_tips < mu_3d)

    # filter out implants that are too short
    implant_lengths = np.linalg.norm(intersection_points_heads - intersection_points_tips, axis=2)
    implant_length_valid = min_implant_length <= implant_lengths

    # determine set of valid implant proposals
    valid_indices = np.where(np.logical_and(implants_intersect_valid, implant_length_valid))

    # solve 'Hungarian' matching problem to uniquely assign LAT-AP detection pairs
    C[valid_indices] = dist_matrix_tips[valid_indices] + dist_matrix_heads[valid_indices]
    row_ind, col_ind = linear_sum_assignment(C)

    # keep only valid assignments (with finite cost)
    valid_pairs_mask = C[row_ind, col_ind] < max_cost_value
    row_valid = row_ind[valid_pairs_mask]
    col_valid = col_ind[valid_pairs_mask]
    
    # check if at least one valid assignment could be established
    if len(row_valid) == 0:
        print("No valid assignments after Hungarian matching!")

        # diagnose the cause
        num_intersect_valid = np.count_nonzero(implants_intersect_valid)
        num_length_valid = np.count_nonzero(implant_length_valid)
        both_valid_mask = np.logical_and(implants_intersect_valid, implant_length_valid)
        num_both_valid = np.count_nonzero(both_valid_mask)
        print(f"\t\t{num_intersect_valid} candidate pairs have valid intersection ...")
        print(f"\t\t{num_length_valid} candidate pairs have valid length ...")
        print(f"\t\t{num_both_valid} candidate pairs passed both geometric filters ...")
        
        return [], []

    # this tuple is equivalent to np.where() output, so it can directly index arrays
    valid_indices = (row_valid, col_valid)

    # compute metrics for assessment of triangulation precision
    combined_intersection_distances = np.stack((dist_matrix_tips[valid_indices], dist_matrix_heads[valid_indices]), axis=0)
    logging.info(f"Mean Ray Intersection Distance\t= {np.mean(combined_intersection_distances):.3f} mm")
    logging.info(f"Std of Ray Intersection Distances\t= {np.std(combined_intersection_distances):.3f} mm")
    logging.info(f"Maximum Ray Intersection Distance\t= {np.max(combined_intersection_distances):.3f} mm")
    logging.info(f"Average Implant Length: {np.mean(implant_lengths[valid_indices]):.2f} mm")

    # generate debug plot with filtered triangulation data
    if debug:
        origins_lat = np.stack((origins_lat_tips, origins_lat_heads), axis=0)
        directions_lat = np.stack((directions_lat_tips, directions_lat_heads), axis=0)
        origins_ap = np.stack((origins_ap_tips, origins_ap_heads), axis=0)
        directions_ap = np.stack((directions_ap_tips, directions_ap_heads), axis=0)
        anchor_points_lat = np.stack((anchor_points_lat_tips[valid_indices], anchor_points_lat_heads[valid_indices]), axis=0)
        anchor_points_ap = np.stack((anchor_points_ap_tips[valid_indices], anchor_points_ap_heads[valid_indices]), axis=0)
        intersection_points = np.stack((intersection_points_tips[valid_indices], intersection_points_heads[valid_indices]), axis=0)
        dist_matrix = np.stack((dist_matrix_tips[valid_indices], dist_matrix_heads[valid_indices]), axis=0)
        plot_rays(origins_lat, directions_lat, origins_ap, directions_ap,
                  anchor_points_lat, anchor_points_ap,
                  intersection_points, dist_matrix)

    # compute 3D positions of implants
    implants_3d = np.zeros((len(valid_indices[0]), 2, 3))
    implants_3d[:, 0, :] = intersection_points_tips[valid_indices]
    implants_3d[:, 1, :] = intersection_points_heads[valid_indices]

    # compute filtered distance matrix
    filtered_distance_matrix = np.zeros((len(valid_indices[0]), 2))
    filtered_distance_matrix[:, 0] = dist_matrix_tips[valid_indices]
    filtered_distance_matrix[:, 1] = dist_matrix_heads[valid_indices]

    # final refinement step to retrieve additional 3D instances based on unmatched 2D detections from potentially occluded implants
    if enable_refinement:
        N_lat, M_ap = len(lat_keypoints), len(ap_keypoints)
        assert dist_matrix_tips.shape == (N_lat, M_ap), dist_matrix_heads.shape == (N_lat, M_ap)

        # existing matched pairs
        matched_pairs = list(zip(row_valid, col_valid))

        # compute backprojected 3D rays for all tip and head keypoints
        origins_lat_tip_kps, directions_lat_tip_kps = compute_ray_from_2d(tips_lat_view, lat_view_idx, source_points, invKRs)    # (N, 3)
        origins_ap_tip_kps, directions_ap_tip_kps = compute_ray_from_2d(tips_ap_view, ap_view_idx, source_points, invKRs)        # (M, 3)
        origins_lat_head_kps, directions_lat_head_kps = compute_ray_from_2d(heads_lat_view, lat_view_idx, source_points, invKRs) # (N, 3)
        origins_ap_head_kps, directions_ap_head_kps = compute_ray_from_2d(heads_ap_view, ap_view_idx, source_points, invKRs)     # (M, 3)

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

                d_tip_3d, d_head_3d = dist_matrix_tips[i, j], dist_matrix_heads[i, j]
                p_tip_3d, p_head_3d = intersection_points_tips[i, j], intersection_points_heads[i, j]

                # Skip pairing candidate if none of the two keypoint triangulations fulfills ray intersection criterion
                if mu_3d <= d_tip_3d and mu_3d <= d_head_3d:
                    logging.info(f"Pairing between AP detection {j} and LAT detection {i} exceeds maximum allowed 3D keypoint error: \
                                 mu_3d = {mu_3d} mm <= tip_dist_3d = {d_tip_3d:.2f} mm and mu_3d = {mu_3d} mm <= head_dist_3d = {d_head_3d:.2f} mm \
                                 -> Skipping to next candidate pair in refinement step ..")
                    continue
                
                # Use standard 'epipolar' triangulation routine if ray intersection criterion is met for both keypoints
                # (-> yields more accurate solutions for second keypoint compared to 'directional' method with fixed implant length)
                if d_tip_3d < mu_3d and d_head_3d < mu_3d:
                    pos_t, pos_h = p_tip_3d, p_head_3d
                    dist_t, dist_h = d_tip_3d, d_head_3d
                    print(f"\t\tRefinement Step: Accepted previously unmatched pairing between AP detection {j} and LAT detection {i} \
                          (tip_quality_3d = {dist_t:.2f} mm, head_quality_3d = {dist_h:.2f} mm)")

                # Use 'directional' triangulation routine if ray intersection criterion is met for a single keypoint only
                # (-> implicit assumption of fixed implant length for determining the second implant keypoint)
                else:

                    # Use keypoint with highest confidence as triangulation start point
                    if d_tip_3d <= d_head_3d:
                        start_point_3d = p_tip_3d
                        intersection_distance = d_tip_3d
                        origins_lat_start_kps, directions_lat_start_kps = origins_lat_tip_kps, directions_lat_tip_kps
                        origins_ap_start_kps, directions_ap_start_kps = origins_ap_tip_kps, directions_ap_tip_kps
                        origins_lat_dir_kps, directions_lat_dir_kps = origins_lat_head_kps, directions_lat_head_kps
                        origins_ap_dir_kps, directions_ap_dir_kps = origins_ap_head_kps, directions_ap_head_kps
                    else:
                        start_point_3d = p_head_3d
                        intersection_distance = d_head_3d
                        origins_lat_start_kps, directions_lat_start_kps = origins_lat_head_kps, directions_lat_head_kps
                        origins_ap_start_kps, directions_ap_start_kps = origins_ap_head_kps, directions_ap_head_kps
                        origins_lat_dir_kps, directions_lat_dir_kps = origins_lat_tip_kps, directions_lat_tip_kps
                        origins_ap_dir_kps, directions_ap_dir_kps = origins_ap_tip_kps, directions_ap_tip_kps
                    
                    # Triangulation settings
                    method = "cp"
                    icr_opts = {"maxiter": 200, "tol": 1e-6}

                    # Triangulate axis for this pairing
                    dir_3d, quality_metric, plane_angle_deg = compute_3d_orientation(
                        start_point_3d, i, j, # Note: (i, j) = (lat_idx, ap_idx)
                        origins_lat_start_kps, directions_lat_start_kps,
                        origins_ap_start_kps, directions_ap_start_kps,
                        origins_lat_dir_kps, directions_lat_dir_kps,
                        origins_ap_dir_kps, directions_ap_dir_kps,
                        method, parallel_eps, plane_angle_switch_deg, icr_opts
                    )

                    # Retrieve second implant 3D keypoint
                    # Note: 'dir_3d' is automatically computed in such a way that it points away from 'start_point_3d'
                    dir_3d_u = dir_3d / np.linalg.norm(dir_3d)
                    end_point_3d = start_point_3d + median_implant_length * dir_3d_u

                    if d_tip_3d <= d_head_3d:
                        pos_t, pos_h = start_point_3d, end_point_3d
                        dist_t, dist_h = d_tip_3d, quality_metric
                    else:
                        pos_t, pos_h = end_point_3d, start_point_3d
                        dist_t, dist_h = quality_metric, d_head_3d
                
                    # # (Optional:) Consistency check between triangulated 3D result and 2D detection
                    # unmatched_tip_2d = tips_ap_view[j]
                    # unmatched_head_2d = heads_ap_view[j]
                    # proj_mat_unmatched_view = proj_mat_pfw_ap

                    # # Use keypoint with highest confidence as triangulation start point
                    # if d_tip_3d <= d_head_3d:
                    #     unmatched_start_kp_2d = unmatched_tip_2d
                    #     unmatched_end_kp_2d = unmatched_head_2d
                    # else:
                    #     unmatched_start_kp_2d = unmatched_head_2d
                    #     unmatched_end_kp_2d = unmatched_tip_2d

                    # # Check if projected 3D tip of matching candidate is close to unmatched 2D tip prediction
                    # proj_start_kp_uv, proj_dir_uv = _project_line_to_view(start_point_3d, dir_3d, proj_mat_unmatched_view)
                    # start_kp_distance_2d = np.linalg.norm(unmatched_start_kp_2d - proj_start_kp_uv)
                    # if kp_2d_thresh_px < start_kp_distance_2d:
                    #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: start_kp_dist_2d = {start_kp_distance_2d:.2f}px]")
                    #     continue # too far, skip this candidate
                    
                    # # Check if projected 3D axis of matching candidate is close to unmatched 2D axis prediction
                    # unmatched_axis_3d_to_2d = proj_start_kp_uv - proj_dir_uv
                    # unmatched_axis_2d = unmatched_start_kp_2d - unmatched_end_kp_2d
                    # axis_angle_2d = _angle_between_vectors_deg(unmatched_axis_3d_to_2d, unmatched_axis_2d)
                    # if axis_2d_thresh_deg < axis_angle_2d:
                    #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: ax_ang_2d = {axis_angle_2d:.2f}°")
                    #     continue # too divergent, skip this candidate

                    print(f"\t\tRefinement Step: Accepted previously unmatched pairing between AP detection {j} and LAT detection {i} \
                          (kp_quality_3d = {intersection_distance:.2f} mm, ax_quality_3d = {quality_metric:.2f} mm)")

                # # (Optional:) Duplicate check with already triangulated implant instances
                # is_duplicate = False
                # for implant_3d in implants_3d:
                #     existing_tip_3d, existing_head_3d = implant_3d[0, :], implant_3d[0, :]
                #     if _is_duplicate_epipolar_implant(existing_tip_3d, existing_head_3d, pos_t, pos_h, duplicate_kp_thresh_mm, duplicate_angle_thresh_deg):
                #         is_duplicate = True # set flag if at least one duplication with already existing matches is detected
                # if is_duplicate:
                #     logging.info(f"Skipping pairing between AP detection {j} and LAT detection {i}: Duplicate detected!")
                #     continue
                
                # Append additional triangulations
                implants_3d = np.vstack((implants_3d, np.reshape(np.stack((pos_t, pos_h), axis=0), (1, 2, 3))))
                filtered_distance_matrix = np.vstack((filtered_distance_matrix, np.array([[dist_t, dist_h]])))
                
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

                d_tip_3d, d_head_3d = dist_matrix_tips[i, j], dist_matrix_heads[i, j]
                p_tip_3d, p_head_3d = intersection_points_tips[i, j], intersection_points_heads[i, j]

                # Skip pairing candidate if none of the two keypoint triangulations fulfills ray intersection criterion
                if mu_3d <= d_tip_3d and mu_3d <= d_head_3d:
                    logging.info(f"Pairing between LAT detection {i} and AP detection {j} exceeds maximum allowed 3D keypoint error: \
                                 mu_3d = {mu_3d} mm <= tip_dist_3d = {d_tip_3d:.2f} mm and mu_3d = {mu_3d} mm <= head_dist_3d = {d_head_3d:.2f} mm \
                                 -> Skipping to next candidate pair in refinement step ..")
                    continue
                
                # Use standard 'epipolar' triangulation routine if ray intersection criterion is met for both keypoints
                # (-> yields more accurate solutions for second keypoint compared to 'directional' method with fixed implant length)
                if d_tip_3d < mu_3d and d_head_3d < mu_3d:
                    pos_t, pos_h = p_tip_3d, p_head_3d
                    dist_t, dist_h = d_tip_3d, d_head_3d
                    print(f"\t\tRefinement Step: Accepted previously unmatched pairing between LAT detection {i} and AP detection {j} \
                          (tip_quality_3d = {dist_t:.2f} mm, head_quality_3d = {dist_h:.2f} mm)")

                # Use 'directional' triangulation routine if ray intersection criterion is met for a single keypoint only
                # (-> implicit assumption of fixed implant length for determining the second implant keypoint)
                else:

                    # Use keypoint with highest confidence as triangulation start point
                    if d_tip_3d <= d_head_3d:
                        start_point_3d = p_tip_3d
                        intersection_distance = d_tip_3d
                        origins_lat_start_kps, directions_lat_start_kps = origins_lat_tip_kps, directions_lat_tip_kps
                        origins_ap_start_kps, directions_ap_start_kps = origins_ap_tip_kps, directions_ap_tip_kps
                        origins_lat_dir_kps, directions_lat_dir_kps = origins_lat_head_kps, directions_lat_head_kps
                        origins_ap_dir_kps, directions_ap_dir_kps = origins_ap_head_kps, directions_ap_head_kps
                    else:
                        start_point_3d = p_head_3d
                        intersection_distance = d_head_3d
                        origins_lat_start_kps, directions_lat_start_kps = origins_lat_head_kps, directions_lat_head_kps
                        origins_ap_start_kps, directions_ap_start_kps = origins_ap_head_kps, directions_ap_head_kps
                        origins_lat_dir_kps, directions_lat_dir_kps = origins_lat_tip_kps, directions_lat_tip_kps
                        origins_ap_dir_kps, directions_ap_dir_kps = origins_ap_tip_kps, directions_ap_tip_kps
                    
                    # Triangulation settings
                    method = "cp"
                    icr_opts = {"maxiter": 200, "tol": 1e-6}

                    # Triangulate axis for this pairing
                    dir_3d, quality_metric, plane_angle_deg = compute_3d_orientation(
                        start_point_3d, i, j, # Note: (i, j) = (lat_idx, ap_idx)
                        origins_lat_start_kps, directions_lat_start_kps,
                        origins_ap_start_kps, directions_ap_start_kps,
                        origins_lat_dir_kps, directions_lat_dir_kps,
                        origins_ap_dir_kps, directions_ap_dir_kps,
                        method, parallel_eps, plane_angle_switch_deg, icr_opts
                    )

                    # Retrieve second implant 3D keypoint
                    # Note: 'dir_3d' is automatically computed in such a way that it points away from 'start_point_3d'
                    dir_3d_u = dir_3d / np.linalg.norm(dir_3d)
                    end_point_3d = start_point_3d + median_implant_length * dir_3d_u

                    if d_tip_3d <= d_head_3d:
                        pos_t, pos_h = start_point_3d, end_point_3d
                        dist_t, dist_h = d_tip_3d, quality_metric
                    else:
                        pos_t, pos_h = end_point_3d, start_point_3d
                        dist_t, dist_h = quality_metric, d_head_3d
                
                    # # (Optional:) Consistency check between triangulated 3D result and 2D detection
                    # unmatched_tip_2d = tips_lat_view[i]
                    # unmatched_head_2d = heads_lat_view[i]
                    # proj_mat_unmatched_view = proj_mat_pfw_lat

                    # # Use keypoint with highest confidence as triangulation start point
                    # if d_tip_3d <= d_head_3d:
                    #     unmatched_start_kp_2d = unmatched_tip_2d
                    #     unmatched_end_kp_2d = unmatched_head_2d
                    # else:
                    #     unmatched_start_kp_2d = unmatched_head_2d
                    #     unmatched_end_kp_2d = unmatched_tip_2d

                    # # Check if projected 3D tip of matching candidate is close to unmatched 2D tip prediction
                    # proj_start_kp_uv, proj_dir_uv = _project_line_to_view(start_point_3d, dir_3d, proj_mat_unmatched_view)
                    # start_kp_distance_2d = np.linalg.norm(unmatched_start_kp_2d - proj_start_kp_uv)
                    # if kp_2d_thresh_px < start_kp_distance_2d:
                    #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: start_kp_dist_2d = {start_kp_distance_2d:.2f}px]")
                    #     continue # too far, skip this candidate
                    
                    # # Check if projected 3D axis of matching candidate is close to unmatched 2D axis prediction
                    # unmatched_axis_3d_to_2d = proj_start_kp_uv - proj_dir_uv
                    # unmatched_axis_2d = unmatched_start_kp_2d - unmatched_end_kp_2d
                    # axis_angle_2d = _angle_between_vectors_deg(unmatched_axis_3d_to_2d, unmatched_axis_2d)
                    # if axis_2d_thresh_deg < axis_angle_2d:
                    #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: ax_ang_2d = {axis_angle_2d:.2f}°")
                    #     continue # too divergent, skip this candidate

                    print(f"\t\tRefinement Step: Accepted previously unmatched pairing between LAT detection {i} and AP detection {j} \
                          (kp_quality_3d = {intersection_distance:.2f} mm, ax_quality_3d = {quality_metric:.2f} mm)")

                # # (Optional:) Duplicate check with already triangulated implant instances
                # is_duplicate = False
                # for implant_3d in implants_3d:
                #     existing_tip_3d, existing_head_3d = implant_3d[0, :], implant_3d[0, :]
                #     if _is_duplicate_epipolar_implant(existing_tip_3d, existing_head_3d, pos_t, pos_h, duplicate_kp_thresh_mm, duplicate_angle_thresh_deg):
                #         is_duplicate = True # set flag if at least one duplication with already existing matches is detected
                # if is_duplicate:
                #     logging.info(f"Skipping pairing between LAT detection {i} and AP detection {j}: Duplicate detected!")
                #     continue
                
                # Append additional triangulations
                implants_3d = np.vstack((implants_3d, np.reshape(np.stack((pos_t, pos_h), axis=0), (1, 2, 3))))
                filtered_distance_matrix = np.vstack((filtered_distance_matrix, np.array([[dist_t, dist_h]])))
                
                matched_pairs.append((i, j)) # avoid doubled triangulations in refinement step
    
    return implants_3d, filtered_distance_matrix
