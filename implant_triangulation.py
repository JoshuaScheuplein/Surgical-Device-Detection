import copy
import json
import time
import argparse
from pathlib import Path

import tifffile
import numpy as np

import cv2
import matplotlib.pyplot as plt

from utils.preprocessing import preprocessing
from utils.projection_matrix import ProjectionMatrix

from triangulation.ray_intersection_triangulation import ray_intersection_triangulation
from triangulation.plane_intersection_triangulation import plane_intersection_triangulation


# maximum ray intersection distance in 3D [mm]
# (i.e., triangulation ray consistency threshold)
mu_3d = 5.0


# define (fixed) projection parameters
detector_shape = (976, 976)
pixel_size = 0.305
volume_shape = (512, 512, 512)
voxel_size = 0.313


# define plot color for each implant category
color_mapping = {
    "screw"         : [1.0, 0.1, 0.1], # Red
    "tulip_short"   : [0.0, 0.8, 1.0], # Cyan
    "tower"         : [0.6, 0.2, 1.0], # Violet
    "tulip_long"    : [0.0, 1.0, 0.4], # Green
    "wire"          : [1.0, 0.7, 0.0], # Orange
}


def perform_triangulation(model_type: str, detection_type: str, view_pairs: list, enable_refinement=False):

    repo_dir = Path(__file__).resolve().parent
    gt_dir = repo_dir / "assets" / "ground_truth"
    image_dir = repo_dir / "assets" / "input_images"

    # load ground truth annotations in COCO format
    coco_gt_file = gt_dir / "implant_detection_simulated_test_gt.json"
    with open(coco_gt_file, 'r') as file:
        coco_annotations = json.load(file)
        categories = coco_annotations["categories"]
        coco_annotations = coco_annotations["annotations"]

    # determine unique scene names on which the model has performed inference
    prediction_dir = repo_dir / "assets" / "model_predictions" / model_type.replace("-", "_")
    scene_names = []
    for item in prediction_dir.iterdir():
        if item.is_dir():
            scene_names.append(item.name)
    print(f"\nFound {len(scene_names)} unique 3D scans ...")

    # create output directories
    output_dir = repo_dir / "assets" / "triangulation_results"
    output_dir.mkdir(parents=False, exist_ok=True)
    output_dir = output_dir / model_type
    output_dir.mkdir(parents=False, exist_ok=True)

    # create flag indicating whether to use ground truth or model predictions for triangulation
    use_gt_labels = True if detection_type == "ground_truth" else False

    # iterate over all scenes in the subset to be evaluated
    for s, scene in enumerate(scene_names):
        start_time_scene = time.perf_counter()
        print(f"\nEvaluating scan '{scene}' ({s+1}/{len(scene_names)}):")

        # generate scene output directory
        scene_out_dir = output_dir / scene
        scene_out_dir.mkdir(parents=False, exist_ok=True)

        # load projection matrices
        proj_mat_file = repo_dir / "assets" / "projection_matrices" / "C_arm_Proj_Mats.xml"
        projection_matrices = ProjectionMatrix.from_xml(proj_mat_file, detector_shape, pixel_size)

        # convert matrices to map world coordinates to pixels
        projection_matrices = projection_matrices.to_pixel_from_world(detector_shape, pixel_size)
        assert projection_matrices.mapping == 'pixel_from_world'

        # compute actual 3x4 projection matrices
        proj_mats_pfw = copy.deepcopy(projection_matrices).as_matrix() # (num_views, 3, 4)

        # compute inverse of projection matrices
        source_points, invKRs = copy.deepcopy(projection_matrices).as_point_and_ray()
        assert source_points.shape[1] == (3) and invKRs.shape[1:] == (3, 3) # (num_views, 3) and (num_views, 3, 3)

        # iterate over all view pairs of the current 3D scan
        for p, (view_idx_1, view_idx_2) in enumerate(view_pairs):
            start_time_view = time.perf_counter()
            print(f"\n\tProcessing view pair {p+1:03d}/{len(view_pairs):03d} (View 1 = {view_idx_1:03d} and View 2 = {view_idx_2:03d}) of scene '{scene}':")
            
            # load projection images and matrices     
            proj_image_1_file_path = image_dir / scene / f"{scene}_v{view_idx_1:03d}.tiff"
            proj_image_1 = tifffile.imread(proj_image_1_file_path).astype(np.float32) # (976, 976)
            proj_image_1 = preprocessing(proj_image_1, clahe=False, subtract_lowpass=False, return_uint8=False)
            assert proj_image_1.dtype == np.float32 and proj_image_1.shape == (976, 976)

            proj_image_2_file_path = image_dir / scene / f"{scene}_v{view_idx_2:03d}.tiff"
            proj_image_2 = tifffile.imread(proj_image_2_file_path).astype(np.float32) # (976, 976)
            proj_image_2 = preprocessing(proj_image_2, clahe=False, subtract_lowpass=False, return_uint8=False)
            assert proj_image_2.dtype == np.float32 and proj_image_2.shape == (976, 976)

            # collect 2D model predictions
            model_pred_2d = {"view_1" : [], "view_2" : []}

            predictions_file_path_1 = prediction_dir / scene / "JSON-Files" / f"{scene}_v{view_idx_1:03d}_pred.json"
            with open(predictions_file_path_1, 'r') as file:
                predictions_1 = json.load(file)
            for pred in predictions_1:
                assert pred["image_id"] == f"{scene}_v{view_idx_1:03d}"
                prediction = {"category_id" : pred["category_id"],
                              "keypoints"   : np.array(pred["keypoints"]).flatten().tolist()}
                model_pred_2d["view_1"].append(prediction)

            predictions_file_path_2 = prediction_dir / scene / "JSON-Files" / f"{scene}_v{view_idx_2:03d}_pred.json"
            with open(predictions_file_path_2, 'r') as file:
                predictions_2 = json.load(file)
            for pred in predictions_2:
                assert pred["image_id"] == f"{scene}_v{view_idx_2:03d}"
                prediction = {"category_id" : pred["category_id"],
                              "keypoints"   : np.array(pred["keypoints"]).flatten().tolist()}
                model_pred_2d["view_2"].append(prediction)

            print(f"\t\tFound {len(model_pred_2d['view_1'])} 2D object [predictions] for view {view_idx_1}")
            print(f"\t\tFound {len(model_pred_2d['view_2'])} 2D object [predictions] for view {view_idx_2}")

            # collect 2D COCO ground truth annotations
            coco_gt_2d = {"view_1" : [], "view_2" : []}
            
            for annotation in coco_annotations:
                if annotation["image_id"] == f"{scene}_v{view_idx_1:03d}":
                    coco_gt_2d["view_1"].append(annotation)
                if annotation["image_id"] == f"{scene}_v{view_idx_2:03d}":
                    coco_gt_2d["view_2"].append(annotation)

            print(f"\t\tFound {len(coco_gt_2d['view_1'])} 2D object [annotations] for view {view_idx_1}")
            print(f"\t\tFound {len(coco_gt_2d['view_2'])} 2D object [annotations] for view {view_idx_2}")            

            ##########################################################################################################################################
            if use_gt_labels:
                detections, preview_img_suffix = coco_gt_2d, "gt"       # use the 2D COCO ground truth annotations (as sanity check) for triangulation
            else:
                detections, preview_img_suffix = model_pred_2d, "pred"  # use the actual model predictions for triangulation
            ##########################################################################################################################################

            # initialize preview plot for verifying triangulation results
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), frameon=True)
            ax[1,0].imshow(proj_image_1, cmap='gray')
            ax[1,1].imshow(proj_image_2, cmap='gray')

            if use_gt_labels:
                view_1_preview_file = gt_dir / scene / f"{scene}_v{view_idx_1:03d}_gt.png"
                view_2_preview_file = gt_dir / scene / f"{scene}_v{view_idx_2:03d}_gt.png"
            else:
                view_1_preview_file = prediction_dir / scene / "PNG-Files" / f"{scene}_v{view_idx_1:03d}_pred.png"
                view_2_preview_file = prediction_dir / scene / "PNG-Files" / f"{scene}_v{view_idx_2:03d}_pred.png"

            view_1_preview_img = cv2.imread(str(view_1_preview_file), cv2.IMREAD_COLOR)
            view_1_preview_img = cv2.resize(view_1_preview_img, (976, 976), interpolation=cv2.INTER_AREA)
            view_1_preview_img = cv2.cvtColor(view_1_preview_img, cv2.COLOR_BGR2RGB)
            ax[0,0].imshow(view_1_preview_img)
            ax[0,0].set_title("View 1", fontsize=16)

            view_2_preview_img = cv2.imread(str(view_2_preview_file), cv2.IMREAD_COLOR)
            view_2_preview_img = cv2.resize(view_2_preview_img, (976, 976), interpolation=cv2.INTER_AREA)
            view_2_preview_img = cv2.cvtColor(view_2_preview_img, cv2.COLOR_BGR2RGB)
            ax[0,1].imshow(view_2_preview_img)
            ax[0,1].set_title("View 2", fontsize=16)

            ax[0,0].set_ylabel("2D Detections", fontsize=16)
            ax[1,0].set_ylabel("3D Triangulations", fontsize=16)

            for category in categories: # i.e., screw, tulip_short, tower, tulip_long, wire
                cat_id, cat_name = category["id"], category["name"]
                print(f"\n\t\tProcessing detections for implant category '{cat_name}':")

                # gather predicted keypoints for a single implant category
                view_1_keypoints = []
                for detection in detections["view_1"]:
                    if detection["category_id"] == cat_id:
                        keypoints = np.asarray(detection["keypoints"]).reshape(4, 3)
                        # check if 2D tip keypoint lies within image boundaries (-> avoid false positive K-wires based on detections close to the image borders)
                        implant_tip_kp = keypoints[0][:2] # extract xy coords of tip keypoint
                        tip_kp_visible = np.all((0 <= implant_tip_kp) * (implant_tip_kp < detector_shape[0]))
                        if tip_kp_visible:
                            keypoints += 0.5 # detectron2 automatically subtracts 0.5px from the acutal 2D ground truth (-> pixel center transformation)
                            view_1_keypoints.append(keypoints)
                print(f"\t\tUsing {len(view_1_keypoints)} 2D '{cat_name}' object [detections] of view {view_idx_1} for triangulation")

                view_2_keypoints = []
                for detection in detections["view_2"]:
                    if detection["category_id"] == cat_id:
                        keypoints = np.asarray(detection["keypoints"]).reshape(4, 3)
                        # check if 2D tip keypoint lies within image boundaries (-> avoid false positive K-wires based on detections close to the image borders)
                        implant_tip_kp = keypoints[0][:2] # extract xy coords of tip keypoint
                        tip_kp_visible = np.all((0 <= implant_tip_kp) * (implant_tip_kp < detector_shape[0]))
                        if tip_kp_visible:
                            keypoints += 0.5 # detectron2 automatically subtracts 0.5px from the acutal 2D ground truth (-> pixel center transformation)
                            view_2_keypoints.append(keypoints)
                print(f"\t\tUsing {len(view_2_keypoints)} 2D '{cat_name}' object [detections] of view {view_idx_2} for triangulation")
                
                #################################################################
                # perform triangulation based on ray intersection
                #################################################################
                implant_kps_3d = []
                if cat_name in ["screw", "tulip_short"]:

                    # specify maximum keypoint distance and minimum implant length in mm required for filtering triangulated implant candidates
                    if cat_name == "screw":
                        min_implant_length = 30.0 
                        median_implant_length = 47.7 # computed based on samples from the training dataset
                    else:
                        min_implant_length = 5.0
                        median_implant_length = 15.0 # computed based on samples from the training dataset

                    triangulated_implants, intersection_distances = ray_intersection_triangulation(np.asarray(view_1_keypoints), np.asarray(view_2_keypoints),
                                                                                                   view_idx_1, view_idx_2, proj_mats_pfw, source_points, invKRs,
                                                                                                   mu_3d, min_implant_length, median_implant_length,
                                                                                                   enable_refinement, debug=False)
                    print(f"\t\ttriangulated {len(triangulated_implants)} '{cat_name}' implants ...")

                    # end this iteration if no 3D implants for this category have been retrieved
                    if len(triangulated_implants) == 0:
                        continue
                    else:
                        # final 3D keypoints used for generating preview plots
                        implant_kps_3d = triangulated_implants

                #################################################################
                # perform triangulation based on plane intersection
                #################################################################
                else:

                    tips_3d, orientations_3d, distance_matrix, debug_data = plane_intersection_triangulation(np.asarray(view_1_keypoints), np.asarray(view_2_keypoints),
                                                                                                             view_idx_1, view_idx_2, proj_mats_pfw, source_points, invKRs,
                                                                                                             mu_3d, method="cp", # method = ["cp", "icr", "auto"]
                                                                                                             enable_refinement=enable_refinement, debug=False)
                    print(f"\t\ttriangulated {len(tips_3d)} '{cat_name}' implants ...")

                    # end this iteration if no 3D implants for this category have been retrieved
                    if len(tips_3d) == 0:
                        continue

                    # final 3D keypoints used for generating preview plots
                    axis_length = 20.0 if cat_name == "wire" else 40.0
                    axis_kps_3d = tips_3d + axis_length * orientations_3d
                    implant_kps_3d = np.stack((tips_3d, axis_kps_3d), axis=1)                        

                ##################################################################################
                # debugging plot / sanity check (-> project 3D points onto detector again)
                ##################################################################################
                color = color_mapping[cat_name]
                assert implant_kps_3d.shape[1:] == (2, 3)

                proj_matrix_view_1 = proj_mats_pfw[view_idx_1]
                proj_matrix_view_2 = proj_mats_pfw[view_idx_2]
                assert proj_matrix_view_1.shape == (3, 4) == proj_matrix_view_2.shape

                # map 3D keypoints onto 2D detector using projection matrices
                kps_2d_view_1 = proj_matrix_view_1 @ np.pad(implant_kps_3d.reshape((-1, 3)), ((0, 0), (0, 1)), constant_values=1).T   # (3, N)
                kps_2d_view_2 = proj_matrix_view_2 @ np.pad(implant_kps_3d.reshape((-1, 3)), ((0, 0), (0, 1)), constant_values=1).T   # (3, N)

                # dehomogenization of projected 2D keypoints
                kps_2d_view_1 = (kps_2d_view_1.squeeze()[:2] / kps_2d_view_1.squeeze()[-1:]).T   # (N, 2)
                kps_2d_view_2 = (kps_2d_view_2.squeeze()[:2] / kps_2d_view_2.squeeze()[-1:]).T   # (N, 2)
                
                # (u, v) -> (x, y)
                kps_2d_view_1 = np.flip(kps_2d_view_1, axis=1)
                kps_2d_view_2 = np.flip(kps_2d_view_2, axis=1)

                # plot 2D tips and heads as well as main axes lines on view_1 projection image
                for tip, head in zip(kps_2d_view_1[::2], kps_2d_view_1[1::2]):
                    ax[1,0].plot([tip[0], head[0]], [tip[1], head[1]], c=color, linestyle='-', linewidth=1.0, alpha=0.7)
                ax[1,0].scatter(*kps_2d_view_1[::2].T,  marker='x', color=color, edgecolors='none')
                ax[1,0].scatter(*kps_2d_view_1[1::2].T, marker='o', color=color, edgecolors='none')

                # plot 2D tips and heads as well as main axes lines on view_2 projection image
                for tip, head in zip(kps_2d_view_2[::2], kps_2d_view_2[1::2]):
                    ax[1,1].plot([tip[0], head[0]], [tip[1], head[1]], c=color, linestyle='-', linewidth=1.0, alpha=0.7)
                ax[1,1].scatter(*kps_2d_view_2[::2].T,  marker='x', color=color, edgecolors='none')
                ax[1,1].scatter(*kps_2d_view_2[1::2].T, marker='o', color=color, edgecolors='none')

            # save preview plot image
            fig.tight_layout()
            output_file_path = scene_out_dir / f"Triangulation_{scene}_v1_{view_idx_1:03d}_v2_{view_idx_2:03d}_{preview_img_suffix}.png"
            plt.savefig(output_file_path, dpi=300, pad_inches=0.0)
            plt.close()

            end_time_view = time.perf_counter()
            elapsed_time_view = end_time_view - start_time_view
            print(f"\n\t\tElapsed computation time [view]: {elapsed_time_view:.3f} seconds")

        end_time_scene = time.perf_counter()
        elapsed_time_scene = end_time_scene - start_time_scene
        print(f"\nElapsed computation time [scene]: {elapsed_time_scene:.3f} seconds")

    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    elif v.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    parser = argparse.ArgumentParser('Computation of Triangulation Results', add_help=False)
    parser.add_argument('--model_type', type=str, default="Mask-R-CNN", choices=["Faster-R-CNN", "Mask-R-CNN"],
                        help='Model type used for performing inference.')
    parser.add_argument('--detection_type', type=str, default="predictions", choices=["ground_truth", "predictions"],
                        help='Flag indicating whether to use ground truth information or actual model predictions for triangulation computation.')
    parser.add_argument('--enable_refinement', type=str2bool, default=False,
                        help='Enable refinement during triangulation (for occluded implants).')
    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Computation of Triangulation Results', parents=[get_args_parser()])
    args = parser.parse_args()
    
    view_pairs = [(20, 200), (40, 220), (60, 240), (80, 260), (100, 280),
                  (120, 300), (140, 320), (160, 340), (180, 360), (200, 380)]
    print(f"\nUsing {len(view_pairs)} projection view pairs for triangulation:\n{view_pairs}")
    
    perform_triangulation(args.model_type, args.detection_type, view_pairs, args.enable_refinement)
