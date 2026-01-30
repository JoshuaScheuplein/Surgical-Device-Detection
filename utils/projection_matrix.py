from pathlib import Path
from typing import Union
from collections import OrderedDict

import pydicom
import xmltodict
import numpy as np
from scipy.spatial.transform import Rotation

from utils.coordinate_systems import pfv_from_canonical
from utils.coordinate_systems import canonical_from_pfw, pfw_from_canonical


class ProjectionMatrix:
    def __init__(self, P_canonical: np.ndarray):
        self.P = P_canonical
        self.detector_shape, self.pixel_size = None, None
        self.volume_shape, self.voxel_size = None, None
        self.affine_world = np.eye(4)  # optional world transform
        self.mapping_choices = ['pixel_from_world', 'pixel_from_voxel', 'canonical']
        self.mapping = 'canonical'

    @classmethod
    def from_xml(cls, path: Union[Path, str], detector_shape=(976, 976), pixel_size=0.305):
        """
        Loads a projection matrix from XML format.
        :param path: specifies the path to an .xml file that contains one or multiple projection matrices
        :param detector_shape: shape of the detector that the matrices map to
        :param pixel_size: size of each pixel on the detector (it is assumed that both dimensions have same size)
        :return: instantiation of a projection matrix
        """
        # input needs to be transformed with from_pixel_from_world
        assert str(path).endswith('.xml')
        with open(path) as fd:
            contents = xmltodict.parse(fd.read())
            matrices = contents['hdr']['ElementList']['PROJECTION_MATRICES']

            # backprojection projection_matrices_rel_path project 2d-projections_rel_path into 3d-space
            # (homogenous coordinates allow for translation)
            proj_mat = np.zeros((len(matrices.keys()), 3, 4))

            # parsing the ordered dict to numpy projection_matrices_rel_path
            for i, key in enumerate(matrices.keys()):
                value_string = matrices[key]

                # expecting 12 entries to construct 3x4 matrix in row-major order (C-contiguous)
                proj_mat[i] = np.array(value_string.split(" "), order='C').reshape((3, 4))

            res = cls.from_pixel_from_world(proj_mat, detector_shape, pixel_size)
            # multiplying with this due to siemens standard (compensate siemens detector flip)
            FLIPU = np.array([[-1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

            res.P = np.asarray([FLIPU @ m for m in res.P]) if res.P.ndim == 3 else FLIPU @ res.P
            return res

    @classmethod
    def from_dicom(cls, path: Path, pixel_size):
        """
        Loads a projection matrix from a dicom file.
        :param path: specifies the path to a dicom file that contains one or multiple projection matrices
        :param pixel_size: size of each pixel on the detector (it is assumed that both dimensions have same size)
        :return: instantiation of a projection matrix
        """
        # input needs to be transformed with from_pixel_from_world
        # Load the DICOM file
        ds = pydicom.dcmread(path)
        assert "3DSCAN" in ds.ImageType, "Make sure that the given path leads to a series of projections."
        projmat = np.frombuffer(ds[0x001710f6]._value, dtype=np.float64)
        projmat_final = np.reshape(projmat, (400, 3, 4)).astype(np.float32)
        detector_shape = (ds.Rows, ds.Columns)
        res = cls.from_pixel_from_world(projmat_final, detector_shape, pixel_size)
        return res

    @classmethod
    def from_params(cls, rot_ang: np.ndarray, rot_orb: np.ndarray, sdd=1164, sid=622):
        """
        Creates a projection matrix from individual parameters.
        :param rot_ang: angular rotation of source point in degrees
        :param rot_orb: orbital rotation of source point in degrees
        :param sdd: source-detector distance
        :param sid: source-isocenter distance
        :return: instantiation of a projection matrices from given rotations and intrinsics
        """
        assert np.all((rot_ang >= -30) & (rot_ang <= 30)), "Rotation angle must lie between -30 and 30 degrees!"
        assert len(rot_ang) == len(rot_orb), "Number of rotation and angulation angles must be the same."
        n_views = len(rot_ang)

        # construct camera intrinsic K
        K = np.array([[sdd, 0, 0],
                      [0, sdd, 0],
                      [0, 0, 1]])  # focal length is twice the iso center distance, pierce point is center of detector

        # flip xyz to zxy and uv to vu convention (numpy vs world coordinates, numpy is row-major order)
        ###########################################################################################################
        Rw = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) # Original code, but xyz -> zxy
        # Rw = np.asarray([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) # Adapted, such that xyz -> zyx
        ###########################################################################################################
        R_det = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

        # construct extrinsics from rotation R and translation t (https://ksimek.github.io/2012/08/22/extrinsic/)
        t = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, sid], [0, 0, 0, 1]])

        # intrinsic rotations (XY vs. xy) https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        _P = np.zeros((n_views, 3, 4))
        for i in range(n_views):
            R = np.eye(4)
            R[:3, :3] = Rotation.from_euler('XY', [-rot_orb[i], -rot_ang[i]], degrees=True).as_matrix()

            # final projection matrix in mm, iso-center (canonical)
            _P[i] = R_det @ K @ np.eye(3, 4) @ t @ R @ Rw

        return cls(_P)

    @classmethod
    def from_pixel_from_world(cls, P, detector_shape, pixel_size):
        """
        Transforms one or multiple projection matrices from the pixel-from-world format to canonical
        :param P: projection matrix in pixel-from-world format
        :param detector_shape: shape of the detector that P maps to
        :param pixel_size: size of a pixel on the detector
        :return: projection matrices in canonical format
        """
        _P = canonical_from_pfw(P, detector_shape, pixel_size)
        tmp = cls(_P)
        tmp.detector_shape = detector_shape
        tmp.pixel_size = pixel_size
        return tmp

    def save_to_xml(self, path: Path):
        """
        Saves the projection matrices to an xml file with similar structure as the INCA xml files.
        :param path: specifies the path to the xml file
        """
        assert path.suffix == ".xml", "File must have .xml extension"

        # generate matrix representation
        if self.mapping == "pixel_from_world":
            print("Converting to canonical (siemens convention, pixel-from-world) format")
        arr = self.to_pixel_from_world().as_matrix()

        # Flatten each 3x4 matrix to 12 values, convert to space-separated strings
        projection_matrices = {
            f"M{i}": " ".join(f"{v:.10f}" for v in arr[i].flatten())
            for i in range(arr.shape[0])
        }

        # Set I0 values to 1.000000
        i0_values = {
            f"D{i:04d}": "1.000000"
            for i in range(arr.shape[0])
        }

        # Build the dictionary
        xml_dict = OrderedDict({
            "hdr": {
                "m_cadStudyUID": "no-study-id",
                "HeaderData": {
                    "m_XCUData": {
                        "nKV": 1109
                    }
                },
                "ElementList": {
                    "PROJECTION_MATRICES": projection_matrices,
                    "I0_VALUES": i0_values
                }
            }
        })

        # Generate the XML string
        xml_str = xmltodict.unparse(xml_dict, pretty=True)

        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)

    # change mapping specification to determine outputs
    def set_affine_world(self, affine_world: np.ndarray):
        """
        Sets affine transform to be applied to all projection matrices
        :param affine_world: affine matrix in canonical format
        """
        assert affine_world.shape == (4, 4), "Given matrix does not have correct format!"
        self.affine_world = affine_world

    def to_pixel_from_voxel(self,
                            detector_shape, pixel_size,
                            volume_shape, voxel_size):
        """
        Changes the requested mapping to 'pixel_from_voxel' and stores all necessary parameters.
        """
        self.mapping = 'pixel_from_voxel'
        self.detector_shape, self.pixel_size = detector_shape, pixel_size
        self.volume_shape, self.voxel_size = volume_shape, voxel_size
        return self

    def to_pixel_from_world(self, detector_shape=None, pixel_size=None):
        """
        Changes the requested mapping to 'pixel_from_world' and stores all necessary parameters.
        """
        # change mapping definition
        self.mapping = 'pixel_from_world'

        # set detector shape if provided
        if detector_shape is not None:
            self.detector_shape = detector_shape
        else:
            assert self.detector_shape is not None, "if detector_shape is not supplied here it must be set before"

        # set pixel_size if provided
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            assert self.pixel_size is not None, "if detector_shape is not supplied here it must be set before"

        return self

    def to_canonical(self):
        """
        Changes the requested mapping to 'canonical'.
        """
        self.mapping = 'canonical'
        return self

    def as_matrix(self):
        """
        Transforms matrices from canonical to the requested format
        :return: the matrices stored in self.P in the format stored in self.mapping
        """
        # assert that mapping specification is correct
        assert self.mapping in ['canonical', 'pixel_from_voxel', 'pixel_from_world'], "Mapping specification incorrect!"

        vu_from_uv = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, 1]])
        if self.P.ndim == 3:
            _P = np.asarray([vu_from_uv @ m @ self.affine_world for m in self.P])
        else:
            _P = vu_from_uv @ self.P @ self.affine_world

        # transform the matrix/matrices P according to the mapping set in self.mapping
        if self.mapping == 'canonical':
            return _P
        elif self.mapping == 'pixel_from_voxel':
            assert self.pixel_size is not None and self.voxel_size is not None and self.detector_shape is not None and \
                   self.volume_shape is not None, "Required parameters must be set before convertion to " \
                                                  "pixel_from_voxel is possible!"

            pfv_mat = pfv_from_canonical(_P, self.pixel_size, self.detector_shape,
                                         self.voxel_size, self.volume_shape)
            return pfv_mat
        elif self.mapping == 'pixel_from_world':
            assert self.pixel_size is not None and self.detector_shape is not None, "Required parameters must be set " \
                                                                                    "before convertion to " \
                                                                                    "pixel_from_world is possible! "
            pfw_mat = pfw_from_canonical(_P, self.detector_shape, self.pixel_size)
            return pfw_mat

    def as_point_and_ray(self):
        """
        Changes format of self.P into source points and rays
        :return: self.P in the format of source points and rays from these points
        """
        # get matrix in requested format
        matrices = self.as_matrix()

        # compute point and ray
        if matrices.ndim == 3:
            source_points = np.asarray([-np.linalg.inv(m[:3, :3]) @ m[:, 3] for m in matrices])
            invAR = np.asarray([np.linalg.inv(m[:3, :3]) for m in matrices])
        else:
            source_points = np.asarray(-np.linalg.inv(matrices[:3, :3]) @ matrices[:, 3])
            invAR = np.asarray(np.linalg.inv(matrices[:3, :3]))

        return source_points, invAR
