import numpy as np


# pfw -> pfv
def pixel_from_voxel(matrices, voxel_size_mm=0.313, volume_shape=np.array([512, 512, 512]), detector_shape=976):
    """
    This method transforms matrices straight from the Dicom header to be used with image data. Note, that both detector
    and volume coordinates are flipped. This is because numpy uses fast-last notation, and as the data is stored rows-
    first, the last coordinate is the x-axis. Because, per convention, the projection matrices assume xyz order, they
    need to be converted to zyx order to be compatible with voxel and pixel indices.
    :param _mats: matrices in shape (n, 3, 4)
    :param voxel_size_mm: voxel size in mm.
    :param volume_shape: volume side lenght in voxels
    :return: projection matrices to map a homogeneous voxel coordinate ijk to a pixel uv via: uv1 = P @ ijk1
    """
    if type(matrices) == np.ndarray:
        assert matrices.ndim == 3 and matrices.shape[1:] == (3, 4)
    v = voxel_size_mm
    uv_from_vu = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
    # this flip is needed as Siemens stores detector rows from right to left (hardware constraint)
    FLIPU = np.array([[-1, 0, detector_shape - 1],
                      [0, 1, 0],
                      [0, 0, 1]])
    xyz_from_zyx = np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])
    zyx_from_iso = np.array([[v, 0, 0, 0],
                             [0, v, 0, 0],
                             [0, 0, v, 0],
                             [0, 0, 0, 1]])
    c = (volume_shape / 2) - 0.5
    iso_from_ijk = np.array([[1, 0, 0, -c[0]],
                             [0, 1, 0, -c[1]],
                             [0, 0, 1, -c[2]],
                             [0, 0, 0, 1]])

    # calculate offset in voxels
    _mats = np.asarray([uv_from_vu @ FLIPU @ P @ xyz_from_zyx @ zyx_from_iso @ iso_from_ijk for P in matrices])
    return _mats


# pfw -> canonical
def canonical_from_pfw(matrices, detector_shape, pixel_size):
    """
    Adapt projection matrices to a canonical format (not considering input & output coordinate system)
    :param matrices: raw projection matrices from dicom header in shape (n, 3, 4)
    :param detector_shape: shape of the detector that matrices map to
    :param pixel_size: size of each pixel on the detector (it is assumed that both dimensions have same size)
    :return: canonical representation in same shape
    """
    origin_from_center = np.asarray(
        [[1, 0, -(detector_shape[0] - 1) / 2],
         [0, 1, -(detector_shape[1] - 1) / 2],
         [0, 0, 1]]
    )
    mm_from_px = np.diag([pixel_size, pixel_size, 1])

    if matrices.ndim == 3:
        matrices = np.asarray([mm_from_px @ origin_from_center @ m for m in matrices])
    else:
        matrices = mm_from_px @ origin_from_center @ matrices
    return matrices


# canonical -> pfw
def pfw_from_canonical(matrices, detector_shape, pixel_size):
    """
    Convert matrices from canonical to pixel-from-world form for a given detector discretization
    :param matrices: matrices in canonical form mapping world coordinates to detector coordinates
    :param detector_shape: shape of array to discretize the detector image
    :param pixel_size: world-dimensions of one pixel side in mm
    :return: matrices mapping from world indices onto detector indices
    """
    center_from_origin = np.asarray([[1, 0, detector_shape[0] / 2], [0, 1, detector_shape[1] / 2], [0, 0, 1]])
    px_from_mm = np.diag([1 / pixel_size, 1 / pixel_size, 1])

    if matrices.ndim == 3:
        matrices = np.asarray([center_from_origin @ px_from_mm @ m for m in matrices])
    else:
        matrices = center_from_origin @ px_from_mm @ matrices
    return matrices


# canonical -> pfv
def pfv_from_canonical(matrices,
                       pixel_size: float = 0.305, detector_shape=(976, 976),
                       voxel_size: float = 0.313, volume_shape=(512, 512, 512)):
    """
    Convert matrices from canonical to pixel-from-voxel form for a given detector and volume discretization
    :param matrices: matrices in canonical form mapping volume to detector coordinates both in (mm, centered)
    :param volume_shape: shape of array to discretize the volume
    :param voxel_size: world-dimensions of one voxel side in mm
    :param detector_shape: shape of array to discretize the detector image
    :param pixel_size: world-dimensions of one pixel side in mm
    :return: matrices mapping an index in a volume array onto detector indices
    """
    # detector side adjustments
    c = np.asarray(detector_shape) / 2 - 0.5  # (976 / 2) - 0.5
    center_from_origin = np.asarray([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]])
    px_from_mm = np.diag([1 / pixel_size, 1 / pixel_size, 1])

    # volume side adjustments
    v_mm = voxel_size
    xyz_from_iso = np.array([[v_mm, 0, 0, 0],
                             [0, v_mm, 0, 0],
                             [0, 0, v_mm, 0],
                             [0, 0, 0, 1]])

    # shift origin to array center
    c = np.asarray(volume_shape) / 2 - 0.5  # (512 / 2) - 0.5
    iso_from_ijk = np.array([[1, 0, 0, -c[0]],
                             [0, 1, 0, -c[1]],
                             [0, 0, 1, -c[2]],
                             [0, 0, 0, 1]])
    if matrices.ndim == 3:
        # multiple matrices in shape (n, 3, 4)
        matrices = np.asarray([center_from_origin @ px_from_mm @ m @ xyz_from_iso @ iso_from_ijk for m in matrices])
    else:
        # a single matrix in shape (3, 4)
        matrices = center_from_origin @ px_from_mm @ matrices @ xyz_from_iso @ iso_from_ijk
    return matrices
