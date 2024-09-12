import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


THRESH = 20
COS_THRESH = np.cos(THRESH * np.pi / 180)

DEFAULT_NORMAL = np.array([0, 0, 1])


def parse_normals(normals):
    """
    ...
    """
    pass


def _normalised(vector):
    return vector / np.linalg.norm(vector)


def cosine_similarity(normals, ref):
    """
    ref : (3,) ndarray
    cand: (W, H, 3) ndarray

    cos_theta = cand x ref
    """
    ref = ref / np.linalg.norm(ref)
    n = np.linalg.norm(normals, axis=-1)
    inner_product = normals @ ref
    cos_theta = inner_product / n
    return np.abs(cos_theta)


def get_mask(normals, ref, thresh=COS_THRESH):
    return cosine_similarity(normals, ref) > thresh


def cost_from_normals(normals, reference_normal=DEFAULT_NORMAL, thresh=COS_THRESH):
    mask = get_mask(normals, reference_normal, thresh)
    cost = np.where(mask, 0, 1e6)
    return cost

def _sphere_fit(points):
    n = points.shape[0]
    #   Assemble the A matrix
    A = np.zeros((n, 4))
    A[:, :3] = points
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((n, 1))
    f[:, 0] = (
        (points[:, 0] * points[:, 0])
        + (points[:, 1] * points[:, 1])
        + (points[:, 2] * points[:, 2])
    )
    C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)
    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)
    center = C[:3, 0]

    return radius, center


def _normal_pcs(points):
    _, center = _sphere_fit(points)
    return _normalised(points[0] - center)


def normals_from_pointcloud(pointcloud, mode="pca"):
    """
    Computes normals from pointcloud

    @params:
        pointcloud : Nx3 array (x,y,z)

    returns:
        normals : Nx3 array (nx, ny, nz)
    """

    # Nearest neighbours using knn
    neigh = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
    pca = PCA()

    neigh.fit(pointcloud)
    dists, nns = neigh.kneighbors(
        return_distance=True
    )  # return dists to permit distance based filtering if required.
    neighbour_ids = np.hstack(
        (np.arange(pointcloud.shape[0])[:, np.newaxis], nns)
    )  # add each point as its own neighbour

    normals = np.zeros_like(pointcloud)

    for i, roi_ids in enumerate(neighbour_ids):
        normal = None
        roi = pointcloud[roi_ids]

        if mode == "pca":
            pca.fit(roi)
            normal = pca.components_[-1]

        elif mode == "sphere_fit":
            m = 10
            roi = roi[:m]  # Limiting spherefit to 10 points
            normal = _normal_pcs(roi)

        normals[i] = normal

    return normals
