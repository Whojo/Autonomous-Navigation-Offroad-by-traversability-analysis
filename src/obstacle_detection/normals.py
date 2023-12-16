import numpy as np

THRESH = 20
COS_THRESH = np.cos(THRESH * np.pi / 180)

DEFAULT_NORMAL = np.array([0, 0, 1])


def parse_normals(normals):
    """
    ...
    """
    pass


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


def project_cost(cost):
    pass
