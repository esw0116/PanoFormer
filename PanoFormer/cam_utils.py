import numpy as np


def uv2sphere(depth, H, W, K):
    depth = depth.reshape(H*W)
    equi_u, equi_v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    equi_u, equi_v = equi_u.reshape(H*W), equi_v.reshape(H*W)
    coord_uv = np.stack([equi_u, equi_v, np.ones(H*W)], axis=0) # (3, H*W)

    coord_erp = np.matmul(np.linalg.inv(K), coord_uv)

    lon = coord_erp[0] # longitude (-pi ~ pi)
    lat = coord_erp[1] # latitude (-pi/2 ~ pi/2)

    x = np.cos(lat) * np.sin(lon) * depth
    y = np.cos(lat) * np.cos(lon) * depth
    z = np.sin(lat) * depth

    points_cartesian = np.stack([x,y,z], axis=0) # (3, H*W)

    return points_cartesian



def sphere2erp(coord_cam):
    # coord_cam: (N, 3), N: # of points, X, Y, Z

    dist = np.sqrt(np.sum(np.power(coord_cam, 2), axis=1, keepdims=True))
    dist_xy = np.sqrt(np.sum(np.power(coord_cam[:2], 2), axis=1, keepdims=True))

    latitude = np.arcsin(coord_cam[2:] / dist)
    longitude = np.arccos(coord_cam[1:2] / dist_xy) * np.sign(coord_cam[:1])

