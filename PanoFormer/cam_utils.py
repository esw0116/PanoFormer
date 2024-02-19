import numpy as np
from scipy.interpolate import griddata


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



def sphere2erp(coord_cam, color_cam, H, W, K):
    # coord_cam: (3, N), N: # of points, (X, Y, Z)
    # color_cam: (3, N), (R, G, B)

    N = coord_cam.shape[1]
    dist = np.sqrt(np.sum(np.power(coord_cam, 2), axis=0)+1e-6)
    dist_xy = np.sqrt(np.sum(np.power(coord_cam[:2], 2), axis=0)+1e-6)

    lat = np.arcsin(coord_cam[2] / dist)
    lon = np.arccos(coord_cam[1] / dist_xy) * np.sign(coord_cam[0])

    coord_erp = np.stack([lon, lat, np.ones(N)], axis=0)
    coord_uv = np.matmul(K, coord_erp)
    u, v = coord_uv[0], coord_uv[1]
    
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)

    omni_img = griddata(coord_uv[:2].transpose(1,0), color_cam.transpose(1,0), grid, method='linear', fill_value=0).reshape(H,W,3)
    omni_depth = griddata(coord_uv[:2].transpose(1,0), dist, grid, method='linear', fill_value=0).reshape(H,W)

    omni_img = np.round(omni_img*255).astype(np.uint8)

    return omni_img, omni_depth
