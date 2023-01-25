import copy

import numpy as np


def project_to_xy_plane(arr):
    """Project a mesh to the xy plane. This is useful for visualization of 3D
    meshes in 2D.
    """
    return np.asarray(arr)[:,2]

def remove_ground(points, z_threshold=(-0.1,float("inf"))):
    mask = np.logical_and(points[:,2] > z_threshold[0], points[:,2] < z_threshold[1])
    points = points[mask]

    return points

def radial_filter_3d(pts, radius):
    points = np.asarray(pts)[:,:2]
    distance_from_origin = np.linalg.norm(points, axis=1)        
    mask = np.where(np.logical_and(distance_from_origin > radius[0],distance_from_origin < radius[1]))[0]
    pts = pts[mask]

    return pts

def find_radius_2d(points):
    points = np.asarray(points)
    distance_from_origin = np.linalg.norm(points, axis=1) 
    
    return np.min(distance_from_origin), np.max(distance_from_origin)