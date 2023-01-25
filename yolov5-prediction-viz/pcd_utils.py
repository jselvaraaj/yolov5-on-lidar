import copy
from skspatial.objects import Plane, Points
import numpy as np
import open3d as o3d

def project_to_xy_plane(arr):
    """Project a mesh to the xy plane. This is useful for visualization of 3D
    meshes in 2D.
    """
    return np.asarray(arr)[:,2]

def get_points_between_z(points, z_threshold):
    #Assumes z_threshold is a tuple of (min,max)
    mask = np.logical_and(points[:,2] < z_threshold[0], points[:,2] > z_threshold[1])
    points = points[mask]

    return points

def remove_ground(points, dist_threshold=0.1):
    points = np.asarray(points)
    plane_eqn = fit_plane(points)
    print("plane equation: ",plane_eqn)
    points = plane_filter(points,plane_eqn, dist_threshold)
    return points, plane_eqn

def radial_filter(pts, radius,apply_before= lambda x: x):
    #radius is a tuple of (min,max)
    points = np.asarray(apply_before(pts))
    distance_from_origin = np.linalg.norm(points, axis=1)        
    mask = np.where(np.logical_and(distance_from_origin > radius[0],distance_from_origin < radius[1]))[0]
    pts = pts[mask]

    return pts


def find_radius(points):
    points = np.asarray(points)
    distance_from_origin = np.linalg.norm(points, axis=1) 
    
    return np.min(distance_from_origin), np.max(distance_from_origin)

def fit_plane(points):
    # # Add a column of ones to the points array for the constant term
    # points = np.hstack((points, np.ones((points.shape[0], 1))))

    # # Solve the least squares problem
    # a, b, c, d = np.linalg.lstsq(points, np.ones((points.shape[0],)), rcond=None)[0]

    points = Points(points)

    plane = Plane.best_fit(points)
    
    a,b,c,d = plane.cartesian()

    return a,b,c,-d

def plane_filter(points,plane_eqn, threshold_distance=0.1):

    points = np.asarray(points)
    
    # Define the coefficients of the plane
    a, b, c, d = plane_eqn

    # Calculate the distance of each point to the plane
    distances = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] - d) / np.sqrt(a**2 + b**2 + c**2)

    # Find the indices of the points that are farther away than the threshold distance
    outlier_indices = np.where(distances <= threshold_distance)

    # Remove the outlier points from the array
    points = np.delete(points, outlier_indices, axis=0)

    return points

def get_points_in_plane(plane_eqn):

    a,b,c,d = plane_eqn

    # Define a range of parameterized coordinates for x and y
    rng = np.random.default_rng(seed=1515)
    x_range = rng.integers(-25,25, size=10000)
    y_range = rng.integers(-25,25, size=10000)

    # Generate the corresponding z values using the equation of the plane
    z_values = (d - a*x_range - b*y_range) / c

    # Generate a set of points by combining the x, y, and z values
    points = np.stack((x_range, y_range, z_values), axis=-1)

    return points

def downsample(points,voxel_size):
    print("\t\tstarted downsampling")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # pcd,indx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    # pcd = pcd.select_by_index(indx)

    points = np.asarray(pcd.points)
    print("\t\tfinished downsampling")
    return points