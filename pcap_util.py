import numpy as np
from ouster.sdk import viz

def translation(orginal,coord):
    x = np.eye(4)
    
    x[:,3] = np.append(coord,1)

    return orginal @ x

def scaled(orginal,scales):
    x = np.eye(4)

    x[[0,1,2],[0,1,2]] = scales

    return orginal @ x

def get_axis():
    x_ = np.array([1, 0, 0]).reshape((-1, 1))
    y_ = np.array([0, 1, 0]).reshape((-1, 1))
    z_ = np.array([0, 0, 1]).reshape((-1, 1))

    axis_n = 100
    line = np.linspace(0, 1, axis_n).reshape((1, -1))

    # basis vector to point cloud
    axis_points = np.hstack((x_ @ line, y_ @ line, z_ @ line)).transpose()

    # colors for basis vectors
    axis_color_mask = np.vstack((np.full(
        (axis_n, 4), [1, 0.1, 0.1, 1]), np.full((axis_n, 4), [0.1, 1, 0.1, 1]),
                                np.full((axis_n, 4), [0.1, 0.1, 1, 1])))

    cloud_axis = viz.Cloud(axis_points.shape[0])

    cloud_axis.set_xyz(axis_points)
    cloud_axis.set_key(np.full(axis_points.shape[0], 0.5))
    cloud_axis.set_mask(axis_color_mask)
    cloud_axis.set_point_size(3)
    
    return cloud_axis

def get_bbox_cloud(bbox,points):
    translated = translation(np.eye(4),bbox.get_center())
    s = scaled(translated,bbox.get_extent())

    r = bbox.get_rotation_matrix_from_xyz(np.zeros(3))


    cloud = viz.Cloud(points.shape[0])

    cloud.set_xyz(points)
    cloud.set_key(np.full(points.shape[0],1))

    print(np.asarray(bbox.get_box_points()))

    box_cloud = viz.Cloud(8)

    box_cloud.set_xyz(np.asarray(bbox.get_box_points()))
    box_cloud.set_key(np.full(8,0.5))
    box_cloud.set_point_size(10)

    return viz.Cuboid(s, (1, 0, 0, 0.5)), cloud, box_cloud
