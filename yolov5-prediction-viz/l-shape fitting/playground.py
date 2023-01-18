from ouster import client
from ouster import pcap
from ouster.sdk import viz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os.path import join
import open3d as o3d
from pytransform3d.transformations import transform_from
from ouster_viz import Visualizer
from lidar_utils import find_radius_2d, remove_ground, radial_filter_3d
from pytransform3d.transformations import transform_from

from PythonRobotics.Mapping.rectangle_fitting import rectangle_fitting

def get_channel():
    metadata_path = join('..','..', 'data','Sample_Data','meta.json')
    pcap_path = join('..','..', 'data','Sample_Data','data.pcap')

    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    source = pcap.Pcap(pcap_path, metadata)
    xyzlut = client.XYZLut(metadata)
    scans = iter(client.Scans(source))
    scan = next(scans)
    range = scan.field(client.ChanField.RANGE)
    signal = scan.field(client.ChanField.SIGNAL)

    return metadata, xyzlut, range, signal

def get_car_coordinates():
    x1 = 260
    y1 = 70
    h = 50
    w = 140
    x2 = x1 + w 
    y2 = y1 + h
    
    destaggered_range = client.destagger(metadata,range_)

    # fig,ax = plt.subplots()
    # rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
    # ax.add_patch(rect)
    # ax.imshow(destaggered_range)
    # plt.show()

    return x1,y1,x2,y2

def plot_xyz(xyz,pose,bbox_xyz):

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

    point_viz = viz.PointViz("Testing")
    viz.add_default_controls(point_viz)
    axis = get_axis()
    point_viz.add(axis)

    cloud_xyz = viz.Cloud(xyz.shape[0] * xyz.shape[1])
    cloud_xyz.set_xyz(np.reshape(xyz, (-1, 3)))
    # cloud_xyz.set_key(ranges.ravel()) #This is for coloring not required

    n = bbox_xyz.shape[0] * bbox_xyz.shape[1]
    cloud_bbox_xyz = viz.Cloud(bbox_xyz.shape[0] * bbox_xyz.shape[1])
    cloud_bbox_xyz.set_xyz(np.reshape(bbox_xyz.T, (-1, 3)))
    cloud_bbox_xyz.set_mask(np.full((n,4),[1,0,0,0.5]))

    bbox = viz.Cuboid(pose, (0.5, 0.5, 0.5))

    point_viz.add(cloud_xyz)
    point_viz.add(bbox)
    point_viz.add(cloud_bbox_xyz)


    point_viz.update()
    point_viz.run()


#get data
metadata, xyzlut, range_, signal = get_channel()
xyz = xyzlut(range_)
x1,y1,x2,y2 = get_car_coordinates()

bbox_xyz = xyz[y1:y2,x1:x2]

bbox_xyz = bbox_xyz.reshape((-1,3))

# plt.hist(bbox_xyz[:,2], bins='auto')
# plt.show()

height_of_the_ouster_sensor = 1.8
bbox_xyz_original = remove_ground(bbox_xyz,z_threshold=(-height_of_the_ouster_sensor,float("inf")))

print("Before downsampling ",bbox_xyz_original.shape)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bbox_xyz_original))
pcd = pcd.voxel_down_sample(voxel_size=0.4)
pcd, indx = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)

bbox_xyz = np.asarray(pcd.points)
print("After downsampling ",bbox_xyz.shape)
# bbox_xyz[:,2] = 0

ox = bbox_xyz[:,0]
oy = bbox_xyz[:,1]
l_shape_fitting =  rectangle_fitting.LShapeFitting()
rects, id_sets = l_shape_fitting.fitting(ox,oy)


    # transform_from(rot_mat, bbox_center)


# print(rect.rect_c_x)
# print(rect.rect_c_y)


plt.cla()
# for stopping simulation with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.axis("equal")
plt.plot(0.0, 0.0, "*r")

# Plot range observation
for ids in id_sets:
    x = [ox[i] for i in range(len(ox)) if i in ids]
    y = [oy[i] for i in range(len(ox)) if i in ids]

    for (ix, iy) in zip(x, y):
        plt.plot([0.0, ix], [0.0, iy], "-og")

    plt.plot([ox[i] for i in range(len(ox)) if i in ids],
                [oy[i] for i in range(len(ox)) if i in ids],
                "o")
for rect in rects:
    rect.plot()

poses = []
for rect in rects:

    rect.calc_rect_contour()

    corner_x, corner_y = rect.rect_c_x, rect.rect_c_y

    c_x = (corner_x[0] + corner_x[2])/2
    c_y = (corner_y[0] + corner_y[2])/2

    p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[1],corner_y[1]])
    x_len = np.linalg.norm(p1-p2)

    p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[3],corner_y[3]])
    y_len = np.linalg.norm(p1-p2)

    theta = np.arccos(rect.a[0])

    radius = find_radius_2d([[corner_x[0],corner_y[0]],
                    [corner_x[1],corner_y[1]],
                    [corner_x[2],corner_y[2]],
                    [corner_x[3],corner_y[3]] ])
    
    obj_points = radial_filter_3d(bbox_xyz, radius)

    o3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_points),robust=True)

    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((0,0,theta))

    c_z = o3d_bbox.get_center()[2]
    z_len = o3d_bbox.extent[2]

    pose_matrix = transform_from(rot_mat, [c_x,c_y,c_z])

    scale = [x_len,y_len,z_len,1]
    pose_matrix = pose_matrix @ np.diag(scale)

    poses.append(pose_matrix)

with Visualizer() as viz:
    viz.add_xyz(xyz.reshape((-1,3)))
    for pose in poses:
        viz.add_bbox(pose)
    viz.run()


# plt.pause(100)
