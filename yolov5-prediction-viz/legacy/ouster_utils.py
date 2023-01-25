import copy
import time
from l_shape_fitting.ouster_viz import Visualizer
from matplotlib import patches, pyplot as plt
import numpy as np
from ouster import client
from ouster import pcap
from ouster.sdk import viz
from os.path import join
import open3d as o3d
from scipy.spatial.distance import cdist
from collections import deque
from threading import Thread
from pytransform3d.transformations import transform_from
import pcd_utils
from l_shape_fitting.PythonRobotics.Mapping.rectangle_fitting import rectangle_fitting

from simple_viz_extension import Cuboid, bbox2D
class ScanWrapper:
    #self.pcd is cleaned but self.xyz is raw
    def __init__(self,scan,metadata,xyzlut,voxel_size=0.4,radial_filter_radius=(3,float("inf")),plane_filter_radius = 0.2):
        self.scan = scan
        self.meta = metadata
        self.xyzlut = xyzlut

        range_ = self.scan.field(client.ChanField.RANGE)
        self.xyz = self.xyzlut(range_)
        
        left_clip = 182
        right_clip = 266
        self.xyz = self.xyz[:,left_clip:-right_clip,:]

        self.radial_filter_radius = radial_filter_radius
        self.plane_filter_radius = plane_filter_radius
        self.voxel_size = voxel_size
        
        self.range = client.destagger(metadata,self.scan.field(client.ChanField.RANGE))
        self.signal = client.destagger(metadata,self.scan.field(client.ChanField.SIGNAL))
        self.near_ir = client.destagger(metadata,self.scan.field(client.ChanField.NEAR_IR))


    def process(self):
        pts = self.xyz.reshape((-1,3))
        self.z_threshold = np.percentile(pts[:,2],25)
        # pts = pcd_utils.downsample(pts,voxel_size=self.voxel_size)
        # pts = pcd_utils.radial_filter(pts,self.radial_filter_radius)

        if pts.size == 0:
            print("No points in scan")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd = pcd

        # self.__init_ground_plane()


    # def __init_ground_plane(self):
    #     pts = pcd_utils.get_points_between_z(np.asarray(self.pcd.points),z_threshold=(self.z_threshold,float("-inf")))
    #     try:
    #         self.ground_eqn = pcd_utils.fit_plane(pts)
    #     except:
    #         print("No ground plane found")

    def __getattr__(self, key):
        return getattr(self.scan,key)
    
    def __deepcopy__(self, memo):
        return ScanWrapper(copy.deepcopy(self.scan),copy.deepcopy(self.meta),copy.deepcopy(self.xyzlut))

    def process_pred(self,x,y,w,h):

        #de normalize
        x *= self.w
        y *= self.h
        w *= self.w
        h *= self.h

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x1 + w) 
        y2 = int(y1 + h)

        return self.__get_3d_bbox(x1,y1,x2,y2), bbox2D(x1,y1,w,h)

    def __get_3d_bbox(self,x1,y1,x2,y2):

        bbox_xyz = self.xyz[y1:y2,x1:x2].reshape((-1,3))
        print("\tBefore downsampling and cleaning: ",bbox_xyz.shape)
        bbox_xyz = pcd_utils.get_points_between_z(bbox_xyz ,z_threshold=(float("inf"),self.z_threshold))
        bbox_xyz = pcd_utils.radial_filter(bbox_xyz, self.radial_filter_radius)
        bbox_xyz = pcd_utils.downsample(bbox_xyz, self.voxel_size)

        print("\tAfter downsampling and cleaning: ",bbox_xyz.shape)

        # bbox_xyz = pcd_utils.plane_filter(bbox_xyz, self.ground_eqn, self.plane_filter_radius)

        # with Visualizer() as viz:
        #     viz.add_xyz(bbox_xyz)
        #     viz.run()

        ox = bbox_xyz[:,0]
        oy = bbox_xyz[:,1]
        
        print("\tstarted l_shape fitting")
        l_shape_fitting =  rectangle_fitting.LShapeFitting()
        
        rects, id_sets = l_shape_fitting.fitting(ox,oy)
        print(f"\tFound {len(rects)} rectangles")
        def rect_to_pose(rect):
            nonlocal bbox_xyz

            rect.calc_rect_contour()

            corner_x, corner_y = rect.rect_c_x, rect.rect_c_y

            c_x = (corner_x[0] + corner_x[2])/2
            c_y = (corner_y[0] + corner_y[2])/2

            p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[1],corner_y[1]])
            x_len = np.linalg.norm(p1-p2)

            p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[3],corner_y[3]])
            y_len = np.linalg.norm(p1-p2)

            theta = np.arccos(rect.a[0])

            radius = pcd_utils.find_radius([[corner_x[0],corner_y[0]],
                            [corner_x[1],corner_y[1]],
                            [corner_x[2],corner_y[2]],
                            [corner_x[3],corner_y[3]] ])
            
            obj_points = pcd_utils.radial_filter(bbox_xyz, radius,lambda x : x[:,:2])

            if obj_points.shape[0] >=4:
                o3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_points),robust=True)

                rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((0,0,theta))

                c_z = o3d_bbox.get_center()[2]
                z_len = o3d_bbox.extent[2]

                pose_matrix = transform_from(rot_mat, [c_x,c_y,c_z])

                scale = [x_len,y_len,z_len,1]
                pose_matrix = pose_matrix @ np.diag(scale)

                return pose_matrix

        poses = []
        for rect in rects:
            pose = rect_to_pose(rect)
            
            if pose is not None:
                poses.append(Cuboid(pose))

            # with Visualizer() as viz:
            #     viz.add_xyz(bbox_xyz.reshape((-1,3)))
            #     for pose in poses:
            #         viz.add_bbox(pose.pose)
            #     viz.run()

        return poses

class Pcap:
    def __init__(self, source, metadata, parent = None):
        self.pcap,self.metadata = Pcap.get_source_and_metadata(source,metadata,parent)
        self.xyzlut = client.XYZLut(self.metadata)

    def __iter__(self):
        self.scans = iter(client.Scans(self.pcap))
        return self

    def __add_frames(self):
        while True:
            try:
                for i in range(self.buffer):
                    self.frames.append(next(self.scans))
            except StopIteration:
                self.done = True
                break
            time.sleep(0.0)

    def __next__(self):
        return ScanWrapper(next(self.scans),self.metadata,self.xyzlut)
        
    def get_frames(self, innerloop, frames):
        
        if type(frames) != list:
            frames = [i for i in range(frames)]

        res = []
        frames = set(frames)
        scans = iter(client.Scans(self.pcap))
        x = 0
        started = False
        for i in range(max(frames)+1):
            scan = next(scans)    

            if i in frames:
                res.append(innerloop(scan))

                x+=1
            
                if x % 10 == 0:
                    print("\t\tLoaded frame: ", i)
                started = True

            
            if not started and i % 10 == 0:
                print("\t\tSkipping frame: ", i)
                continue

        return res

    def get_scans_at_frames(self,frames=[0]):
        def innerloop(scan):
            return ScanWrapper(scan,self.metadata,self.xyzlut)
            
        return self.get_frames(innerloop,frames)


    def get_pcd_at_frames(self,frames=[0]):

        def innerloop(scan):
            range_ = scan.field(client.ChanField.RANGE)
            xyz = self.xyzlut(range_)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1,3)))
            return pcd

        return self.get_frames(innerloop,frames)

    @staticmethod
    def get_source_and_metadata(pcap_, metadata=None,parent =None):
        if metadata is None:
            metadata = pcap_
        
            metadata = join(parent,metadata)
            pcap_ = join(parent,pcap_)

        print("Loading metadata: ", metadata)
        with open(metadata, 'r') as f:
            metadata = client.SensorInfo(f.read())

        print("Loading pcap: ",pcap_)
        source = pcap.Pcap(pcap_, metadata)

        return source, metadata

    @staticmethod
    def clean_pcd_in_scans(scans,**kwargs):
        for scan in scans:

            points,signal = Pcap.clean_pcd(np.asarray(scan.pcd.points),scan.signal,**kwargs)
            scan.signal = signal

            # pcd.points = o3d.utility.Vector3dVector(points[np.where(mask)[0]])

            scan.pcd.points = o3d.utility.Vector3dVector(points)
            
            scan.pcd = scan.pcd.voxel_down_sample(voxel_size=0.1)
            scan.pcd, indx = scan.pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
            scan.signal = scan.signal[indx]

        return scans
    
    @staticmethod
    def clean_pcd(points, signal, radius,z_threshold):
        points = np.asarray(points)

        distance_from_origin = np.linalg.norm(points, axis=1)        
        
        mask = np.where(np.logical_and(distance_from_origin > radius[0],distance_from_origin < radius[1]))[0]

        points = points[mask]
        signal = signal[mask]
        
        mask = np.logical_and(points[:,2] > z_threshold[0], points[:,2] < z_threshold[1])
        points = points[mask]
        signal = signal[mask]

        return points,signal

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    data[np.logical_and(-m>s,s>m)] = 0
    return data
