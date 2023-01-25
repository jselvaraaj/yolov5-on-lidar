from os.path import join
import numpy as np
from ouster import client
from ouster import pcap
import open3d as o3d
import pcd_utils
from simple_viz_extension import Cuboid, bbox2D
from pytransform3d.transformations import transform_from
from src.pythonrobotics.Mapping.rectangle_fitting import rectangle_fitting

class PcapWrapper:

    def __init__(self,pcapPath, metadataPath=None,parent =None) -> None:
        if metadataPath is None:
            metadataPath = pcapPath

        if parent is not None:
            metadataPath = join(parent,metadataPath)
            pcapPath = join(parent,pcapPath)

        print("Loading metadata: ", metadataPath)
        with open(metadataPath, 'r') as f:
            self.metadata = client.SensorInfo(f.read())

        print("Loading pcap: ",pcapPath)
        self.source = pcap.Pcap(pcapPath, self.metadata)
    
    def __iter__(self):
        self.source_iter = iter(client.Scans(self.source))
        return self
    
    def __next__(self):
        return ScanWrapper(next(self.source_iter),self.metadata)

class ScanWrapper:
    def __init__(self,scan,metadata) -> None:
        self.scan = scan

        left_clip = 182
        right_clip = 266

        self.xyz = client.XYZLut(metadata)(self.scan.field(client.ChanField.RANGE))[:,left_clip:-right_clip,:]
        
        self.h,self.w,_ = self.xyz.shape

        self.voxel_size = 0.4
        self.radial_filter_radius = (3,float("inf"))
        self.z_threshold = np.percentile(self.xyz[:,2],25)

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

    def __getattr__(self, key):
        return getattr(self.scan,key)

        

        