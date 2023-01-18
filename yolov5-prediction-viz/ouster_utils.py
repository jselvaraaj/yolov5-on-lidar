import copy
import time
from matplotlib import patches, pyplot as plt
import numpy as np
from ouster import client
from ouster import pcap
from os.path import join
import open3d as o3d
from scipy.spatial.distance import cdist
from collections import deque
from threading import Thread
from pytransform3d.transformations import transform_from

class ScanWrapper:
    def __init__(self,scan,metadata,xyzlut):
        self.scan = scan
        self.meta = metadata
        self.xyzlut = xyzlut

        range_ = self.scan.field(client.ChanField.RANGE)
        self.xyz = self.xyzlut(range_)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz.reshape((-1,3)))
        self.pcd = pcd

        self.range = self.scan.field(client.ChanField.RANGE)

        self.signal = self.scan.field(client.ChanField.SIGNAL)

        # self.signal = np.asarray(self.scan.field(client.ChanField.SIGNAL)).ravel()
        # self.signal = reject_outliers(self.signal,3)
        # self.signal = self.signal/ np.max(self.signal)

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

        return self.__get_3d_bbox(x1,y1,x2,y2)
        
    def __get_3d_bbox(self,x1,y1,x2,y2):

        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(self.xyz[y1:y2,x1:x2].reshape((-1,3))),robust=True)
        bbox.color = [0, 0,0]
        bbox_center = bbox.get_center()

        pose_matrix = transform_from(bbox.R, bbox_center)
        scale = list(bbox.extent)
        scale.append(1)

        return pose_matrix @ np.diag(scale)


class Pcap:
    def __init__(self, source, metadata,parent = None):
        
        self.pcap,self.metadata = Pcap.get_source_and_metadata(source,metadata,parent)
        self.xyzlut = client.XYZLut(self.metadata)

    def __iter__(self):
        self.scans = iter(client.Scans(self.pcap))
        self.thread = Thread(target=self.__add_frames,daemon=True)
        self.frames = deque()
        self.done = False
        self.buffer = 60 * 10

        try:
            self.frames.append(next(self.scans))
        except StopIteration:
            self.done = True
            pass

        self.thread.start()

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
        try:
            scan = self.frames.popleft()
        except IndexError:
            if self.done:
                raise StopIteration
            else:
                time.sleep(1)
                try:
                    return self.__next__()
                except RecursionError:
                    raise Exception("Timed out waiting to load pcap frames")
        return ScanWrapper(scan,self.metadata,self.xyzlut)

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

            points,signal = Pcap.clean_pcd(scan.pcd.points,scan.signal,**kwargs)
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