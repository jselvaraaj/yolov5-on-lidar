import pickle
from ouster.sdk import viz
from ouster import client,pcap
import numpy as np
from dataclasses import dataclass

NUM_SCENE_DATA = 100

@dataclass
class bbox2D:
    x: float
    y: float
    w: float
    h: float  

@dataclass
class Cuboid:
    pose: np.ndarray

class LidarScanWrapper:
    def __init__(self, scan, otherScenceData:list=None):
        self.lidarScan = scan
        
        self.otherScenceData = otherScenceData if otherScenceData is not None else []
        # if len(self.otherScenceData) > NUM_SCENE_DATA:
        #     raise ValueError("otherScenceData must be less than or equal to " + str(NUM_SCENE_DATA))
        
    def __getattr__(self, name):
        return getattr(self.lidarScan, name)


class LidarScanVizWrapper(viz.LidarScanViz):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._cuboids = [viz.Cuboid(np.zeros((4,4)), (0.5, 0.5, 0.5)) for _ in range(NUM_SCENE_DATA)]
        for i in range(NUM_SCENE_DATA):
            self._viz.add(self._cuboids[i])

        self._bbox2D = [None for _ in range(NUM_SCENE_DATA)]
        for i in range(NUM_SCENE_DATA):
            self._viz.add(self._bbox2D[i])

        self.__draw_bbox2d()

    def __draw_bbox2d(self):
        pass

    #overrides draw in ouster.sdk.viz.LidarScanViz
    def draw(self, update: bool = True) -> bool:
        super().draw(update)

        for cuboid in self._cuboids:
            cuboid.set_transform(np.zeros((4,4)))

        numCuboids = 0
        numBbox2D = 0
        for obj in self.scan.otherScenceData:#[:NUM_SCENE_DATA]:
            if type(obj).__name__ == Cuboid.__name__:
                self._cuboids[numCuboids].set_transform(obj.pose)
                numCuboids += 1
            elif type(obj).__name__ == bbox2D.__name__:
                self._bbox2D[numBbox2D] = obj
                numBbox2D += 1
        
        self.__draw_bbox2d()


def scanToScanWrapperFactory(otherScenceDataDict):
    def wrap_scan(scan):

        otherScenceData = []
        if scan.frame_id in otherScenceDataDict:
            otherScenceData = otherScenceDataDict[scan.frame_id]

        return LidarScanWrapper(scan, otherScenceData=otherScenceData)
    return wrap_scan

def scanIterToScanWrapperIter(scanIter, otherScenceDataDict):
    return map(scanToScanWrapperFactory(otherScenceDataDict), scanIter)

def get_source_and_metadata(pcapPath, metadataPath=None):
    if metadataPath is None:
        metadataPath = pcapPath

    print("Loading metadata: ", metadataPath)
    with open(metadataPath, 'r') as f:
        metadata = client.SensorInfo(f.read())

    print("Loading pcap: ",pcapPath)
    source = pcap.Pcap(pcapPath,metadata)

    return source, metadata

if __name__ == "__main__":

    pcapPath = r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.pcap"
    metadataPath = r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\meta.json"
    viz_bbox_data_path = r".\viz_bbox_data.pickle"

    source,metadata = get_source_and_metadata(pcapPath,metadataPath)

    viz_bbox_data = {}

    with open(viz_bbox_data_path, 'rb') as handle:
        viz_bbox_data = pickle.load(handle)

    lidarScanViz = LidarScanVizWrapper(metadata)
    
    streamViz = viz.SimpleViz(lidarScanViz,rate=0.5)

    streamViz.run(scanIterToScanWrapperIter(iter(client.Scans(source)), viz_bbox_data))

    
    


