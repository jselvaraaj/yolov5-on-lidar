import pickle
from ouster.sdk import viz
from ouster import client,pcap
import numpy as np
from dataclasses import dataclass
from process_2d_preds import process_scan_factory, read_preds
from viz_utils import ScanWrapper
from pathlib import Path
import typer

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


def scanToScanWrapperFactory(process_scan,metadata):
    def wrap_scan(scan):
        nonlocal process_scan, metadata
        otherScenceData = []
        otherScenceData = process_scan(ScanWrapper(scan,metadata))

        return LidarScanWrapper(scan, otherScenceData=otherScenceData)
    return wrap_scan

def scanIterToScanWrapperIter(scanIter, process_scan, metadata):
    return map(scanToScanWrapperFactory(process_scan,metadata), scanIter)

def get_source_and_metadata(pcapPath, metadataPath=None):
    if metadataPath is None:
        metadataPath = pcapPath

    print("Loading metadata: ", metadataPath)
    with open(metadataPath, 'r') as f:
        metadata = client.SensorInfo(f.read())

    print("Loading pcap: ",pcapPath)
    source = pcap.Pcap(pcapPath,metadata)

    return source, metadata

def main(pcapPath: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)
    ,metadataPath: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)
    ,preds_path: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)
    ):
    '''
    pcap_path: the pcap file

    meta_path: the metadata file of the pcap
    
    preds_path: the csv file containing the yolov5 predictions

    '''

    pcapPath = str(pcapPath)
    metadataPath = str(metadataPath)
    preds_path = str(preds_path)

    # pcapPath = r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.pcap"
    # metadataPath = r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\meta.json"
    # preds_path = r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.csv"

    source,metadata = get_source_and_metadata(pcapPath,metadataPath)

    preds = read_preds(preds_path)
    process_scan = process_scan_factory(preds)


    lidarScanViz = LidarScanVizWrapper(metadata)
    
    streamViz = viz.SimpleViz(lidarScanViz,rate=0.5)

    streamViz.run(scanIterToScanWrapperIter(iter(client.Scans(source)), process_scan,metadata))

    
if __name__ == "__main__":
    typer.run(main)
    

