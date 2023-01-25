import cv2
from ouster.sdk import viz
from ouster import client
import numpy as np
import time
from threading import Thread

class Visualizer:
    def __init__(self,name="Viz",use_pcd=False, unique_color = False):
        self.name=name
        self.use_pcd = use_pcd
        self.unique_color = unique_color
        self.objs = []
        self.num_geo = 0

    def __enter__(self):

        self.viewer = viz.PointViz(self.name)
        self.add_axis()
        self.add_grid()
        viz.add_default_controls(self.viewer)

        return self

    def add_grid(self):
        a,b,c,d = 0,0,1,0

        # Define a range of parameterized coordinates for x and y
        rng = np.random.default_rng(seed=1515)
        x_range = rng.integers(-25,25, size=10000)
        y_range = rng.integers(-25,25, size=10000)

        # Generate the corresponding z values using the equation of the plane
        z_values = (d - a*x_range - b*y_range) / c

        # Generate a set of points by combining the x, y, and z values
        points = np.stack((x_range, y_range, z_values), axis=-1)

        self.add_xyz(points)

    def add_axis(self):
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
        self.viewer.add(get_axis())

    def __add_image(self, scan,img,two_d_coord,position ="top"):
        # creating Image viz elements
        r_img = viz.Image()
        if two_d_coord:
            for x,y,w,h in two_d_coord:
                r_img.add_rectangle(x,y,w,h)
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        r_img.set_image(img)
        if position == "top":
            # top center position
            r_img.set_position(-scan.img_screen_len / 2, scan.img_screen_len / 2,
                                1 - scan.img_screen_height, 1)
        else:
            #bottom center position
            r_img.set_position(-scan.img_screen_len / 2, scan.img_screen_len / 2, -1,
                            -1 + scan.img_screen_height)

        self.viewer.add(r_img)

    def add_scan(self,scan,two_d_coord = None):
        # print(xyz.shape)

        if not self.use_pcd:
            cloud_scan = viz.Cloud(scan.meta)
            cloud_scan.set_range(scan.field(client.ChanField.RANGE))
            cloud_scan.set_key(scan.field(client.ChanField.SIGNAL))
            self.viewer.add(cloud_scan)
        else:
            xyz = pcd_to_numpy(scan.pcd)
            cloud_scan = viz.Cloud(xyz.shape[0])
            cloud_scan.set_xyz(np.ravel(xyz.T))

            if self.unique_color:
                cloud_scan.set_mask(np.full((xyz.shape[0],4),[*Visualizer.palatte[self.num_geo % len(Visualizer.palatte)],0.5]))
            else:
                # cloud_scan.set_key(scan.signal)
                # mask = np.full((xyz.shape[0],4),[*Visualizer.palatte[self.num_geo],0.5])
                # mask[:,3] = scan.signal
                # print(scan.signal)
                # cloud_scan.set_mask(mask)
                pass
            self.viewer.add(cloud_scan)
            self.num_geo += 1

        self.__add_image(scan,scan.range,two_d_coord,"top")
        self.__add_image(scan,scan.near_ir,two_d_coord, "bottom")

        self.viewer.update()
        self.objs.append(cloud_scan)
        return cloud_scan

    def add_xyz(self,xyz):
        cloud_scan = viz.Cloud(xyz.shape[0])
        cloud_scan.set_xyz(np.ravel(xyz.T))

        # cloud_scan.set_mask(np.full((xyz.shape[0],4),[*Visualizer.palatte[self.num_geo % len(Visualizer.palatte)],0.5]))

        self.viewer.add(cloud_scan)
        self.num_geo += 1
        self.viewer.update()
        self.objs.append(cloud_scan)


    def add_pcd(self, pcd):
        xyz = pcd_to_numpy(pcd)
        self.add_xyz(xyz)

    def add_bbox(self, pose):
        bbox = viz.Cuboid(pose, (0.5, 0.5, 0.5))

        self.viewer.add(bbox)
        self.viewer.update()
        self.objs.append(bbox)

    def clear(self):
        for obj in self.objs:
            self.viewer.remove(obj)

    def run(self):
        self.viewer.update()
        self.viewer.run()

    def remove_obj(self, obj):
        self.viewer.remove(obj)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    palatte = np.array([[1.0, 0.0, 0.0],    # Red
                  [0.0, 1.0, 0.0],    # Green
                  [0.0, 0.0, 1.0],    # Blue
                  [0.5, 0.0, 0.0],    # Maroon
                  [0.0, 0.5, 0.0],    # Forest Green
                  [0.0, 0.0, 0.5],    # Navy Blue
                  [0.5, 0.5, 0.5],    # Gray
                  [1.0, 1.0, 0.0],    # Yellow
                  [1.0, 0.5, 0.0],    # Orange
                  [0.5, 1.0, 0.0],    # Lime
                  [0.5, 0.5, 1.0],    # Light Blue
                  [1.0, 0.0, 0.5],    # Purple
                  [0.0, 1.0, 0.5],    # Turquoise
                  [1.0, 1.0, 1.0],    # White
                  [0.0, 0.0, 0.0],    # Black
                  [1.0, 0.27, 0.0],   # Deep Orange
                  [0.2, 0.6, 0.8],    # Light Sky Blue
                  [1.0, 0.72, 0.77],  # Pink
                  [0.6, 0.6, 0.6],    # Gray
                  [0.2, 0.2, 0.2],    # Dark Gray
                  [0.9, 0.9, 0.9],    # Light Gray
                  [0.6, 0.8, 1.0],    # Light Steel Blue
                  [1.0, 0.84, 0.0],   # Golden
                  [0.0, 0.65, 0.31],  # Dark Olive Green
                  [0.66, 0.66, 0.66], # Silver
                  [0.74, 0.83, 0.9],  # Light Grey
                  [0.78, 1.0, 0.86],  # Pale Green
                  [1.0, 0.0, 1.0],    # Magenta
                  [0.0, 1.0, 1.0],    # Cyan
                  [0.89, 0.47, 0.76], # Lavender
                  [1.0, 0.75, 0.8],   # Peach
                  [0.76, 0.87, 0.78], # Light Olive
                  [0.93, 0.93, 0.93], # Gainsboro
                  [0.8, 0.8, 0.8],    # Dark Gray
                  [1.0, 0.94, 0.0],   # Yellow
                  [0.0, 0.98, 0.6],   # Lime Green
                  [0.49, 0.48, 0.47], # Dark Gray
                  [0.9, 0.0, 1.0],    # Lavender
                  [0.0, 0.9, 0.9],    # Mint Green
                  [0.8, 0.4, 0.0],    # Sienna
                  [0.7, 0.0, 0.7],    # Dark Violet
                  [0.9, 0.6, 0.6],    # Light Coral
                  [0.6, 0.2, 0.8],    # Purple
                  [0.7, 0.5, 0.9],    # Lavender Blush
                  [0.8, 0.8, 0.0],    # Olive
                  [0.2, 0.6, 0.2],    # Dark Sea Green
                  [0.9, 0.9, 0.9],    # Gainsboro
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.6, 0.6, 0.6],    # Dim Gray
                  [0.2, 0.2, 0.2],    # Black
                  [0.9, 0.6, 0.0],    # Dark Orange
                  [0.9, 0.9, 0.0],    # Dark Khaki
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.8, 0.8, 0.8],    # Light Gray
                  [0.0, 0.0, 0.9],    # Dark Blue
                  [0.6, 0.6, 0.6],    # Gray
                  [0.9, 0.3, 0.3],    # Indian Red
                  [0.2, 0.8, 0.2],    # Lime Green
                  [0.7, 0.7, 0.7],    # Gray
                  [0.6, 0.2, 0.2],    # Maroon
                  [0.2, 0.2, 0.6],    # Medium Blue
                  [0.7, 0.4, 0.4],    # Rosy Brown
                  [0.9, 0.7, 0.9],    # Lavender
                  [0.3, 0.3, 0.3],    # Dark Gray
                  [0.0, 0.0, 0.6],    # Navy
                  [0.8, 0.2, 0.2],    # Brown
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.6, 0.6, 0.6],    # Gray
                  [0.2, 0.2, 0.2],    # Black
                  [0.9, 0.9, 0.0],    # Yellow
                  [0.7, 0.7, 0.7],    # Gray
                  [0.5, 0.5, 0.5],    # Gray
                  [0.6, 0.6, 0.6],    # Gray
                  [0.0, 0.5, 0.5],    # Teal
                  [0.9, 0.9, 0.9],    # White Smoke
                  [0.6, 0.2, 0.8],    # Purple
                  [0.7, 0.5, 0.9],    # Lavender Blush
                  [0.8, 0.8, 0.0],])  # Olive


    @staticmethod
    def visualize_scans(scans,**kwargs):
        with Visualizer(**kwargs) as viz:
            for scan in scans:
                viz.add_scan(scan)
            viz.run()

    @staticmethod
    def play_scans(scans,fps=20):

        def helper(viz):
            last_scan = None
            i = 0

            while viz.viewer.running():

                if i < len(scans):
                    scan = scans[i]
                    i += 1
                else:
                    i = 0
                    scan = scans[i]
                    i += 1

                if last_scan is not None:
                    viz.viewer.remove(last_scan)

                last_scan = viz.add_scan(scan)

                viz.viewer.update()

                time.sleep(1/fps)

        with Visualizer() as viz:
            thread = Thread(target=helper, args=(viz,))
            thread.start()
            viz.run()
            thread.join()

def pcd_to_numpy(pcd):
    return np.asarray(pcd.points)
