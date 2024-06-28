import argparse
import requests
import matplotlib.pyplot as plt
import pickle

# See https://github.com/columbia-ai-robotics/PyKinect


class KinectClient:
    def __init__(self, ip='XXX.XXX.X.XXX', port=8080):
        self.ip = ip
        self.port = port

    @property
    def color_intr(self):
        return self.get_intr()

    def get_intr(self):
        return pickle.loads(requests.get(f'http://{self.ip}:{self.port}/intr').content)

    def get_rgbd(self, repeats=5):
        data = pickle.loads(requests.get(
            f'http://{self.ip}:{self.port}/pickle/{repeats}').content)
        return data['color_img'], data['depth_img']

if __name__=='__main__':
    parser = argparse.ArgumentParser("Kinect Server")
    parser.add_argument("--ip", type=str, default='0.0.0.0', help="ip")
    parser.add_argument("--port", type=int, default=1111, help="port")
    args = parser.parse_args()

    kinect = KinectClient(args.ip, args.port)
    intr = kinect.get_intr()
    camera_data = kinect.get_rgbd()[1]
    print(intr)
    print(camera_data.shape)
    plt.show(camera_data)
    plt.show()
