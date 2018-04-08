import argparse
import os

import numpy as np
import cv2
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError

import rosbag

from matplotlib import pyplot as plt


class Bag2CameraInfo:
    """ class that extracts camera info from bag """
    def __init__(self, path_to_bag, topic, t_max=-1):
        self.K = None
        self.D = None
        self.distortion_model = None
        self.P = None
        self.R = None
        self.msg = None

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            for topic, msg, t in bag.read_messages(topics=[topic]):
                self.msg = msg
                self.K = np.array(msg.K).reshape((3, 3))
                self.D = msg.D
                self.distortion_model = msg.distortion_model
                self.P = np.array(msg.P).reshape((3,4))
                self.R = np.array(msg.R).reshape((3,3)).astype(np.float32)
                break


class Bag2Images:
    def __init__(self, path_to_bag, topic, secs0=None, nsecs0=None, max_num_messages_to_read=-1, verbose=True):
        self.secs0 = secs0
        self.nsecs0 = nsecs0

        self.times = []
        self.images = []

        self.bridge = CvBridge()

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            total_num_msgs = bag.get_message_count(topic)
            num_msgs_to_read = max_num_messages_to_read if max_num_messages_to_read > 0 else total_num_msgs

            if verbose:
                print('Reading {} messages'.format(num_msgs_to_read))

            msg_idx = 1
            for topic, msg, t in bag.read_messages(topics=[topic]):
                self.addImage(msg)
                msg_idx += 1

                if msg_idx % 10 == 0 and verbose:
                    print('{}/{}'.format(msg_idx, num_msgs_to_read))

                if msg_idx >= num_msgs_to_read:
                    break

        if verbose:
            print('Finished reading {} messages.'.format(num_msgs_to_read))

    def addImage(self, msg):
        if self.secs0 == None:
            self.secs0 = msg.header.stamp.secs
            self.nsecs0 = msg.header.stamp.nsecs

        time = msg.header.stamp.secs - self.secs0 + 1e-9 * (msg.header.stamp.nsecs - self.nsecs0)
        self.times.append(time)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.images.append(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Generate yaml file''')
    parser.add_argument('--bag', help='Root of folders.', default='')
    parser.add_argument('--out', default="")
    parser.add_argument('--cam_info_topic', default="")
    parser.add_argument('--image_topic', default="")

    args = parser.parse_args()

    bag_path = os.path.join(os.getcwd(),args.bag)

    bag = Bag2CameraInfo(bag_path, args.cam_info_topic)
    bag_im = Bag2Images(bag_path, args.image_topic)

    c = PinholeCameraModel()
    c.fromCameraInfo(bag.msg)
    size = c.fullResolution()
    K_new, _ = cv2.getOptimalNewCameraMatrix(c.fullIntrinsicMatrix(), c.distortionCoeffs(), size, 0.0)

    pts = []
    for x in range(size[0]):
        for y in range(size[1]):
            pts.append([[x,y]])

    new_points = cv2.undistortPoints(np.array(pts,dtype=np.float32), bag.K, bag.D, P=K_new)
    map_x = np.reshape(new_points[:,0,0], newshape=(size[0], size[1])).transpose()
    map_y = np.reshape(new_points[:,0,1], newshape=(size[0], size[1])).transpose()


    # test:
    # undistort image
    img = bag_im.images[0]
    undist_img = cv2.undistort(img, bag.K, bag.D, newCameraMatrix=K_new)

    # use undist_img and map_x, map_y to get original image
    orig_img = cv2.remap(undist_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    fig,ax = plt.subplots(ncols=2)
    ax[0].set_title("original")
    ax[0].imshow(img)
    ax[1].set_title("processed")
    ax[1].imshow(orig_img)

    #plt.show()



    out_path = os.path.join(os.getcwd(), args.out)


    np.savetxt(out_path+"_x.csv", map_x, delimiter=",")
    np.savetxt(out_path+"_y.csv", map_y, delimiter=",")