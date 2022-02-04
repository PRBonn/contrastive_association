import numpy as np
import torch


def remove_outliers(points):
    m ,_ = np.median(points, 0)
    d = ((points - m) ** 2).sum(1)
    return d < np.mean(d)*2

def kalman_box_to_eight_point(kalman_bbox):

    # x, y, z, theta, l, w, h to x1,x2,y1,y2,z1,z2
    x1 = kalman_bbox[0]-kalman_bbox[4]/2
    x2 = kalman_bbox[0]+kalman_bbox[4]/2
    y1 = kalman_bbox[1]-kalman_bbox[5]/2
    y2 = kalman_bbox[1]+kalman_bbox[5]/2
    z1 = kalman_bbox[2]-kalman_bbox[6]/2
    z2 = kalman_bbox[2]+kalman_bbox[6]/2

    return [x1,y1,z1,x2,y2,z2]

def get_bbox_from_points(points):
    """
    Runs the loss on outputs of the model
    :param points: instance points Nx3
    :return: 3D bbox [x1,y1,z1,x2,y2,z2]
    """

    x1 = np.min(points[:, 0])
    x2 = np.max(points[:, 0])
    y1 = np.min(points[:, 1])
    y2 = np.max(points[:, 1])
    z1 = np.min(points[:, 2])
    z2 = np.max(points[:, 2])

    return [x1,y1,z1,x2,y2,z2], np.array([x1 + (x2-x1)/2, y1+ (y2-y1)/2,z1+ (z2-z1)/2, 0, x2-x1,y2-y1,z2-z1]) # x, y, z, theta, l, w, h

def get_2d_bbox(points):

    x1 = np.min(points[0, :])
    x2 = np.max(points[0, :])
    y1 = np.min(points[1, :])
    y2 = np.max(points[1, :])

    return [x1, y1, x2, y2]


def IoU(bbox0, bbox1):
    """
    Runs the intersection over union of two bbox
    :param bbox0: bbox1 list
    :param bbox1: bbox2 list

    :return: IoU
    """

    dim = int(len(bbox0)/2)
    overlap = [max(0, min(bbox0[i+dim], bbox1[i+dim]) - max(bbox0[i], bbox1[i])) for i in range(dim)]
    intersection = 1
    for i in range(dim):
        intersection = intersection * overlap[i]
    area0 = 1
    area1 = 1
    for i in range(dim):
        area0 *= (bbox0[i + dim] - bbox0[i])
        area1 *= (bbox1[i + dim] - bbox1[i])
    union = area0 + area1 - intersection
    if union == 0:
        return 0
    return intersection/union

def do_range_projection(points):
    #https: // github.com / PRBonn / semantic - kitti - api / blob / c4ef8140e21e589e6c795ec548584e13b2925b0f / auxiliary / laserscanvis.py  # L11
    proj_H = 128
    proj_W = 2048
    fov_up = 3.0
    fov_down = -25.0
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    depth = np.linalg.norm(points, 2, axis=1)
    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]


    return np.vstack((proj_x, proj_y))

def get_median_center_from_points(points):

    x = np.median(points[:, 0])
    y = np.median(points[:, 1])
    z = np.median(points[:, 2])

    return [x,y,z]

def euclidean_dist(b1, b2):
    ret_sum = 0
    for i in range(3):
        ret_sum += (b1[i] - b2[i])**2
    return  np.sqrt(ret_sum)
