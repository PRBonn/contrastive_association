import numpy as np

def normalize_points(points):
    """ Normalize the points to the unit sphere,
        Input:
            Nx3 array
        Output:
            Nx3 array
    """
    N = points.shape[0]
    if N <= 1:
        return points
    centroid = np.mean(points, axis=0)
    normalized = points - centroid
    m = np.max(abs(normalized),axis=0)
    normalized = normalized / m
    return normalized

def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N = points.shape[0]
    assert(clip > 0)
    jittered_points = np.clip(sigma * np.random.randn(N, 3), -1*clip, clip)
    jittered_points += points
    return jittered_points

def random_drop_n_cuboids(points, features):
    """ Randomly drop N cuboids from the point cloud and remove
        the corresponding point features.
        Input:
            Nx3 array, points of the point cloud
            NxC array, per point features
        Return:
            Mx3 array, point cloud without dropped points
            MxC array, per point features without dropped features
    """
    dropped_pcd, dropped_feat = random_drop_point_cloud(points, features)
    cuboids_count = 1
    while cuboids_count < 7 and np.random.uniform(0., 1.) > 0.3:
        dropped_pcd, dropped_feat = random_drop_point_cloud(dropped_pcd, dropped_feat)
        cuboids_count += 1

    return dropped_pcd, dropped_feat

def random_drop_point_cloud(points, features):
    """ Randomly drop cuboids from the point cloud.
        Input:
            Nx3 array, original point cloud
            NxC array, original per point features
        Return:
            Mx3 array, dropped point cloud
            MxC array, dropped per point features
    """
    N = points.shape[0]
    new_points = []
    range_xyz = np.max(points[:,0:3], axis=0) - np.min(points[:,0:3], axis=0)

    crop_range = np.random.uniform(0.1, 0.15)
    new_range = range_xyz * crop_range / 2.0
    sample_center = points[np.random.choice(len(points)), 0:3]
    max_xyz = sample_center + new_range
    min_xyz = sample_center - new_range

    upper_idx = np.sum((points[:,0:3] < max_xyz).astype(np.int32), 1) == 3
    lower_idx = np.sum((points[:,0:3] > min_xyz).astype(np.int32), 1) == 3

    new_pointidx = ~((upper_idx) & (lower_idx))
    new_points = points[new_pointidx,:]
    new_features = features[new_pointidx,:]

    return new_points, new_features

def random_point_dropout(points, features, max_dropout_ratio=0.875):
    """ Randomly drop points from the point cloud.
        Input:
            Nx3 array, original point cloud
            NxC array, original per point features
        Return:
            Mx3 array, dropped point cloud
            MxC array, dropped per point features
    """
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((points.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0 and len(drop_idx) < len(points)/2:
        new_points = np.delete(points,drop_idx,axis=0)
        new_features = np.delete(features,drop_idx,axis=0)
        return new_points, new_features
    else:
        return points, features

def sample_random_normals(N, dim=3):
    """Returnns N unit normals of dimesion dim sampled on an unit ball
    http://en.wikipedia.org/wiki/N-sphere#Generating_random_points."""
    normals = np.random.normal(size=(N, dim))
    return normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

def random_plane_dropout(points, features):
    """ Randomly drop points on once side of a plane
        Input:
            Nx3 array, original point cloud
            NxC array, original per point features
        Return:
            Mx3 array, dropped point cloud
            MxC array, dropped per point features
    """
    #apply only sometimes
    prob = np.random.random()
    if prob < 0.6:
        normalized = normalize_points(points)
        normal = sample_random_normals(1)[0]
        tr_idx = int(np.random.random(1)*normalized.shape[0])
        trans_pt = normalized[tr_idx]
        remove = []
        for i in range(normalized.shape[0]):
            res = np.dot(normalized[i]-trans_pt,normal)
            if res > 0:
                remove.append(i)
        if len(remove) < normalized.shape[0]:
            new_points = np.delete(points,remove,axis=0)
            new_features = np.delete(features,remove,axis=0)
            return new_points, new_features
        else:
            return points, features
    else:
        return points, features

def contour_dropout(points, features):
    """ Randomly drop the most outer points
        Input:
            Nx3 array, original point cloud
            NxC array, original per point features
        Return:
            Mx3 array, dropped point cloud
            MxC array, dropped per point features
    """
    #apply only sometimes
    prob = np.random.random()
    if prob < 0.6:
        r = np.random.random()
        if r < 0.4:
            return points, features
        #shrink instance
        #keep only points in a certain range
        normalized = normalize_points(points)
        rx = (normalized[:,0]<r) * (normalized[:,0]>-r)
        ry = (normalized[:,1]<r) * (normalized[:,1]>-r)
        rz = (normalized[:,2]<r) * (normalized[:,2]>-r)
        new_points = points[rx*ry*rz]
        new_features = features[rx*ry*rz]
        return new_points, new_features
    else:
        return points, features
