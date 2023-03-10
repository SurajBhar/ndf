import numpy as np
import open3d as o3d

# Loading the .npz file from the path
data = np.load('/work/ws-tmp/g051176-SB_NDF/ndf/experiments/shapenet_cars_pretrained/evaluation/generation/02958343/1a56d596c77ad5936fa87a658faf1d26/dense_point_cloud_7.npz')

# Accessing the point cloud stored in the file
points = data['point_cloud']

# Creating an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Saving Point cloud object in a ply file
o3d.io.write_point_cloud("pcd.ply", pcd)

# Visualising the point cloud
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.webrtc_server.enable_webrtc()