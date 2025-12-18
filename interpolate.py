
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def smooth_and_resample_3d(coords, n_points, smoothness):
    coords = np.array(coords, dtype=float)

    if not np.allclose(coords[0],coords[-1]):
        coords = np.vstack((coords,coords[0]))

    deltas = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1)) # list of lengths of edges
    cumdist = np.insert(np.cumsum(seg_lengths), 0, 0)

    #Interpolate to create equally spaced points 
    total_len = cumdist[-1]
    uniform_d = np.linspace(0, total_len, n_points)
    x_uniform = np.interp(uniform_d, cumdist, coords[:, 0])
    y_uniform = np.interp(uniform_d, cumdist, coords[:, 1])
    #z_uniform = np.interp(uniform_d, cumdist, coords[:, 2])
    scale_factor = 3 
    z_uniform = np.interp(uniform_d, cumdist, coords[:, 2]) * scale_factor

    def neighbor_average(arr, iterations):
        arr = arr.copy()
        n = len(arr)
        for _ in range(iterations):
            arr_new = arr.copy()
            for i in range(n):
                arr_new[i] = (arr[(i-1)%n] + arr[i] + arr[(i+1)%n]) / 3
            arr = arr_new
        return arr

    if smoothness > 1:
        x_smooth = neighbor_average(x_uniform, smoothness)
        y_smooth = neighbor_average(y_uniform, smoothness)
        z_smooth = neighbor_average(z_uniform, smoothness)
    else:
        x_smooth, y_smooth, z_smooth = x_uniform, y_uniform, z_uniform

    return np.column_stack((x_smooth, y_smooth, z_smooth))

# # NEW COORDS. coords is the original coords list. n_points is how many points you want in the new list.
# # increasing smoothness makes the knot more rounded.
# og_len=len(coords) # original number of points.



# og_smoothness=1 # no extra smoothing
# new_coords_3d = smooth_and_resample_3d(coords, n_points=og_len*5, smoothness=og_smoothness)

# closed_coords = np.vstack([new_coords_3d, new_coords_3d[0]])

# # PLOT 
# fig = plt.figure(figsize=(9,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(closed_coords[:,0], closed_coords[:,1], closed_coords[:,2],
#         color='green', lw=2)
# ax.scatter(new_coords_3d[:,0], new_coords_3d[:,1], new_coords_3d[:,2],
#            color='blue', s=5)

# # PRINT THE COORDS AFTER INTERPOLATION TO FILE
# fidd=open('file-coords-interpolated.txt','w')
# for t in closed_coords:
#     print(t[0],t[1],t[2],file=fidd)
# fidd.close()

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Knot')
# ax.legend()

# plt.show()