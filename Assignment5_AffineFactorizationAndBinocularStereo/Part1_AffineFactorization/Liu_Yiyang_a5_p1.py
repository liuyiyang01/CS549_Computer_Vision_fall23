import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky
from scipy.optimize import minimize
import matplotlib.image as mpimg

data_matrix = np.loadtxt("factorization_data/measurement_matrix.txt")
print("data matrix shape (2M x N): ", data_matrix.shape)

# # Split the data matrix into x and y coordinates


print("------------------------normalizing the data point (2D space)------------------------")

mean_points = np.mean(data_matrix, axis=1)
print("mean point shape: ", mean_points.shape)
normalized_data = data_matrix - mean_points[:, np.newaxis]
print("normalized data shape: ", normalized_data.shape)

# Apply SVD
U, W, Vt = np.linalg.svd(normalized_data)
print("U shape", U.shape)
print("W shape", W.shape)
print("Vt shape", Vt.shape)
# print(W[:, :3])

U = U[:, :3]  # Take the first three columns of U
W = np.diag(W[:3]) 
V = Vt[:3, :]  # Take the first three columns of V and transpose
# print(np.sqrt(W))

# Derive structure and motion matrices
motion_matrix = U @ np.sqrt(W)
structure_matrix = np.sqrt(W) @ V

print("------------------------solving for L using gradient------------------------")
# Define the cost function
def cost_function(L_flat, A):
    L = L_flat.reshape((3, 3))
    sum = 0
    for i in range(A.shape[0]):
        ALAT = A[i] @ L @ A[i].T
        sum += 0.5 * np.linalg.norm(ALAT - np.eye(2), 'fro')**2
    sum /= A.shape[0]
    return sum

# Set up constraints for the optimizer
# constraints = ({'type': 'eq', 'fun': lambda L_flat: np.trace(L_flat.reshape((2, 2))) - 1})

# Initial guess for L
initial_guess = np.eye(3).flatten()

A_submatrices = np.stack([motion_matrix[i:i+2, :] for i in range(0, motion_matrix.shape[0], 2)])
print("A_submatrices shape: ", A_submatrices.shape)

# Call the optimizer
# result = minimize(cost_function, initial_guess, args=(motion_matrix,), constraints=constraints)
result = minimize(cost_function, initial_guess, args=(A_submatrices,))

# Recover the optimized L
L_optimized = result.x.reshape((3, 3))
print("optimized L:", L_optimized)
print("check optimized L: ", A_submatrices[32] @ L_optimized @ A_submatrices[32].T)

print("------------------------Recovering Q from L by Cholesky decomposition------------------------")
Q = np.linalg.cholesky(L_optimized)
print("Recovered Q: ", Q)
print("Check Recovered Q: ", Q @ Q.T - L_optimized)

print("------------------------Updating motion_matrix and structure_matrix------------------------")
motion_matrix = motion_matrix @ Q
structure_matrix = np.linalg.inv(Q) @ structure_matrix
# print("updated motion matrix: ", motion_matrix)
# print("updated structure matrix: ", structure_matrix)
print("done")

print("------------------------visualization------------------------")
# Reconstruct 3D points
reconstructed_points = structure_matrix
print("reconstructed_points shape: ", reconstructed_points.shape)
# Display 3D structure
fig = plt.figure("3D Structure of Points")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reconstructed_points[0, :], reconstructed_points[1, :], reconstructed_points[2, :])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Structure')
plt.show()



# Display three frames with observed and estimated points overlayed
x_coords = normalized_data[::2, :]
y_coords = normalized_data[1::2, :]
observed_points = np.stack((x_coords, y_coords), axis=0)
print("observed_points shape: ", observed_points.shape)
for frame in [0,50,100]:
    # 指定图片路径
    image_path = "factorization_data/frame"+str(frame+1).zfill(8)+'.jpg'

    # 读取图片
    img = mpimg.imread(image_path)

    plt.figure(f'Frame {frame + 1}')
    plt.imshow(img)
    plt.scatter(observed_points[0, frame, :]+mean_points[2*frame], observed_points[1, frame, :]+mean_points[2*frame+1], label='Observed', marker='o')
    projected_points = motion_matrix[2*frame:2*frame+2,:] @ reconstructed_points
    plt.scatter(projected_points[0, :]+mean_points[2*frame], projected_points[1, :]+mean_points[2*frame+1], label='Estimated', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Frame {frame + 1}')
    plt.show()

residuals_per_frame = []

for frame in range(observed_points.shape[1]):
    projected_points = motion_matrix[2*frame:2*frame+2,:] @ reconstructed_points
    residuals = np.linalg.norm(observed_points[:2, frame, :] - projected_points, axis=0)
    # print(residuals)
    # Sum of squared residuals for this frame
    sum_of_squares_residuals = np.sum(residuals)
    # Append residuals for this frame to the list
    residuals_per_frame.append(sum_of_squares_residuals)

# # Plot residuals per frame
# plt.figure()
# plt.plot(range(1, num_frames + 1), residuals_per_frame, marker='o')
# plt.title('Residuals per Frame')
# plt.xlabel('Frame Number')
# plt.ylabel('Sum of Squared Residuals (in pixels)')
# plt.show()


# Calculate total residual
residuals = np.sum(residuals_per_frame)
print(f'Total Residual: {residuals}')

# Plot per-frame residual
plt.figure('Per-Frame Residual vs Frame Number')
plt.plot(residuals_per_frame)
plt.xlabel('Frame Number')
plt.ylabel('Per-Frame Residual')
plt.title('Per-Frame Residual vs Frame Number')
plt.show()
