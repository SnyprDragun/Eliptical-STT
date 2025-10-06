#!/Users/subhodeep/venv/bin/python3
import math
import numpy as np
# print(math.cos(2*math.pi))

# para_T_values = np.arange(0, 2 * math.pi, 0.1)
# for para_T in para_T_values:
#     print(math.cos(para_T))

# print(math.pow(5, 2))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
# x_center = 2       # x-coordinate of center
# y_center = 3       # y-coordinate of center
# a = 4              # length of major axis (along x-axis)
# b = 2              # length of minor axis (along y-axis)

# # Ellipse parameterization
# theta = np.linspace(0, 2*np.pi, 200)
# x = x_center + (a / 2) * np.cos(theta)  # major axis is a, so a/2 is radius
# y = y_center + (b / 2) * np.sin(theta)  # minor axis is b, so b/2 is radius
# z = np.zeros_like(x)  # for keeping the ellipse in xy-plane

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Draw ellipse
# ax.plot(x, y, z, label='Ellipse in XY plane', color='blue')

# # Set labels and limits
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title('3D View of Ellipse in XY Plane')

# # Optional: better view
# ax.view_init(elev=30, azim=60)
# ax.legend()
# plt.tight_layout()
# plt.show()


# axis_an = {}
# axes_ranges = [[1,2], [2,3], [3,4]]
# for i, axis_range in enumerate(axes_ranges):
#     axis_an[f'a{i}_min'] = axis_range[0]
#     axis_an[f'a{i}_max'] = axis_range[1]

# print(axis_an)


# import z3
# dimension = 2
# degree = 3

# # solver = z3.Solver()
# # z3.set_param("parallel.enable", True)
# N = (dimension * (degree + 1)) // 2
# # C = [z3.Real(f'C_{axis},{i}') for axis in ['x', 'y'] for i in range(N)]

# An_dict = {}
# for dim in range(dimension):
#     An_dict[f'a{dim + 1}'] = [z3.Real(f'C_a{dim + 1},{i}') for i in range(N)]

# print(An_dict, C)

# def get_polynomial_expressions(t):
#     """
#     Returns a dictionary mapping each 'a{dim}' to its symbolic polynomial expression:
#     C_a{dim},0 + C_a{dim},1 * t + C_a{dim},2 * t^2 + ... + C_a{dim},d * t^d
#     where d = degree
#     """
#     expr_dict = {}
#     for dim in range(dimension):
#         coeffs = An_dict[f'a{dim+1}']
#         expr = sum(coeffs[i] * (t ** i) for i in range(degree + 1))
#         expr_dict[f'a{dim+1}'] = expr
#     return expr_dict

# t = 5
# poly_exprs = get_polynomial_expressions(t)
# print(poly_exprs['a2'])


# def get_polynomial_expressions(t):
#     """
#     Returns a dictionary mapping each key in An_dict (e.g., 'a0', 'a1', ...) 
#     to its symbolic polynomial expression:
#     C_key,0 + C_key,1 * t + C_key,2 * t^2 + ... + C_key,d * t^d
#     where d = self.degree
#     """
#     expr_dict = {}
#     for key, coeffs in An_dict.items():
#         expr = sum(coeffs[i] * (t ** i) for i in range(degree + 1))
#         expr_dict[key] = expr
#     return expr_dict

# t = 5
# polynomials = get_polynomial_expressions(t)

# print(polynomials)
# for key, expr in polynomials.items():
#     print(f"{key}(t) =", expr)

# Flatten Z3 variables from An_dict
# diagonal_vars = []
# for key in sorted(An_dict.keys()):
#     diagonal_vars.extend(An_dict[key])

# # Create symbolic diagonal matrix
# size = len(diagonal_vars)
# D_sym = np.empty((size, size), dtype=object)
# D_sym[:] = 0  # fill with zeros

# for i in range(size):
#     D_sym[i, i] = diagonal_vars[i]

# # Display the symbolic diagonal matrix
# for row in D_sym:
#     print(row)


import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# def generate_u_thetas(thetas):
#     """
#     Compute u(theta) in n-D hyperspherical coordinates.
#     Input: thetas = [θ1, θ2, ..., θ_{n-1}]
#     Output: u (unit vector on hypersphere)
#     """
#     n = len(thetas) + 1
#     u = np.zeros(n)
    
#     for i in range(n):
#         prod = 1
#         for j in range(i):
#             prod *= np.sin(thetas[j])
#         if i < n - 1:
#             prod *= np.cos(thetas[i])
#         u[i] = prod
#     return u

# def sample_theta_grid(n_dim, step_deg=15):
#     """
#     Samples angles in hyperspherical space.
#     θ₁ to θₙ₋₂ ∈ [0, π], θₙ₋₁ ∈ [0, 2π]
#     """
#     theta_ranges = [np.deg2rad(np.arange(0, 180 + step_deg, step_deg))] * (n_dim - 2)
#     theta_ranges.append(np.deg2rad(np.arange(0, 360, step_deg)))  # Last theta in [0, 2π)

#     return product(*theta_ranges)

# def generate_ellipsoid_points(axis_lengths, center=None, step_deg=15):
#     """
#     axis_lengths: list or array of semi-axis lengths [a1, a2, ..., an]
#     step_deg: angular sampling resolution in degrees
#     """
#     n = len(axis_lengths)
#     A = np.diag(axis_lengths)

#     theta_grid = sample_theta_grid(n, step_deg)
#     points = []

#     if center is None:
#         center = np.zeros(n)
#     else:
#         center = np.array(center)
#         assert len(center) == n, "Center must have same dimension as axis_lengths"

#     for thetas in theta_grid:
#         u = generate_u_thetas(thetas)
#         x = A @ u + center # Ellipsoid point
#         points.append(x)

#     return np.array(points)

# def visualize(points, ax):
#     """
#     Visualize 2D or 3D projection of high-dimensional ellipsoid.
#     """
#     dim = 3

#     if dim == 2:
#         plt.plot(points[:,0], points[:,1], 'o', markersize=2)
#     elif dim == 3:
#         ax.scatter(points[:,0], points[:,1], points[:,2], s=2)
#     else:
#         # For n > 3, show first 3 dims as 3D projection
#         ax.scatter(points[:,0], points[:,1], points[:,2], s=2)


# # === Example usage ===
# if __name__ == "__main__":
#     all_points = []
#     dim = 3
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     if dim == 2:
#         plt.gca().set_aspect('equal')
#         plt.title("2D Ellipsoid Boundary")
#     elif dim == 3:
#         ax.set_title("3D Ellipsoid Boundary")
#     else:
#         ax.set_title(f"{dim}D Ellipsoid (First 3D Projection)")
#     for i in range(5):
#         axis_lengths = [3, 2, 1.5]  # For 3D ellipsoid (change for higher D)
#         center = [5+i*i, -2+i, i]            # Variable center
#         points = generate_ellipsoid_points(axis_lengths, center, step_deg=15)
#         all_points.append(points)
        
#     for point_dict in all_points:
#         visualize(point_dict, ax)
#     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import product

# def generate_u_thetas(thetas):
#     """
#     Compute u(theta) in n-D hyperspherical coordinates.
#     Input: thetas = [θ1, θ2, ..., θ_{n-1}]
#     Output: u (unit vector on hypersphere)
#     """
#     n = len(thetas) + 1
#     u = np.zeros(n)

#     for i in range(n):
#         prod = 1
#         for j in range(i):
#             prod *= np.sin(thetas[j])
#         if i < n - 1:
#             prod *= np.cos(thetas[i])
#         u[i] = prod
#     return u

# def sample_theta_grid(n_dim, step_deg=15):
#     """
#     Samples angles in hyperspherical space.
#     θ₁ to θₙ₋₂ ∈ [0, π], θₙ₋₁ ∈ [0, 2π]
#     """
#     theta_ranges = [np.deg2rad(np.arange(0, 180 + step_deg, step_deg))] * (n_dim - 2)
#     theta_ranges.append(np.deg2rad(np.arange(0, 360, step_deg)))  # Last theta in [0, 2π)

#     return product(*theta_ranges)

# def generate_ellipsoid_points(axis_lengths, center=None, step_deg=15):
#     """
#     axis_lengths: list of semi-axis lengths [a1, a2, ..., an]
#     center: list of center coordinates [c1, c2, ..., cn]
#     step_deg: angular sampling resolution
#     """
#     n = len(axis_lengths)
#     A = np.diag(axis_lengths)

#     if center is None:
#         center = np.zeros(n)
#     else:
#         center = np.array(center)
#         assert len(center) == n, "Center must have same dimension as axis_lengths"

#     theta_grid = sample_theta_grid(n, step_deg)
#     points = []

#     for thetas in theta_grid:
#         u = generate_u_thetas(thetas)
#         x = A @ u + center
#         points.append(x)

#     points = np.array(points)

#     # Split into x1, x2, ..., xn
#     coordinate_dict = {f'x{i+1}': points[:, i] for i in range(n)}
#     return coordinate_dict

# def visualize(points):
#     """
#     Visualize 2D or 3D projection of high-dimensional ellipsoid.
#     """
#     dim = points.shape[1]
#     if dim == 2:
#         plt.plot(points[:, 0], points[:, 1], 'o', markersize=2)
#         plt.gca().set_aspect('equal')
#         plt.title("2D Ellipsoid (with center)")
#     elif dim == 3:
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
#         ax.set_title("3D Ellipsoid (with center)")
#     else:
#         # Project to first 3 dimensions
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
#         ax.set_title(f"{dim}D Ellipsoid (First 3D Projection)")
#     plt.show()

# # === Example usage ===
# if __name__ == "__main__":
#     axis_lengths = [3, 2, 1.5]     # 3D ellipsoid
#     center = [5, -2, 1]            # Variable center
#     points = generate_ellipsoid_points(axis_lengths, center=center, step_deg=15)
#     visualize(points)


# import numpy as np
# import z3
# from itertools import product
# import math

# def generate_u_thetas(thetas):
#     """
#     Compute u(theta) in n-D hyperspherical coordinates (all numerical).
#     Input: thetas = [θ1, θ2, ..., θ_{n-1}]
#     Output: u (unit vector on hypersphere)
#     """
#     n = len(thetas) + 1
#     u = np.zeros(n)

#     for i in range(n):
#         prod = 1
#         for j in range(i):
#             prod *= np.sin(thetas[j])
#         if i < n - 1:
#             prod *= np.cos(thetas[i])
#         u[i] = prod
#     return u

# def sample_theta_grid(n_dim, step_deg=15):
#     """
#     Samples angles in hyperspherical space.
#     θ₁ to θₙ₋₂ ∈ [0, π], θₙ₋₁ ∈ [0, 2π]
#     """
#     theta_ranges = [np.deg2rad(np.arange(0, 180 + step_deg, step_deg))] * (n_dim - 2)
#     theta_ranges.append(np.deg2rad(np.arange(0, 360, step_deg)))  # θₙ₋₁ ∈ [0, 2π)

#     return product(*theta_ranges)

# def generate_symbolic_ellipsoid(n_dim, step_deg=15):
#     """
#     Generates a list of Z3 expressions x_i = a_i * u_i + c_i
#     axis_lengths and centers are symbolic variables
#     """
#     axis_vars = [z3.Real(f'a{i+1}') for i in range(n_dim)]
#     center_vars = [z3.Real(f'c{i+1}') for i in range(n_dim)]

#     theta_grid = sample_theta_grid(n_dim, step_deg)

#     symbolic_points = []
#     for thetas in theta_grid:
#         u = generate_u_thetas(thetas)  # numerical vector
#         expr_point = [axis_vars[i] * u[i] + center_vars[i] for i in range(n_dim)]
#         symbolic_points.append(expr_point)

#     # Return dict of expressions by axis
#     coord_dict = {f'x{i+1}': [p[i] for p in symbolic_points] for i in range(n_dim)}

#     return coord_dict, axis_vars, center_vars


# if __name__ == "__main__":
#     coord_dict, axis_vars, center_vars = generate_symbolic_ellipsoid(n_dim=3, step_deg=90)

#     print("Z3 symbolic variables:")
#     print("Axes:", axis_vars)
#     print("Center:", center_vars)

#     print("\nSample symbolic x1 expressions:")
#     print(coord_dict)
#     # for expr in coord_dict['x1'][:25]:  # Print first 5 expressions
#     #     print(expr)


C_list = [0.5, 0.0, 0.2, 0.0, 0.8, -0.06160856687172519, 0.6233766233766276, -0.07290954659375785, 0.8, -0.057419996513785675, 0.5807622504537178, -0.06562285315861204] 
an_dict = {'a1': [0.5, 0.0, 0.0, 0.0], 'a2': [0.8, 0.0, 0.0, 0.0], 'a3': [0.8, 0.0, 0.0, 0.0]}

import csv
# from tabulate import tabulate
def write_coefficients_to_csv(C_list, an_dict, filename='coefficients.csv'):
    """
    Write C_list as first column and each key in an_dict as a separate column.
    Shorter columns will be padded with empty strings.
    """
    # Prepare header
    header = ["Tube Coefficients"] + list(an_dict.keys())

    # Determine max number of rows needed
    num_rows = max(len(C_list), max(len(v) for v in an_dict.values()))

    # Prepare rows
    rows = []
    for i in range(num_rows):
        row = []
        # First column: C_list value or empty
        row.append(C_list[i] if i < len(C_list) else "")
        # Remaining columns: values from each a{i}, or empty if out of range
        for key in an_dict:
            vals = an_dict[key]
            row.append(vals[i] if i < len(vals) else "")
        rows.append(row)

    # After building `rows` and `header`
    # print(tabulate(rows, headers=header, tablefmt='grid'))


    # Write to CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ CSV written to {filename}")

write_coefficients_to_csv(C_list, an_dict, filename="coefficients_output.csv")
