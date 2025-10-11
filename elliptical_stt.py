#!/Users/subhodeep/venv/bin/python3
'''script for `Rotating Rigid Spacecraft` example'''
import z3
import csv
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class STT_Solver():
    '''class for generating STT based on constraints on trajectory'''
    def __init__(self, degree, dimension, time_step, semi_minor_axis_range, semi_major_axis_range):
        self.setpoints = []
        self.obstacles = []
        self.goal = []
        self._start = 0
        self._finish = 0
        self._step = time_step
        self._range = 0
        self._x_start = 0
        self._x_finish = 0
        self._y_start = 0
        self._y_finish = 0
        self._z_start = 0
        self._z_finish = 0

        self.semi_minor_axis_min = semi_minor_axis_range[0]
        self.semi_minor_axis_max = semi_minor_axis_range[1]
        self.semi_major_axis_min = semi_major_axis_range[0]
        self.semi_major_axis_max = semi_major_axis_range[1]

        self.lambda_values = np.arange(0, 1.1, 0.1)
        self.degree = degree
        self.dimension = dimension
        self.solver = z3.Solver()
        z3.set_param("parallel.enable", True)
        N = (self.dimension * (self.degree + 1)) // 2
        self.C = [z3.Real(f'C_{axis},{i}') for axis in ['x', 'y'] for i in range(N)]
        self.major_axis_length = z3.Real(f'A')
        self.minor_axis_length = z3.Real(f'B')
        self.A = 0.0
        self.B = 0.0
        self.solver.add(self.major_axis_length > self.minor_axis_length)

    def gammas(self, t):
        '''method to calculate tube equations'''
        tubes = [z3.Real(f'g_{i}') for i in range(self.dimension)]

        for i in range(self.dimension):
            tubes[i] = 0
            power = 0
            for j in range(self.degree + 1):
                tubes[i] += ((self.C[j + i * (self.degree + 1)]) * (t ** power))
                power += 1
        return tubes

    def real_gammas(self, t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(self.dimension)

        for i in range(self.dimension):
            power = 0
            for j in range(self.degree + 1): #each tube eq has {degree+1} terms
                real_tubes[i] += ((C_fin[j + i * (self.degree + 1)]) * (t ** power))
                power += 1
        return real_tubes

    def gamma_dot(self, t):
        '''method to calculate tube equations'''
        tubes = [z3.Real(f'gd_{i}') for i in range(self.dimension)]

        for i in range(self.dimension):
            tubes[i] = 0
            power = 0
            for j in range(self.degree + 1):
                if power < 1:
                    tubes[i] += 0
                    power += 1
                else:
                    tubes[i] += power * ((self.C[j + i * (self.degree + 1)]) * (t ** (power - 1)))
                    power += 1
        return tubes

    def real_gamma_dot(self, t, C_fin):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(self.dimension)

        for i in range(self.dimension):
            power = 0
            for j in range(self.degree + 1):
                if power < 1:
                    real_tubes[i] += 0
                    power += 1
                else:
                    real_tubes[i] += power * ((C_fin[j + i * (self.degree + 1)]) * (t ** (power - 1)))
                    power += 1
        return real_tubes

    def general(self):
        '''method for general specifications'''
        constraint_major_axis = z3.And(self.major_axis_length > 2 * self.semi_major_axis_min, self.major_axis_length < 2 * self.semi_major_axis_max)
        constraint_minor_axis = z3.And(self.minor_axis_length > 2 * self.semi_minor_axis_min, self.minor_axis_length < 2 * self.semi_minor_axis_max)
        eccentricity = 1 - ((self.minor_axis_length / self.major_axis_length) * (self.minor_axis_length / self.major_axis_length))
        constraint_eccentricity = z3.And(eccentricity > 0, eccentricity < 1)
        self.solver.add(constraint_major_axis)
        self.solver.add(constraint_minor_axis)
        self.solver.add(constraint_eccentricity)

    def plot_for_2D(self, C_fin):
        x = np.zeros(self.getRange())
        y = np.zeros(self.getRange())

        gd_x = np.zeros(self.getRange())
        gd_y = np.zeros(self.getRange())

        for i in range(self.getRange()):
            tube_gamma = self.real_gammas(self.getStart() + i * self._step, C_fin)
            x[i] = tube_gamma[0]
            y[i] = tube_gamma[1]

            tube_gamma_dot = self.real_gamma_dot(self.getStart() + i * self._step, C_fin)
            gd_x[i] = tube_gamma_dot[0]
            gd_y[i] = tube_gamma_dot[1]

        # print("gamma_x: ", x)
        # print("gamma_y: ", y)

        # print("gamma_dot for x = ", gd_x)
        # print("gamma_dot for y = ", gd_y)

        fig1, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        ax, bx = axs
        for i in self.setpoints:        # t1  x1/y1/z1  t2    t1  x2/y2/z2  x1
            square_x = patches.Rectangle((i[4], i[0]), i[5] - i[4], i[1] - i[0], edgecolor='green', facecolor='none')
            square_y = patches.Rectangle((i[4], i[2]), i[5] - i[4], i[3] - i[2], edgecolor='green', facecolor='none')
            ax.add_patch(square_x)
            bx.add_patch(square_y)

        for i in self.obstacles:        # t1  x1/y1/z1  t2    t1  x2/y2/z2  x1
            square_x = patches.Rectangle((i[4], i[0]), i[5] - i[4], i[1] - i[0], edgecolor='red', facecolor='none')
            square_y = patches.Rectangle((i[4], i[2]), i[5] - i[4], i[3] - i[2], edgecolor='red', facecolor='none')
            ax.add_patch(square_x)
            bx.add_patch(square_y)

        t = np.linspace(self.getStart(), self.getFinish(), self.getRange())
        print("range: ", self.getRange(), "\nstart: ", self.getStart(), "\nfinish: ", self.getFinish(), "\nstep: ", self._step)

        ax.plot(t, x)
        bx.plot(t, y)
        ax.set_title("t vs x")
        bx.set_title("t vs y")

        #-------------------------------- 3D (X vs Y vs T) --------------------------------#
        fig2 = plt.figure()
        cx = fig2.add_subplot(111, projection='3d')

        for i in range(self.getRange()):
            tube_gamma = self.real_gammas(self.getStart() + i * self._step, C_fin)
            x_center = tube_gamma[0]
            y_center = tube_gamma[1]

            theta = np.linspace(0, 2*np.pi, 200)
            x = x_center + (self.A / 2) * np.cos(theta)
            y = y_center + (self.B / 2) * np.sin(theta)
            z = self.getStart() + i * self._step

            cx.plot(x, y, z, label='Ellipse in XY plane', color='blue')
            cx.set_xlabel('X axis')
            cx.set_ylabel('Y axis')
            cx.set_zlabel('Z axis')
            cx.set_title('3D View of Ellipse in XY Plane')
            cx.view_init(elev=30, azim=60)
            plt.tight_layout()

        for i in self.obstacles:
            cx.add_collection3d(Poly3DCollection(self.faces(i), facecolors='red', edgecolors='r', alpha=0.25))

        for i in self.setpoints:
            cx.add_collection3d(Poly3DCollection(self.faces(i), facecolors='green', edgecolors='green', alpha=0.25))

    def find_solution(self):
        '''method to plot the tubes'''
        start = time.time()
        print("Solving...")

        self.setAll()
        self.general()

        if self.solver.check() == z3.sat:
            model = self.solver.model()
            self.A = np.float64(model[self.major_axis_length].numerator().as_long())/np.float64(model[self.major_axis_length].denominator().as_long())
            self.B = np.float64(model[self.minor_axis_length].numerator().as_long())/np.float64(model[self.minor_axis_length].denominator().as_long())
            print("A = ", self.A, "\nB = ", self.B, "\ne = ", math.sqrt(1 - math.pow((self.B/self.A), 2)))
            xi = np.zeros((self.dimension) * (self.degree + 1))
            Coeffs = []
            C_fin = np.zeros((self.dimension) * (self.degree + 1))
            for i in range(len(self.C)):
                xi[i] = (np.float64(model[self.C[i]].numerator().as_long()))/(np.float64(model[self.C[i]].denominator().as_long()))
                print("{} = {}".format(self.C[i], xi[i]))
                Coeffs.append(xi[i])

            for i in range(len(Coeffs)):
                C_fin[i] = Coeffs[i]

            # fieldnames = ['Coefficient', 'Value']
            # data_dicts = []
            # for i in range(len(Coeffs)):
            #     data_dicts.append({'Coefficient': self.C[i], 'Value': Coeffs[i]})

            # with open('Spacecraft.csv', 'w', newline='') as file:
            #     writer = csv.DictWriter(file, fieldnames=fieldnames)
            #     if file.tell() == 0:
            #         writer.writeheader()  # Write headers only if the file is empty
            #     writer.writerows(data_dicts)

            self.plot_for_2D(C_fin)
            self.print_equation(C_fin)
            end = time.time()
            self.displayTime(start, end)
            plt.show(block=True)

        else:
            print("No solution found.")
            print("range: ", self.getRange(), "\nstart: ", self.getStart(), "\nfinish: ", self.getFinish(), "\nstep: ", self._step)
            end = time.time()
            self.displayTime(start, end)

    def print_equation(self, C):
        for i in range(self.dimension):
            print("gamma", i, "= ", end = "")
            power = 0
            for j in range(self.degree + 1):
                print("C", j + i * (self.degree + 1), "* t.^", power, "+ ", end = "")
                power += 1
            print("\n")

    def faces(self, i):
        vertices = [[i[0], i[2], i[4]], [i[1], i[2], i[4]], [i[1], i[3], i[4]], [i[0], i[3], i[4]],  # Bottom face
                    [i[0], i[2], i[5]], [i[1], i[2], i[5]], [i[1], i[3], i[5]], [i[0], i[3], i[5]]]   # Top face

        # Define the 6 faces of the cube using the vertices
        faces = [   [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
                    [vertices[0], vertices[3], vertices[7], vertices[4]]]  # Left face
        return faces

    def setAll(self):
        all_points = self.setpoints + self.obstacles
        x1, x2, y1, y2, t1, t2 = [], [], [], [], [], []
        for i in all_points:
            x1.append(i[0])
            x2.append(i[1])
            y1.append(i[2])
            y2.append(i[3])
            t1.append(i[4])
            t2.append(i[5])

        self.setStart(min(t1))
        self.setFinish(max(t2))
        self.set_x_start(min(x1))
        self.set_x_finish(max(x2))
        self.set_y_start(min(y1))
        self.set_y_finish(max(y2))
        self.setRange(int((self.getFinish() - self.getStart() + self._step) / self._step))

    def displayTime(self, start, end):
        k = int(end - start)
        days = k // (3600 * 24)
        hrs = (k // 3600) - (days * 24)
        mins = (k // 60) - (hrs * 60)
        if end - start < 1:
            secs = (((end - start) * 10000) // 100) / 100
        else:
            secs = k - (mins * 60) - (hrs * 3600) - (days * 24 * 3600)
        print("Time taken: ", days, "days, ", hrs , "hours, ", mins, "minutes, ", secs, "seconds")

    def getStart(self):
        return self._start

    def setStart(self, start_value):
        self._start = start_value

    def getFinish(self):
        return self._finish

    def setFinish(self, finish_value):
        self._finish = finish_value

    def getRange(self):
        return self._range

    def setRange(self, value):
        self._range = value

    def get_x_start(self):
        return self._x_start

    def set_x_start(self, value):
        self._x_start = value

    def get_x_finish(self):
        return self._x_finish

    def set_x_finish(self, value):
        self._x_finish = value

    def get_y_start(self):
        return self._y_start

    def set_y_start(self, value):
        self._y_start = value

    def get_y_finish(self):
        return self._y_finish

    def set_y_finish(self, value):
        self._y_finish = value


solver = STT_Solver(degree=4, dimension=2, time_step=1, semi_minor_axis_range=[0.5, 0.51], semi_major_axis_range=[0.5, 0.51])


def reach(x1, x2, y1, y2, t1, t2):
    solver.setpoints.append([x1, x2, y1, y2, t1, t2])
    all_constraints = []
    t_values = np.arange(t1, t2, solver._step)
    para_T_values = np.arange(0, 2 * math.pi, solver._step)

    for t in t_values:
        gamma = solver.gammas(t)
        gamma_x = gamma[0]
        gamma_y = gamma[1]

        constraint_x_sat = z3.And(gamma_x > x1, gamma_x < x2)
        constraint_y_sat = z3.And(gamma_y > y1, gamma_y < y2)
        all_constraints.append(constraint_x_sat)
        all_constraints.append(constraint_y_sat)

        for para_T in para_T_values:
            constraint_x = z3.And((((solver.major_axis_length * math.cos(para_T)) / 2) + gamma_x) > x1, (((solver.major_axis_length * math.cos(para_T)) / 2) + gamma_x) < x2 )
            constraint_y = z3.And((((solver.minor_axis_length * math.sin(para_T)) / 2) + gamma_y) > y1, (((solver.minor_axis_length * math.sin(para_T)) / 2) + gamma_y) < y2 )
            all_constraints.append(constraint_x)
            all_constraints.append(constraint_y)

    print("Added Reach Constraints: ", solver.setpoints)
    end = time.time()
    solver.displayTime(start, end)
    return all_constraints

def avoid(x1, x2, y1, y2, t1, t2):
    solver.obstacles.append([x1, x2, y1, y2, t1, t2])
    all_constraints = []
    t_values = np.arange(t1, t2, solver._step)
    para_T_values = np.arange(0, 2 * math.pi, solver._step)

    for t in t_values:
        gamma = solver.gammas(t)
        gamma_x = gamma[0]
        gamma_y = gamma[1]

        constraint_xy_sat = z3.Or(z3.Or(gamma_x < x1, gamma_x > x2), z3.Or(gamma_y < y1, gamma_y > y2))
        all_constraints.append(constraint_xy_sat)

        for para_T in para_T_values:
            constraint_xy = z3.Or(z3.Or((((solver.major_axis_length * math.cos(para_T)) / 2) + gamma_x) < x1, (((solver.major_axis_length * math.cos(para_T)) / 2) + gamma_x) > x2 ), z3.Or((((solver.minor_axis_length * math.sin(para_T)) / 2) + gamma_y) < y1, (((solver.minor_axis_length * math.sin(para_T)) / 2) + gamma_y) > y2 ))
            all_constraints.append(constraint_xy)

    print("Added Avoid Constraints: ", solver.obstacles)
    end = time.time()
    solver.displayTime(start, end)
    return all_constraints


start = time.time()

# S_constraints_list = reach(0, 3, 0, 3, 0, 1)
# O_constraints_list = avoid(3.5, 4.5, 3.5, 4.5, 3, 4)
# G_constraints_list = reach(5, 8, 5, 8, 5, 6)

# for S in S_constraints_list:
#     solver.solver.add(S)

# for O in O_constraints_list:
#     solver.solver.add(O)

# for G in G_constraints_list:
#     solver.solver.add(G)

# solver.find_solution()

S_constraints_list = reach(0, 3, 0, 3, 0, 1)
T1_constraints_list = reach(6, 9, 6, 9, 6, 7)
T2_constraints_list = reach(12, 15, 6, 9, 6, 7)
G_constraints_list = reach(18, 21, 15, 18, 14, 18)
O_constraints_list = avoid(9, 12, 6, 9, 3, 10)

for S in S_constraints_list:
    solver.solver.add(S)

for O in O_constraints_list:
    solver.solver.add(O)

for G in G_constraints_list:
    solver.solver.add(G)

T_choice = 1#random.randint(1, 2)
if T_choice == 1:
    print("Choosing T1")
    for T1 in T1_constraints_list:
        solver.solver.add(T1)
else:
    print("Choosing T2")
    for T2 in T2_constraints_list:
        solver.solver.add(T2)

solver.find_solution()
