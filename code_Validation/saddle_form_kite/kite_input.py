#%%


import sys, os
sys.path.append(os.path.abspath('../..'))

"""
Input file for validation of PS, benchmark case where form of self-stressed saddle is sought
"""
import numpy as np

## kite
def connectivity_matrix_kite(points_ini: np.ndarray,connections_kite: np.ndarray):

    n = points_ini.shape[0]
    matrix = np.zeros((n, n))

    for indices in connections_kite:                 # enter connections at correct index in matrix
        matrix[indices[0], indices[1]] += 1
        matrix[indices[1], indices[0]] += 1

    matrix[matrix > 1] = 1                      # remove double connections

    return matrix

def initial_conditions_kite(points_ini: np.ndarray, m_array: np.ndarray, fixed_nodes: list):

    # fill with: position, initial velocity?, mass, fixed boolean
    conditions = []
    n = points_ini.shape[0]
    for i in range(n):
        if i in fixed_nodes:
            conditions.append([points_ini[i], [0, 0, 0], m_array[i], True])
        else:
            conditions.append([points_ini[i], [0, 0, 0], m_array[i], False])

    return conditions

# dictionary of required parameters
params = {
    # model parameters
    # "n": 10,  # [-]       number of particles
    # "k_t": 100,  # [N/m]     spring stiffness
    "c": 2e3,  # [N s/m] damping coefficient
    # "L": 10,  # [m]       tether length
    # "m_block": 100,  # [kg]     mass attached to end of tether
    # "rho_tether": 0.1,  # [kg/m]    mass density tether

    # simulation settings
    "dt": 0.001,  # [s]       simulation timestep
    "t_steps": 10,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations

    # physical parameters
    "g": 9.807,  # [m/s^2]   gravitational acceleration
    # "v_w": [5, 0, 0],  # [m/s]     wind velocity vector
    # 'rho': 1.225,  # [kg/ m3]  air density
    # 'c_d_bridle': 1.05,  # [-]       drag-coefficient of bridles
    # "d_bridle": 0.02  # [m]       diameter of bridle lines
}


# instantiate connectivity matrix and initial conditions array
import code_Validation.saddle_form_kite.kite_functions as kite_functions

points_ini = np.load('/home/jellepoland/surfdrive/phd/code/Msc_Alexander_Batchelor/code_Validation/saddle_form_kite/surfplan_points_V3.npy')

bridle_ci, bridle_cj = kite_functions.extract_bridle_connectivity()
plate_point_indices = kite_functions.extract_plate_point_indices()
wing_ci, wing_cj = kite_functions.extract_wing_connectivity(plate_point_indices)

wing_connectivity = np.column_stack((wing_ci, wing_cj))
bridle_connectivity = np.column_stack((bridle_ci, bridle_cj))
connections_kite = np.vstack((wing_connectivity, bridle_connectivity))



## compression and tension resistance

# Defining stiffnesses
stiffness_canopy = 2.5e3
stiffness_tube =  1.e4
stiffness_bridle = 1.2e4

# initialising connectivities
pulley_indices = [80]
#TODO: for now just making each wing element a tube.
canopy_indices = []
tube_indices = [i for i,conn in enumerate(wing_connectivity)]

#initializing empty lists
is_compression_list = []
is_tension_list = []
is_rotational_list = []
stiffness_list = []
is_pulley_list = []

for i,conn in enumerate(connections_kite): 
    if float(i) < len(wing_connectivity):
        is_tension_list.append(True)
        is_pulley_list.append(False)
        
        if i in canopy_indices:
            stiffness_list.append(stiffness_canopy)
            is_compression_list.append(False)
            is_rotational_list.append(False)
        elif i in tube_indices:
            stiffness_list.append(stiffness_tube)
            is_compression_list.append(True)
            is_rotational_list.append(True)
        else:
            print("ERROR - wing element is neither canopy nor tube?")

    else: # if bridle-lines
        is_compression_list.append(False)
        is_tension_list.append(True)
        is_rotational_list.append(True)
        stiffness_list.append(stiffness_bridle)
        if i in pulley_indices:
            is_pulley_list.append(True)
        else:
            is_pulley_list.append(False)

params["k"] = np.array(stiffness_list)#2.5e4*np.ones(len(rest_lengths))
params["is_compression"] = is_compression_list
params["is_tension"] = is_tension_list
params["is_pulley"] = is_pulley_list

pulley_line_index = 80
idx_p3,idx_p4 = 20,21
#rest_length = rest_lengths[pulley_line_index] 
rest_length = 1.
params["pulley_other_line_pair"] = {'80': [idx_p3,idx_p4,rest_length]}
params["is_rotational"] = is_rotational_list

# needed for plate_aero
vel_app = np.array([20,0,6])
area_projected = 19.5
rho = 1.225

force_aero_plate = kite_functions.calculate_force_aero_plate(plate_point_indices,points_ini,vel_app,area_projected,rho,equal_boolean=False)
print(f'force_aero_plate: {force_aero_plate}')

# calculated parameters
params["n"] = points_ini.shape[0]
params["m_segment"] = .1

#TODO: shouldn't have a uniform mass-distribution
m_array = np.array([params["m_segment"] for i in range(params["n"])])

c_matrix = connectivity_matrix_kite(points_ini,connections_kite)
# f_nodes are the boundary conditions, handled by not updating the points position
f_nodes = [0]

# init_cond is of the form:
#TODO: Particles
# [Points, 
# Velocity, 
# Mass, 
# Fixed: True/False,  
# Line Attachment Points: True/False ]


#TODO: Elements
# stiffness - springdampers
# elongation/contraction/both -resistance,
# Rotational resistance,
# Pulley
# Type: bridle, wing,

init_cond = initial_conditions_kite(points_ini, m_array, f_nodes)

# calculating intial rest-lengths
position_initial = [position for position, velocity, mass, fixed in init_cond]
rest_lengths = []
connections = np.column_stack(np.nonzero(np.triu(c_matrix)))
for i,connections_i in enumerate(connections):
    p1 = np.array(position_initial[connections_i[0]])
    p2 = np.array(position_initial[connections_i[1]])
    rest_lengths.append(np.linalg.norm(p2-p1))

# params["l0"] = np.zeros(len(rest_lengths))
params["l0"] = .99*np.array(rest_lengths)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    # %matplotlib widget

    x = []
    y = []
    z = []
    for i in range(len(init_cond)):
        x.append(init_cond[i][0][0])
        y.append(init_cond[i][0][1])
        z.append(init_cond[i][0][2])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)

    b = np.nonzero(np.triu(c_matrix))
    b = np.column_stack((b[0], b[1]))

    ax.scatter(x, y, z, c='red')
    for indices in b:
        ax.plot([x[indices[0]], x[indices[1]]], [y[indices[0]], y[indices[1]]], [z[indices[0]], z[indices[1]]],
                color='black')

    # ax.plot(x, y, z, color='black')

    plt.show()
# %%
