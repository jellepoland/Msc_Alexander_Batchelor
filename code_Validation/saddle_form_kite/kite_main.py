#%%
"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""
%load_ext autoreload
%autoreload 2

import sys, os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import code_Validation.saddle_form_kite.kite_input as input
import code_Validation.saddle_form_kite.kite_functions as kite_functions
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import time
from src.particleSystem.ParticleSystem import ParticleSystem
%matplotlib widget

#changing the input parameters
input.params["dt"] = 0.005
input.params["t_steps"] = int(2e2)
input.vel_app = np.array([17,0,3])
input.params["c"] = 0 # *np.ones(len(rest_lengths))


# params
# particle settings
# element settings
### INPUT


def instantiate_ps():
    # c_matrix = damping
    # init_cond = initial_conditions
    # params = input.params
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)

def plot(psystem: ParticleSystem, psystem2: ParticleSystem):
    n = input.params['n']
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])

    x = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))

    position = pd.DataFrame(index=t_vector, columns=x)
    position2 = pd.DataFrame(index=t_vector, columns=x)

    n = input.params["n"]
    points = np.array([init_cond_i[0] for init_cond_i in input.init_cond])
    force_aero_wing = kite_functions.calculate_force_aero_plate(input.plate_point_indices,points,input.vel_app,input.area_projected,input.rho,equal_boolean=False)

    start_time = time.time()
    for i,step in enumerate(t_vector):           # propagating the simulation for each timestep and saving results
        
        ## external force
        # Plate-aero static aero
        # force_aero_wing = kite_functions.calculate_force_aero_plate(input.plate_point_indices,points,input.vel_app,input.area_projected,input.rho,equal_boolean=False)

        # print(f'force_aero_wing: {force_aero_wing}')
        # old definition
        # f_ext = np.array([[0, 0, 1e3] for i in range(n)]).flatten()

        f_ext = force_aero_wing.flatten()
        ## internal force (structural solver - the magic happens here) 
        # position.loc[step], _ = psystem.simulate(f_ext)
        position.loc[step],_ = psystem.kin_damp_sim(f_ext)

        # saving the points for next-iteration
        # points = []
        # for n_i in range(n):
        #     X = (position[f"x{n_i + 1}"].iloc[i])
        #     Y = (position[f"y{n_i + 1}"].iloc[i])
        #     Z = (position[f"z{n_i + 1}"].iloc[i])
        #     points.append(np.array([X,Y,Z]))
        # points = np.array(points)

        points = np.array([
                [position[f'x{n_i + 1}'].iloc[i], 
                position[f'y{n_i + 1}'].iloc[i], 
                position[f'z{n_i + 1}'].iloc[i]]
                for n_i in range(n)
        ])

        residual_f = np.abs(psystem.f_int[3:-3])

        if np.linalg.norm(residual_f) <= 1e-3:
            print("Classic PS converged")
            break

    stop_time = time.time()

    # start_time2 = time.time()
    # for step in t_vector:  # propagating the simulation for each timestep and saving results
    #     position2.loc[step], _ = psystem2.kin_damp_sim(f_ext)
    #
    #     residual_f = np.abs(psystem2.f_int[3:-3])
    #     if np.linalg.norm(residual_f) <= 1e-3:
    #         print("Kinetic damping PS converged")
    #         break
    # stop_time2 = time.time()

    print(f'PS classic: {(stop_time - start_time):.4f} s')
    # print(f'PS kinetic: {(stop_time2 - start_time2):.4f} s')

    # print(position)

    # plotting & graph configuration
    # Data from layout after 1 iteration step
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(position[f"x{i + 1}"].iloc[0])
        Y.append(position[f"y{i + 1}"].iloc[0])
        Z.append(position[f"z{i + 1}"].iloc[0])

    fig= plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax2 = fig.add_subplot(1, 2, 1, projection="3d")

    # ensuring the axis are scaled properly
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_zlim(0, 10)

    b = np.nonzero(np.triu(input.c_matrix))
    b = np.column_stack((b[0], b[1]))

    # data from final timestep
    X_f = []
    Y_f = []
    Z_f = []
    for i in range(n):
        X_f.append(position[f"x{i + 1}"].iloc[-1])
        Y_f.append(position[f"y{i + 1}"].iloc[-1])
        Z_f.append(position[f"z{i + 1}"].iloc[-1])

    

    # plot inital layout
    ax.scatter(X, Y, Z, c='red')
    for indices in b:
        ax.plot([X[indices[0]], X[indices[1]]], [Y[indices[0]], Y[indices[1]]], [Z[indices[0]], Z[indices[1]]],
                color='black')

    # plot final found shape
    ax2.scatter(X_f, Y_f, Z_f, c='red')
    for indices in b:
        ax2.plot([X_f[indices[0]], X_f[indices[1]]], [Y_f[indices[0]], Y_f[indices[1]]], [Z_f[indices[0]],
                Z_f[indices[1]]], color='black')

    # surf = ax.plot_surface(X, Y, Z)#, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # plt.xlabel("time [s]")
    # plt.ylabel("position [m]")
    # plt.title("Validation PS framework, deflection of particles by wind flow, with Implicit Euler scheme")
    # plt.legend([f"displacement particle {i + 1}" for i in range(n)] + [f"kinetic damped particle {i + 1}" for i in range(n)])
    # plt.grid()

    # # saving resulting figure
    # figure = plt.gcf()
    # figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper
    #
    # # Not sure if this is the smartest way to automate saving results relative to other users directories
    # file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/" \
    #                           "tether_deflection_windFlow/"
    # img_name = f"{input.params['n']}Particles-{input.params['k_t']}stiffness-{input.params['c']}damping_coefficient-" \
    #            f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    # plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()
    return


if __name__ == "__main__":
    ps = instantiate_ps()
    ps2 = instantiate_ps()

    plot(ps, ps2)

# %%
