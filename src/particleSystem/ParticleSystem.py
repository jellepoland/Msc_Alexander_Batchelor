"""
ParticleSystem framework
...
"""
import numpy as np
import numpy.typing as npt
from src.particleSystem.Particle import Particle
from src.particleSystem.SpringDamper import SpringDamper
from scipy.sparse.linalg import bicgstab


class ParticleSystem:
    def __init__(self, connectivity_matrix: npt.ArrayLike, initial_conditions: npt.ArrayLike,
                 sim_param: dict):
        """
        Constructor for ParticleSystem object, model made up of n particles
        :param connectivity_matrix: sparse n-by-n matrix, where an 1 at index (i,j) means
                                    that particle i and j are connected
        :param initial_conditions: Array of n arrays to instantiate particles. Each array must contain the information
                                   required for the particle constructor: [initial_pos, initial_vel, mass, fixed: bool]
        :param sim_param: Dictionary of other parameters required (k, l0, dt, ...)
        """

        ##TODO: don't need to define these in here
        ## can also just call them within the functions
        # e.g. self__k = sim_param["k"] is only used in the __one_d_force_vector function
        # so we can just call it there and leave it out of the beginning
        # THIS does create import problems, just calling self is easier for now

        # particles
        # self.__position 
        # self.__velocity
        # self.__mass
        # self.__fixed
        ##TODO: add a particle specific damping coefficient

        # springdampers
        self.__connectivity_matrix = np.array(connectivity_matrix)
        self.__k = sim_param["k"]
        self.__l0 = sim_param["l0"]
        self.__is_compression = sim_param["is_compression"]
        self.__is_tension = sim_param["is_tension"]
        self.__is_pulley = sim_param["is_pulley"]
        self.__pulley_other_line_pair= sim_param["pulley_other_line_pair"]
        self.__is_rotational = sim_param["is_rotational"]

        # other
        self.__dt = sim_param["dt"]
        self.__g = sim_param["g"]
        self.__n = sim_param["n"]

        ##TODO: split damping particle & spring-damping specific
        self.__c = sim_param["c"]
        
        self.__rtol = sim_param["rel_tol"]
        self.__atol = sim_param["abs_tol"]
        self.__maxiter = sim_param["max_iter"]

        # allocate memory
        self.__particles = []
        self.__springdampers = []
        self.__f = np.zeros((self.__n * 3, ))
        self.__jx = np.zeros((self.__n * 3, self.__n * 3))
        self.__jv = np.zeros((self.__n * 3, self.__n * 3))

        self.__instantiate_particles(initial_conditions)
        self.__m_matrix = self.__construct_m_matrix()
        self.__instantiate_springdampers()

        # Variables required for kinetic damping & residual force damping
        self.__w_kin = self.__calc_kin_energy()
        self.__w_kin_min1 = self.__calc_kin_energy()
        self.__w_kin_min2 = self.__calc_kin_energy()
        self.__vis_damp = True
        self.__x_min1 = np.zeros(self.__n, )
        self.__x_min2 = np.zeros(self.__n, )

        return

    def __str__(self):
        print("ParticleSystem object instantiated with attributes\nConnectivity matrix:")
        print(self.__connectivity_matrix)
        print("Instantiated particles:")
        n = 1
        for particle in self.__particles:
            print(f" p{n}: ", particle)
            n += 1
        return ""

    def __instantiate_particles(self, initial_conditions):
        for set_of_initial_cond in initial_conditions:
            x = set_of_initial_cond[0]
            v = set_of_initial_cond[1]
            m = set_of_initial_cond[2]
            f = set_of_initial_cond[3]
            self.__particles.append(Particle(x, v, m, f))
        return

    def __instantiate_springdampers(self):
        b = np.nonzero(np.triu(self.__connectivity_matrix))
        self.__b = np.column_stack((b[0], b[1]))
        for i,index in enumerate(self.__b):
            self.__springdampers.append(SpringDamper(self.__particles[index[0]], self.__particles[index[1]],
                                        self.__k[i], self.__l0[i], self.__c, self.__dt, self.__is_compression[i],
                                        self.__is_tension[i], self.__is_pulley[i],self.__is_rotational[i]))
        return

    def __construct_m_matrix(self):
        matrix = np.zeros((self.__n * 3, self.__n * 3))

        for i in range(self.__n):
            matrix[i*3:i*3+3, i*3:i*3+3] += np.identity(3)*self.__particles[i].m

        return matrix

    def __calc_kin_energy(self):
        v = self.__pack_v_current()
        w_kin = np.matmul(np.matmul(v.T, self.__m_matrix), v)      # Kinetic energy, 0.5 constant can be neglected
        return w_kin

    def simulate(self, f_external: npt.ArrayLike = ()):
        if not len(f_external):
            f_external = np.zeros(self.__n * 3, )
        f = self.__one_d_force_vector() + f_external

        v_current = self.__pack_v_current()
        x_current = self.__pack_x_current()

        jx, jv = self.__system_jacobians()

        # constructing A matrix and b vector for solver
        A = self.__m_matrix - self.__dt * jv - self.__dt ** 2 * jx
        b = self.__dt * f + self.__dt ** 2 * np.matmul(jx, v_current)

        # checking conditioning of A and b
        # print("conditioning A:", np.linalg.cond(A))

        for i in range(self.__n):
            if self.__particles[i].fixed == True:
                A[i * 3: (i + 1) * 3] = 0        # zeroes out row i to i + 3
                A[:, i * 3: (i + 1) * 3] = 0     # zeroes out column i to i + 3
                b[i * 3: (i + 1) * 3] = 0        # zeroes out row i

        # BiCGSTAB from scipy library
        dv, _ = bicgstab(A, b, tol=self.__rtol, atol=self.__atol, maxiter=self.__maxiter)

        v_next = v_current + dv
        x_next = x_current + self.__dt * v_next

        # function returns the pos. and vel. for the next timestep, but for fixed particles this value doesn't update!
        self.__update_x_v(x_next, v_next)
        return x_next, v_next

    def kin_damp_sim(self, f_ext: npt.ArrayLike):       # kinetic damping alghorithm
        if self.__vis_damp == True:         # Condition resetting viscous damping to 0
            self.__c = 0
            self.__springdampers = []
            self.__instantiate_springdampers()
            self.__vis_damp = False

        if len(f_ext):              # condition checking if an f_ext is passed as argument
            self.__save_state()
            x_next, v_next = self.simulate(f_ext)
        else:
            self.__save_state()
            x_next, v_next = self.simulate()

        w_kin_new = self.__calc_kin_energy()

        if w_kin_new > self.__w_kin:
            self.__update_w_kin(w_kin_new)
        else:
            v_next = np.zeros(self.__n*3, )
            self.__update_x_v(x_next, v_next)
            self.__update_w_kin(w_kin_new)

        return x_next, v_next

    def __pack_v_current(self):
        return np.array([particle.v for particle in self.__particles]).flatten()

    def __pack_x_current(self):
        return np.array([particle.x for particle in self.__particles]).flatten()

    def __one_d_force_vector(self):
        self.__f[self.__f != 0] = 0

        # print(self.__b)

        for idx in range(len(self.__springdampers)):
            
            # Pulleys
            if self.__is_pulley[idx] == True:
                idx_p3,idx_p4, rest_length_p3p4 = self.__pulley_other_line_pair[str(idx)]
                points = self.__pack_x_current()
                p3 = points[idx_p3]
                p4 = points[idx_p4]
                norm_p3p4 = np.linalg.norm(p3 - p4)
                extra_rest_length = norm_p3p4 - rest_length_p3p4
                fs, fd = self.__springdampers[idx].force_value(extra_rest_length)
            else:
                fs, fd = self.__springdampers[idx].force_value()
            
            i, j = self.__b[idx]

            self.__f[i*3: i*3 + 3] += fs + fd
            self.__f[j*3: j*3 + 3] -= fs + fd

        return self.__f

    def __system_jacobians(self):
        self.__jx[self.__jx != 0] = 0
        self.__jv[self.__jv != 0] = 0

        # print(np.shape(self.__jx), np.shape(self.__jv))
        # print(self.__n)
        # print(len(self.__springdampers))
        # print(self.__connectivity_matrix)
        # b = np.nonzero(np.triu(self.__connectivity_matrix))
        # b = np.column_stack((b[0], b[1]))
        # print(b)
        # print(len(b))

        for n in range(len(self.__springdampers)):
            jx, jv = self.__springdampers[n].calculate_jacobian()
            i, j = self.__b[n]

            self.__jx[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jx
            self.__jx[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jx
            self.__jx[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jx
            self.__jx[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jx

            self.__jv[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jv
            self.__jv[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jv
            self.__jv[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jv
            self.__jv[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jv

        # i = 0
        # j = 1
        # for springdamper in self.__springdampers:
        #     jx, jv = springdamper.calculate_jacobian()
        #     if j > 40:
        #         print(i, j)
        #         print(j * 3, j * 3 + 3, j * 3, j * 3 + 3)
        #         print()
        #     self.__jx[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jx
        #     self.__jx[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jx
        #     self.__jx[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jx
        #     self.__jx[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jx
        #
        #     self.__jv[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jv
        #     self.__jv[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jv
        #     self.__jv[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jv
        #     self.__jv[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jv
        #
        #     i += 1
        #     j += 1

        return self.__jx, self.__jv

    def __update_x_v(self, x_next: npt.ArrayLike, v_next: npt.ArrayLike):
        for i in range(self.__n):
            self.__particles[i].update_pos(x_next[i * 3:i * 3 + 3])
            self.__particles[i].update_vel(v_next[i * 3:i * 3 + 3])
        return

    def __update_w_kin(self, w_kin_new: float):
        self.__w_kin_min2 = self.__w_kin_min1
        self.__w_kin_min1 = self.__w_kin
        self.__w_kin = w_kin_new
        return

    # def __update_f_res(self, f_res_new: float):
    #     self.__f_res_min2 = self.__f_res_min1
    #     self.__f_res_min1 = self.__f_res
    #     self.__f_res = f_res_new
    #     return

    def __save_state(self):
        self.__x_min2 = self.__x_min1
        self.__x_min1 = self.__pack_x_current()
        return

    @property
    def particles(self):            # Temporary solution to calculate external aerodynamic forces
        return self.__particles

    @property
    def springdampers(self):
        return self.__springdampers

    @property
    def stiffness_m(self):
        self.__system_jacobians()
        return self.__jx

    @property
    def f_int(self):
        return self.__f


if __name__ == "__main__":

    c_matrix = [[0, 1], [1, 0]]
    init_cond = [[[0, 0, 0], [0, 0, 0], 1, True], [[0, 0, 0], [0, 0, 0], 1, False]]

    params = {
        # model parameters
        "n": 2,  # [-] number of particles
        "k": 2e4,  # [N/m] spring stiffness
        "c": 0,  # [N s/m] damping coefficient
        "l0": 0,  # [m] rest length

        # simulation settings
        "dt": 0.001,  # [s] simulation timestep
        "t_steps": 1000,  # [-] number of simulated time steps
        "abs_tol": 1e-50,  # [m/s] absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-] relative error tolerance iterative solver
        "max_iter": 1e5,  # [-] maximum number of iterations

        # physical parameters
        "g": 9.81           # [m/s^2] gravitational acceleration
    }

    ps = ParticleSystem(c_matrix, init_cond, params)
    print(ps)
    print(ps.system_energy)
    pass
