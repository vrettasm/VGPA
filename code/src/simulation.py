import h5py
import time
import numpy as np
from pathlib import Path

from .fwd_ode import FwdOde
from .bwd_ode import BwdOde

from ..dynamics.sp_lorenz_63 import Lorenz63
from ..dynamics.sp_lorenz_96 import Lorenz96
from ..dynamics.sp_double_well import DoubleWell
from ..dynamics.sp_ornstein_uhlenbeck import OrnsteinUhlenbeck

from ..src.variational import VarGP
from ..src.prior_kl0 import PriorKL0
from ..src.gaussian_like import GaussianLikelihood

from ..numerics.optim_scg import SCG


class Simulation(object):
    """
    Main simulation class. The normal workflow is as follow:

    1) Create a simulation object. You can also give a name
    that will be used when saving the data.

        >> sim_vgpa = Simulation("Sim_01")

    2) Setup its simulation parameters.

        >> sim_vgpa.setup(params, data)

    3) Run the simulation (this step might take a while).

        >> sim_vgpa.run()

    4) Finally save the results in a hdf5 (compressed) file.

        >> sim_vgpa.save()
    """

    __slots__ = ("name", "m_data", "rng", "output")

    def __init__(self, name=None, seed=None):
        """
        Default constructor of the Simulation class.

        :param name: (string) is optional but it will
        be used for constructing a meaningful filename
        to save the results at the end of the simulation.

        :param seed: (int) is used to initialize the
        random generator. If None, the generator will
        be initialized at random from the OS.
        """

        # Check if a simulation name has been given.
        if name is not None:
            self.name = name
        else:
            self.name = "ID_None"
        # _end_if_

        # This dictionary will hold
        # all the simulation data.
        self.m_data = {}

        # Create a random number generator.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        # _end_if_

        # Place holder for the output storage.
        self.output = {}
    # _end_def_

    def setup(self, params, data):
        """
        This method is called BEFORE the run() and sets up
        all the variables for the simulation. It is also
        responsible for checking the validity of the input
        parameters before use.

        :param params: (dict) contains all the given parameters.
        If None, then it will use the default parameters.

        :param data: array with observation times and values (t, yt).

        :return: None.

        :raises ValueError: if some data are wrong.
        """

        # Extract drift parameters.
        self.m_data["drift"] = params["Drift"]

        # Extract noise parameters.
        self.m_data["noise"] = params["Noise"]

        # Extract time window.
        self.m_data["time_window"] = params["Time-window"]

        # Extract ODE solver.
        self.m_data["ode_solver"] = params["Ode-method"]

        # Extract Random seed.
        self.m_data["random_seed"] = params["Random-Seed"]

        # Extract observation setup.
        self.m_data["obs_setup"] = params["Observations"]

        # Extract prior parameters.
        self.m_data["prior"] = params["Prior"]

        # Stochastic model.
        if params["Model"].upper() == "DW":

            # Create the model.
            self.m_data["model"] = DoubleWell(self.m_data["noise"]["sys"],
                                              self.m_data["drift"]["theta"],
                                              self.m_data["random_seed"])
            # One-dimensional model.
            self.m_data["single_dim"] = True

        elif params["Model"].upper() == "OU":

            # Create the model.
            self.m_data["model"] = OrnsteinUhlenbeck(self.m_data["noise"]["sys"],
                                                     self.m_data["drift"]["theta"],
                                                     self.m_data["random_seed"])
            # One-dimensional model.
            self.m_data["single_dim"] = True

        elif params["Model"].upper() == "L63":

            # Create the model.
            self.m_data["model"] = Lorenz63(self.m_data["noise"]["sys"],
                                            self.m_data["drift"]["theta"],
                                            self.m_data["random_seed"])
            # Three-dimensional model.
            self.m_data["single_dim"] = False

        elif params["Model"].upper() == "L96":

            # Create the model.
            self.m_data["model"] = Lorenz96(self.m_data["noise"]["sys"],
                                            self.m_data["drift"]["theta"],
                                            self.m_data["random_seed"])
            # Forty-dimensional model.
            self.m_data["single_dim"] = False

        else:
            raise ValueError(" {0}: Unknown stochastic model ->"
                             " {1}".format(self.__class__.__name__, params["Model"]))
        # _end_if_

        # Make the trajectory (of the stochastic process).
        self.m_data["model"].make_trajectory(self.m_data["time_window"]["t0"],
                                             self.m_data["time_window"]["tf"],
                                             self.m_data["time_window"]["dt"])
        # This needs to be revisited.
        if data is not None:
            self.m_data["obs_t"] = data[0]
            self.m_data["obs_y"] = data[1]
        else:
            # Sample observations from the trajectory.
            obs_t, obs_y, obs_noise = self.m_data["model"].collect_obs(self.m_data["obs_setup"]["density"],
                                                                       self.m_data["noise"]["obs"],
                                                                       self.m_data["obs_setup"]["operator"])
            # Add them to the dictionary.
            self.m_data["obs_t"] = obs_t
            self.m_data["obs_y"] = obs_y
            self.m_data["obs_noise"] = obs_noise
        # _end_if_

        # Initial (marginal) moments.
        if self.m_data["single_dim"]:
            self.m_data["m0"] = self.m_data["model"].sample_path[0] +\
                                0.1 * self.m_data["model"].rng.standard_normal()
            self.m_data["s0"] = 0.2
        else:
            # Get the system dimensions.
            dim_d = self.m_data["model"].sample_path.shape[-1]

            # Initial (marginal) moments.
            self.m_data["m0"] = self.m_data["model"].sample_path[0] +\
                                0.1 * self.m_data["model"].rng.standard_normal(dim_d)
            self.m_data["s0"] = 0.2 * np.eye(dim_d)

            # Adjust the prior values.
            self.m_data["mu0"] = self.m_data["prior"]["mu0"] * np.ones(dim_d)
            self.m_data["tau0"] = self.m_data["prior"]["tau0"] * np.eye(dim_d)
        # _end_if_

    # _end_def_

    def run(self):
        """
        Optimizes the VGPA model in time. All the output information
        is store in self.output dictionary, where we can later save
        it to the disk for further analysis.

        :return: None.
        """

        # Forward ODE solver.
        fwd_ode = FwdOde(self.m_data["time_window"]["dt"],
                         self.m_data["ode_solver"],
                         self.m_data["single_dim"])

        # Backward ODE solver.
        bwd_ode = BwdOde(self.m_data["time_window"]["dt"],
                         self.m_data["ode_solver"],
                         self.m_data["single_dim"])

        # Likelihood object.
        likelihood = GaussianLikelihood(self.m_data["obs_y"],
                                        self.m_data["obs_t"],
                                        self.m_data["obs_noise"],
                                        self.m_data["obs_setup"]["operator"],
                                        self.m_data["single_dim"])
        # Prior moments.
        kl0 = PriorKL0(self.m_data["mu0"], self.m_data["tau0"],
                       self.m_data["single_dim"])

        # Variational GP model.
        vgpa = VarGP(self.m_data["model"],
                     self.m_data["m0"], self.m_data["s0"],
                     fwd_ode, bwd_ode, likelihood, kl0,
                     self.m_data["obs_y"], self.m_data["obs_t"])

        # Setup SCG options.
        options = {'max_it': 500, 'x_tol': 1.0e-6, 'f_tol': 1.0e-8}

        # Create an SCG optimization object.
        optimize = SCG(vgpa.free_energy, vgpa.gradient, options)

        # Start the timer.
        time_t0 = time.time()

        # Run the optimization procedure.
        x, fx = optimize(vgpa.initialization())

        # Stop the timer.
        time_tf = time.time()

        # Print final duration in seconds.
        print(" Elapsed time: {0:.2f} seconds.\n".format(time_tf - time_t0))

        # Unpack optimization output data.
        if self.m_data["model"].single_dim:
            # Get the dimensions.
            dim_n = self.m_data["model"].sample_path.size

            # Store to the output.
            self.output["at"] = x[:dim_n]
            self.output["bt"] = x[dim_n:]
        else:
            # Get the dimensions.
            dim_n, dim_d = self.m_data["model"].sample_path.shape

            # Total dimensions of the linear parameters.
            dim_tot = dim_n * dim_d * dim_d

            # Store to the output.
            self.output["at"] = x[:dim_tot].reshape(dim_n, dim_d, dim_d)
            self.output["bt"] = x[dim_tot:].reshape(dim_n, dim_d)
        # _end_if_

        # Store the optimization minimum.
        self.output["fx"] = fx

        # Merge the outputs in one dict.
        self.output.update(vgpa.arg_out)
    # _end_def_

    def save(self):
        """
        Saves the simulation results to a file. All the data should
        be stored inside the  self.output dictionary and be of type
        numpy.ndarray. For the moment the file is saved in the same
        directory as the main program.

        :return: None.
        """

        # Check if the output dictionary is empty.
        if not self.output:
            # Print a message and do not save anything.
            print(" {0}: Simulation data structure 'output'"
                  " is empty.".format(self.__class__.__name__))
        else:
            # Initial message.
            print(" Saving the results to: {0}.h5".format(self.name))

            # Create the output filename. Remove spaces (if any).
            file_out = Path(self.name.strip().replace(" ", "_") + ".h5")

            # Save the data to an 'HDF5' file format.
            # NOTE:  Create file; truncate if exists.
            with h5py.File(file_out, 'w') as out_file:
                # Local reference.
                data = self.output

                # Extract all the data.
                for key in data:

                    # Convert scalars to 1D arrays.
                    if np.isscalar(data[key]):
                        data[key] = np.atleast_1d(data[key])
                    # _end_if_

                    # Default compressions level is '4'.
                    out_file.create_dataset(key, data=data[key],
                                            shape=data[key].shape,
                                            compression='gzip')
                # _end_for_
            # _end_with_
        # _end_if_
    # _end_def_

# _end_class_

# Auxiliary function.
def load(filename=None):
    """
    Loads the simulation data that have been stored in the hdf5 file
    using the object's method saveResults().

    :param filename: (string) is the '.h5' file that contains the data.

    :return: a dictionary with all the data.
    """

    # Check if we have given input file.
    if filename is None:
        raise RuntimeError(" load_data: No input file is given.")
    # _end_if_

    # Simulation data.
    sim_data = {}

    # Open the file for read only.
    with h5py.File(Path(filename), 'r') as input_file:

        # Extract all the data.
        for key in input_file:
            sim_data[key] = np.array(input_file[key])
        # _end_for_

    # _end_with_

    # Return the dictionary.
    return sim_data
# _end_def_
