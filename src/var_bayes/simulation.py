import h5py
import time
import numpy as np
from pathlib import Path

from src.var_bayes.fwd_ode import FwdOde
from src.var_bayes.bwd_ode import BwdOde

from src.dynamics.lorenz_63 import Lorenz63
from src.dynamics.lorenz_96 import Lorenz96
from src.dynamics.double_well import DoubleWell
from src.dynamics.ornstein_uhlenbeck import OrnsteinUhlenbeck

from src.var_bayes.variational import VarGP
from src.var_bayes.prior_kl0 import PriorKL0
from src.var_bayes.gaussian_like import GaussianLikelihood
from src.numerics.optim_scg import SCG

# Define a dictionary with all the dynamical systems.
dynamical_systems = {"DW": DoubleWell, "OU": OrnsteinUhlenbeck,
                     "L63": Lorenz63, "L96": Lorenz96}


class Simulation(object):
    """
    Main simulation class. The normal workflow is as follows:

    1) Create a simulation object. You can also give a name
    that will be used when saving the data.

        >> sim_vgpa = Simulation("Sim_01")

    2) Set up its simulation parameters.

        >> sim_vgpa.setup(params, data)

    3) Run the simulation (this step might take a while).

        >> sim_vgpa.run()

    4) Finally save the results in a hdf5 (compressed) file.

        >> sim_vgpa.save()
    """

    __slots__ = ("name", "m_data", "output")

    def __init__(self, name: str = None) -> None:
        """
        Default constructor of the Simulation class.

        :param name: (string) is optional, but it will
        be used for constructing a meaningful filename
        to save the results at the end of the simulation.
        """
        # Check if a simulation name has been given.
        self.name = str(name) if name else "ID_None"

        # This dictionary will hold the simulation data.
        self.m_data = {}

        # Placeholder for the output storage.
        self.output = {}
    # _end_def_

    @classmethod
    def _stochastic_model(cls, model: str, *args):
        """
        Auxiliary (class) method that returns
        an instance of the requested stochastic
        model.

        :param model: (string) type of the model.

        :param args: additional parameters for the
        constructors. (noise, drift, random seed).

        :return: an instance of the model.

        :raises ValueError: if the model is unknown.
        """
        # Make input string upper-case.
        model_upper = str(model).upper()

        try:
            return dynamical_systems[model_upper](*args)
        except KeyError:
            raise ValueError(f" {cls.__name__}:"
                             f" Unknown stochastic model -> {model_upper}")
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
        self.m_data["mu0"] = params["Prior"]["mu0"]
        self.m_data["tau0"] = params["Prior"]["tau0"]

        # Stochastic model parameters.
        system_noise = self.m_data["noise"]["sys"]
        drift_vector = self.m_data["drift"]["theta"]
        random_seed_ = self.m_data["random_seed"]

        # Stochastic model.
        self.m_data["model"] = Simulation._stochastic_model(params["Model"],
                                                            system_noise,
                                                            drift_vector,
                                                            random_seed_)
        # Get the dimensionality flag from the "model".
        self.m_data["single_dim"] = self.m_data["model"].single_dim

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
            self.m_data["mu0"] *= np.ones(dim_d)
            self.m_data["tau0"] *= np.eye(dim_d)
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
        options = {"max_it": 500, "x_tol": 1.0e-6, "f_tol": 1.0e-8,
                   "display": True}

        # Create an SCG optimization object.
        optimize = SCG(vgpa.free_energy, vgpa.gradient, options)

        # Get the initial vector.
        x0 = vgpa.initialization()

        # Initial gradient test.
        # optimize.check_gradient_function(x0.copy())

        # Start the timer.
        time_t0 = time.perf_counter()

        # Run the optimization procedure.
        x, fx = optimize(x0.copy())

        # Stop the timer.
        time_tf = time.perf_counter()

        # Final gradient test.
        # optimize.check_gradient_function(x.copy())

        # Print final duration in seconds.
        print(f" Elapsed time: {(time_tf - time_t0):.2f} seconds.", end='\n')

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
            print(f" {self.__class__.__name__}:"
                  f" Simulation data structure 'output' is empty.")
        else:
            # Initial message.
            print(f" Saving the results to: {self.name}.h5")

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
            sim_data[key] = np.array(input_file[key], dtype=object)
        # _end_for_

    # _end_with_

    # Return the dictionary.
    return sim_data
# _end_def_
