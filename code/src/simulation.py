import h5py
import numpy as np
from pathlib import Path

class Simulation(object):
    """
    Main simulation class. The normal workflow is as follow:

    1) Create a simulation object. You can also give a name
    that will be used when saving the data.

        >> sim_01 = Simulation("Sim_01")

    2) Setup its simulation parameters.

        >> sim_01.setup(params)

    3) Run the simulation (this step might take a while).

        >> sim_01.run()

    4) Finally save the results in a hdf5 (compressed) file.

        >> sim_01.save()
    """

    __slots__ = ("name", "m_data", "rng", "output")

    def __init__(self, name=None, seed=None):
        """
        Default constructor of the Simulation class.

        :param name: (string) is optional but it will be used for constructing
        a meaningful filename to save the results at the end of the simulation.

        :param seed: (int) is used to initialize the random generator. If None,
        the generator will be initialized at random from the OS.
        """
        # Check if a simulation name has been given.
        if name is not None:
            self.name = name
        else:
            self.name = "ID_None"
        # _end_if_

        # This dictionary will hold all the simulation data.
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
        pass
    # _end_def_

    def run(self):
        # Create an optimizer object.
        # optimizer = SCG()

        # Run the optimization.
        # x, fx = optimizer(fun, grad_fun)

        # Store the results.
        # self.m_data["x"] = x
        # self.m_data["fx"] = fx
        pass
    # _end_def_

    def save(self):
        pass
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
