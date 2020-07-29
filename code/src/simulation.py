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
        pass
    # _end_def_

    def save(self):
        """
        Saves the simulation results to a file. All the data should be stored
        inside the self.output dictionary and be of type "numpy.ndarray". For
        the moment the file is saved in the same directory as the main program.
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
                    # Default compressions level is '4'.
                    out_file.create_dataset(key, data=data[key], compression='gzip')
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
