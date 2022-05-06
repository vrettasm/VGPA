#!/usr/bin/env python3

import sys
import json
import pandas as pd
from pathlib import Path
from src.var_bayes.simulation import Simulation

# INFO:
__author__ = "Michail Vrettas, PhD"
__email__ = "michail.vrettas@gmail.com"


def validateInputParametersFile(filename):
    """
    Validates an input (json) file to check if
    it contains the required keys. It does not
    validate the values of the keys.

    :param filename: Is a "Path" object that
    contains the input model parameters for the
    simulation.

    :return: A dictionary loaded from the input
    file.

    :raises ValueError: if a keyword is missing
    from the file.
    """

    # Open the file in "Read Only" mode.
    with open(filename, 'r') as input_file:

        # Load the model parameters.
        model_params = json.load(input_file)

        # Required keys in the json file.
        required_keys = {"Output_Name",  "Model",  "Ode-method",
                         "Time-window", "Noise", "Observations",
                         "Drift", "Prior", "Random-Seed"}

        # Check the keywords for membership.
        for k in required_keys:

            # The order in here doesn't matter.
            if k not in model_params:
                raise ValueError(f" Key: {k}, is not given.")
            # _end_if_

        # _end_for_

        # Show message.
        print(" Model parameters are given correctly.")
    # _end_with_

    # Dictionary will contain all the input parameters.
    return model_params
# _end_def_


# Main function.
def main(params_file=None, data_file=None):
    """
    As the name suggests, this is the main function
    that is called to initiate the simulation run.

    :param params_file: (string) that points to the
    input file for the parameters.

    :param data_file: (string) that points to the
    input file for the observations (optional).

    :return: None.
    """

    # Check if we got model parameters.
    if params_file is not None:
        try:
            # Make sure params_file is a Path object.
            params_file = Path(params_file)

            # Check if everything is ok.
            params = validateInputParametersFile(params_file)
        except ValueError as e0:
            # Show the error message.
            print(e0)

            # Exit the program.
            sys.exit(1)
        # _end_try_
    else:
        print(" The simulation can't run without input parameters.")

        # Exit the program.
        sys.exit(1)
    # _end_if_

    # Set this to None.
    obs_data = None

    # Check if we have given data separately.
    if data_file is not None:
        # Make sure it's a Path object.
        data_file = Path(data_file)

        # Display where we got the observational data.
        print(f" Simulation observational data file: {data_file}")

        # Open the data in "Read Only" mode.
        with open(data_file, 'r') as input_file:
            # The file should have two columns.
            obs_data = pd.read_csv(input_file,
                                   names=["t", "Yt"])
        # _end_with_
    # _end_if_

    # Get the output name from the file.
    output_name = params["Output_Name"]

    # If nothing has been given set a
    # default name.
    if output_name is None:
        output_name = "Sim_00"
    # _end_if_

    try:
        # Create a simulation object.
        sim_vgpa = Simulation(output_name)

        # Setup parameters (initialization).
        sim_vgpa.setup(params, obs_data)

        # Run the simulation (smoothing).
        sim_vgpa.run()

        # Save the results.
        sim_vgpa.save()
    except Exception as e1:
        print(e1)

        # Exit the program.
        sys.exit(1)
    # _end_try_

# _end_main_


# Run the script.
if __name__ == "__main__":

    # Check if we have given input parameters.
    if len(sys.argv) > 1:
        # Local import.
        import argparse

        # Create a parser object
        parser = argparse.ArgumentParser(description=" VGPA (1.0) ")

        # Input file with simulation parameters.
        parser.add_argument("--params",
                            help=" Input file (.json) with simulation parameters.")

        # Input file with simulation data.
        parser.add_argument("--data",
                            help=" Input file (.csv) with observational data.")

        # Parse the arguments.
        args = parser.parse_args()

        # Call the main function.
        main(args.params, args.data)

        # Display final info.
        print(' Simulation completed.')
    else:
        sys.exit('Error: Not enough input parameters.')
    # _end_if_

# _end_program_
