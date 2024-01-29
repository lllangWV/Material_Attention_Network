import os
import csv
from pymatgen.core import Structure


from matattnet.utils import LOGGER, DATA_DIR
def process_xyz_files(directory):
    structures = []
    targets = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                # Read the .xyz file and create a Structure object
                lines = file.readlines()
                lattice = [[float(x) for x in lines[2].split()]]
                species = [line.split()[0] for line in lines[2:]]
                coords = [[float(x) for x in line.split()[1:4]] for line in lines[2:]]
                structure = Structure(lattice, species, coords)

                # Append the Structure object and target value to the lists
                structures.append(structure)
                targets.append(lines[1].strip())

    # Save the structures and targets into a CSV file
    csv_filepath = os.path.join(directory, "structures.csv")
    with open(csv_filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Structure", "Target"])
        for structure, target in zip(structures, targets):
            writer.writerow([structure, target])

    print("Structures and targets saved to structures.csv")


if __name__ == "__main__":
    # Example usage
    process_xyz_files("/path/to/xyz_files_directory")
