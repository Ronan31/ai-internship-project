# Author: Ronan Cantin
# Creation Date: 13/11/23
# Description: Reduce dataset

import fire


################################Main#############################################
def main(
        dataset_path,
        new_dataset_path,
        nb_lines,
):
    try:
        # Open the input file for reading
        with open(dataset_path, 'r') as in_file:
            # Read all lines from the input file
            lines = in_file.readlines()

            # Check if the number of lines to collect is valid
            if nb_lines > len(lines):
                print(
                    "Error: The number of lines to collect is greater than the total number of lines in the file.")
                return

            # Open the output file for writing
            with open(new_dataset_path, 'w') as out_file:
                # Write the specified number of lines to the output file
                out_file.writelines(lines[:nb_lines])

        print(f"{nb_lines} lines collected successfully from '{dataset_path}' and saved to '{new_dataset_path}'.")
    except FileNotFoundError:
        print("Error: Input file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fire.Fire(main)
