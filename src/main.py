import argparse
from subject_qc import count_subjects, FD_exclusion



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc", action="prepares a list of subjects which are suitable for analysis")
    

    args = parser.parse_args()

    if args.preproc:

        print("Preparing a list of all subjects suitable for analysis")



if __name__ == "__main__":
    main()

