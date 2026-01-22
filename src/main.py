import argparse
import yaml
from subject_qc import count_subjects, FD_exclusion, final_subject_list
from pathlib import Path



def load_config():
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc", action="store_true",help="prepares a list of subjects which are suitable for analysis")
    

    args = parser.parse_args()

    #Load the config file:
    cfg = load_config()

    # Step:1 Count the number of subjects we have
    if args.preproc:

        print("Preparing a list of all subjects suitable for analysis")
        

        
        total_stricon_folders = count_subjects("STRICON", cfg['stricon_folder_location'])
        total_velas_folder = count_subjects("VELAS", cfg['velas_folder_location'])

        print(f"For Stricon we have {len(total_stricon_folders)} and for VELAS we have {len(total_velas_folder)}")


        bad_stricon_subjects = FD_exclusion("STRICON", cfg['stricon_folder_location'])
        bad_velas_subjects = FD_exclusion("VELAS", cfg['velas_folder_location'])

        print(f"For Stricon we removed {len(bad_stricon_subjects)} subjects and they are {bad_stricon_subjects}")
        print(f"For VELAS we removed {len(bad_velas_subjects)} subjects and they are {bad_velas_subjects}")

        total_bad_subjects_stricon = cfg['sus_subjects_stricon'] + bad_stricon_subjects
        total_bad_subjects_velas = cfg['sus_subjects_velas'] + bad_velas_subjects

        final_stricon_subjects = final_subject_list("STRICON", cfg['stricon_folder_location'], total_bad_subjects_stricon)
        final_velas_subjects = final_subject_list("VELAS", cfg['velas_folder_location'], total_bad_subjects_velas)

        print(f'Finally we are left with {len(final_stricon_subjects)} STRICON subjects and {len(final_velas_subjects)} VELAS subjects')



if __name__ == "__main__":
    main()

