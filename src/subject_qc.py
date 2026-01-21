import os
import glob
from pathlib import Path
import pandas as pd 


def count_subjects(name, root_dir):

    # This function counts the number of subjects we have in all
    # Input: Project name and folder location
    # Output: Number of subjects

    root = Path(root_dir)

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder path: {root_dir}")

    # required patterns relative to each subject folder
    if name == "VELAS": 
        required_patterns = [
            "anat/sub-*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "anat/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.json",
            "anat/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
            "anat/sub-*_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz",
            "func/sub-*_task-rest_desc-confounds_timeseries.tsv",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.json",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        ]

    elif name == "STRICON":
        required_patterns = [
            "anat/sub-*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
            "anat/sub-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.json",
            "anat/sub-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
            "anat/sub-*_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz",
            "func/sub-*_task-rest_desc-confounds_timeseries.tsv",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json",
            "func/sub-*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        ]


    subjects = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    #print(f"Found {len(subjects)} subject folders starting with 'sub-'")

    valid_subject_ids = []

    for subj_path in subjects:
        all_ok = True

        for pat in required_patterns:
            matches = glob.glob(str(subj_path / pat))
            if len(matches) == 0:
                all_ok = False
                break

        if all_ok:
            subj_id = subj_path.name.replace("sub-", "", 1)
            valid_subject_ids.append(subj_id)

    return valid_subject_ids

def FD (name, root_dir, subj_id):
    
    # This function removes subjects with high FD motion,
    # Input: Project name and folder location and the list of subject Ids.
    # Output: Number of subjects left after removing the problematic ones.

        root = Path(root_dir)

        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid folder path: {root_dir}")

        # required patterns relative to each subject folder
        confounds_file =  "func/sub-*_task-rest_desc-confounds_timeseries.tsv"
        


def main ():

    ### Define Stuff ----------------------------------
    stricon_folder_location = "/mnt/nfs/stricon_data/stricon_resting_state/derivatives_new"
    velas_folder_location = "/mnt/nfs/stricon_data/VELAS_data"

    sus_subjects_stricon = [19,21,48]  # Solely based on Registration
    sus_subjects_velas = [75,400,489,724,800,1031,1220,1344,1347,1403,2101,2210,2218,2222]  # Based on Registration alone

    ###------------------------------------

    total_stricon_folders = count_subjects("STRICON", stricon_folder_location)
    total_velas_folder = count_subjects("VELAS", velas_folder_location)

    print(f"For Stricon we have {len(total_stricon_folders)} and for VELAS we have {len(total_velas_folder)}")

    


main()


