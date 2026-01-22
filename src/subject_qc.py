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

def FD_exclusion (name, root_dir):
    
    # This function removes subjects with high FD motion,
    # Input: Project name and folder location and the list of subject Ids.
    # Output: Number of subjects left after removing the problematic ones.

        root = Path(root_dir)

        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid folder path: {root_dir}")
    
        
        # All Subjects
        subjects = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
        bad_subjects = [ ]
        # required patterns relative to each subject folder
        for subj in subjects:

            # Extracting subject number
            p = Path(subj)
            subj_num = p.name.replace("sub-", "")
            
            confounds_file = p / "func" / f"sub-{subj_num}_task-rest_desc-confounds_timeseries.tsv"
            conf_df = pd.read_csv(confounds_file, sep = '\t')

            # number of time points greater than FD threshold (0.5 times of resolution)
            percent = (conf_df['framewise_displacement'] > 0.5).mean() * 100
            #print(f'For subject {subj_num} the percentage is {percent}')

            if percent > 20:

                bad_subjects.append(subj_num)


        return bad_subjects
        


def final_subject_list(name, root_dir, exclusion_subjects):

    # This function gives the final list of usable subjects.
    # Input: Project name and folder location and the list of excluded subject Ids.
    # Output: Final subject id lists

    root = Path(root_dir)

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder path: {root_dir}")
    
    subjects_to_keep = []
    
        
    # All Subjects
    subjects = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    
    for subj in subjects:

        p = Path(subj)
        subj_num = p.name.replace("sub-", "")
        if subj_num not in exclusion_subjects:
            subjects_to_keep.append(subj)

    return subjects_to_keep


                      


