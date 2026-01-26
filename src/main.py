import argparse
import yaml
from subject_qc import count_subjects, FD_exclusion, final_subject_list
from fmri_denoise import denoise_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from process_excel_file import build_roi_dataframe
from connectivity import run_dataset
from group_stats import load_groups_from_excel, load_fc_dataset, stats_stricon, stats_velas
from match_lit_trends import match_stricon, match_velas
from panss_fc_correlations import run_dataset_panss
import pandas as pd
import numpy as np
from pathlib import Path



def load_config():
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_list", action="store_true",help="prepares a list of subjects which are suitable for analysis")
    parser.add_argument("--preproc", action="store_true", help = "Does minimal pre-processing on the data")
    parser.add_argument("--out_csv",action="store_true", help="Optional path to save expanded CSV")
    parser.add_argument("--connectivity",action="store_true", help = "Find the seed based correlation between regions" )
    parser.add_argument("--load_scores",type = str, help = "Load Behaviour scores" )
    parser.add_argument("--stats",action="store_true", help = "Find the seed based correlation between regions" )

    # To perform literature match trends.
    parser.add_argument("--tests", action = "store_true", help = "Find patterns")
    parser.add_argument("--velas_csv", required=True, help="velas FC connectivity scores file")
    parser.add_argument("--stricon_csv", required=True, help="stricon FC connectivity scores file")
    

    args = parser.parse_args()
    #Load the config file:
    cfg = load_config()
    out_path = Path(cfg['out_dir'])

    # Step:1 Count the number of subjects we have
    if args.subject_list:

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

    
    if args.preproc:

        # Run if you want to pre-process data, do it only once:
        print("Running resting-state denoising for STRICON and VELAS in parallel")

        jobs = [
            dict(
                dataset_name="STRICON",
                deriv_root=final_stricon_subjects,
                out_root=cfg["stricon_preproc_folder"],
                tr = 1.17,
                hp=0.008,
                lp=0.09,
                smooth_fwhm=4.0,
                n_acompcor=5,
                overwrite=False,
            ),
            dict(
                dataset_name="VELAS",
                deriv_root=final_velas_subjects,
                out_root=cfg["velas_preproc_folder"],
                tr = 1.17,
                hp=0.008,
                lp=0.09,
                smooth_fwhm=4.0,
                n_acompcor=5,
                overwrite=False,
            ),
        ]

        # 2 workers = run STRICON + VELAS simultaneously
        with ProcessPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(denoise_dataset, **job): job["dataset_name"] for job in jobs}

            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    print(f"{name}: completed successfully.")
                except Exception as e:
                    print(f"{name}: FAILED -> {e}")
                    raise

    
    rois = build_roi_dataframe(cfg['roi_excel_file'])
    #print(rois.head(20).to_string(index=False))
    #print(f"\nTotal ROIs after expansion: {len(rois)}")

    if args.out_csv:
        out_path = Path("/mnt/nfs/stricon_data/gPPI/CST_loops/new_rois.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rois.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")

    if args.connectivity:

        print(f"processing for STRICON Data")
        run_dataset(preproc_root=cfg['stricon_preproc_folder'], roi_csv=cfg['roi_csv_file'],overwrite = True, radius_mm=4)

        print(f"processing for VELAS Data")
        run_dataset(preproc_root=cfg['velas_preproc_folder'], roi_csv=cfg['roi_csv_file'],overwrite = True,radius_mm=4)

    if args.stats:

        print(f'Loading behaviour scores')
        stricon_groups = load_groups_from_excel(args.load_scores, sheet="STRICON", group_col="Group")
        velas_groups = load_groups_from_excel(args.load_scores, sheet="VELAS", group_col="group")

        s_subs, s_rois, s_mats = load_fc_dataset(cfg['stricon_preproc_folder'])
        v_subs, v_rois, v_mats = load_fc_dataset(cfg['velas_preproc_folder'])

        # Checking NaNs
        print(f"STRICON NaNs per subject: " + ", ".join([f"{sid}:{int(np.isnan(s_mats[k]).sum())}" for k, sid in enumerate(s_subs) if np.isnan(s_mats[k]).any()]) or "none", flush=True)
        print(f"VELAS   NaNs per subject: " + ", ".join([f"{sid}:{int(np.isnan(v_mats[k]).sum())}" for k, sid in enumerate(v_subs) if np.isnan(v_mats[k]).any()]) or "none", flush=True)

        # Sanity: ROI list must match across datasets for comparable edges
        if list(s_rois) != list(v_rois):
            print("WARNING: ROI order differs between datasets. Stats will still run separately, but edges won't align across studies.")

        # Run stats
        res_s = stats_stricon(s_subs, s_mats, s_rois, stricon_groups)
        res_v = stats_velas(v_subs, v_mats, v_rois, velas_groups)

        # Save
        res_s.to_csv(out_path / "stricon_patients_vs_healthy_edges.csv", index=False)
        res_v.to_csv(out_path / "velas_trend_ls_hs_patient_edges.csv", index=False)

        # Quick top hits
        print("\nTop STRICON edges (lowest q_fdr):")
        print(res_s.head(10)[["roi_a","roi_b","direction","diff_pat_minus_hc","p","q_fdr"]].to_string(index=False))

        print("\nTop VELAS edges (lowest q_fdr_slope):")
        print(res_v.head(10)[["roi_a","roi_b","direction","slope_z","p_slope_z","q_fdr_slope"]].to_string(index=False))

    if args.tests:

        #match_stricon(args.stricon_csv, str(out_path/ "stricon_matched.csv"))
        #match_velas(args.velas_csv, str(out_path/ "velas_matched.csv"))
        
        stricon_cols = ["PANSS_P1","PANSS_P2","PANSS_P3","PANSS_P4","PANSS_P5","PANSS_P6","PANSS_P7"]
        run_dataset(excel_path=args.load_scores,sheet="STRICON",panss_cols=stricon_cols,
                    preproc_root=cfg['stricon_preproc_folder'], out_csv=str(out_path / "stricon_panss_edge_correlations.csv"),
                    require_all_for_total=args.require_all)
        
        velas_cols = ["panss_p01","panss_p02","panss_p03","panss_p04","panss_p05","panss_p06","panss_p07"]
        run_dataset(excel_path=args.load_scores,sheet="VELAS",panss_cols=velas_cols,
                    preproc_root=cfg['velas_preproc_folder'], out_csv=str(out_path / "velas_panss_edge_correlations.csv"),
                    require_all_for_total=args.require_all)


if __name__ == "__main__":
    main()

