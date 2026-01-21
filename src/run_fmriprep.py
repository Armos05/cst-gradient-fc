#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Use this to run this file - nohup python ./fmriprep_docker_run.py /mnt/data/stricon_bids > ./run.log 2>&1 &

import sys,os
import pidfile
import glob
import multiprocessing
from multiprocessing import Process
from multiprocessing import Semaphore
import subprocess

def start_fmriprep(semaphore,subject):
 with semaphore:
  print(sys.argv[1]+'/'+subject)
  subprocess.call(['fmriprep-docker','--image', 'nipreps/fmriprep:23.2.2',sys.argv[1],sys.argv[1]+'/derivatives_rsfmri_new','participant','--participant-label',subject,'--fs-license-file','/mnt/data/freesurfer_license.txt','--skip_bids_validation','--fs-no-reconall','--output-spaces', 'MNI152NLin2009cAsym:res-2','--no-tty','--nprocs','1'])


def main():
 processqueue=[]
 #subjects = glob.glob(sys.argv[1]+'/sub-*')
 subjects = ['sub-5', 'sub-7', 'sub-8', 'sub-15', 'sub-18', 'sub-89', 'sub-91', 'sub-92', 'sub-93', 'sub-94', 'sub-95', 'sub-96', 'sub-97', 'sub-98', 'sub-99', 'sub-100', 'sub-101']
 for subject in subjects:
  print(os.path.basename(subject))
  processqueue.append(os.path.basename(subject))

 # prepare multiprocessing
 concurrency = multiprocessing.cpu_count()-1
 # Change afterwards (was 8 before)
 concurrency = 2 
 procs = []
 semaphore = Semaphore(concurrency)
 # work processqueue
 for subject in processqueue:
  print("launching " + subject)
  worker = Process(target=start_fmriprep, args=(semaphore, subject))
  procs.append(worker)
  worker.start()
 for t in procs:
  t.join()




if __name__ == '__main__':
 if len(sys.argv) == 2:
  try:
   with pidfile.PIDFile():
    print('fmridocker_run.py started.')
    main()
    print('fmridocker_run.py ended.')
  except pidfile.AlreadyRunningError:
    print('fmridocker_run.py already running.')
 else:
  print('input folder missing.')

