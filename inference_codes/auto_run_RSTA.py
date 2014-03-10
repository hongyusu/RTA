
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)



def singleRSTA(filename,graph_type,t,node,kth_fold):
  try:
    with open("../outputs/%s_%s_%s_f%s_RSTAr.log" % (filename,graph_type,t,kth_fold)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s' %( node,filename,graph_type,t,kth_fold))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s' %( node,filename,graph_type,t,kth_fold))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /home/group/urenzyme/workspace/colt2014/experiments/inference_codes/; rm -rf /var/tmp/.matlab; mkdir /var/tmp/.matlab; export MATLABPATH=/var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "run_RSTA '%s' '%s' '%s' '0' '%s'" > /var/tmp/tmp_%s_%s_%s_f%s_RSTAr' """ % (node,filename,graph_type,t,kth_fold,filename,graph_type,t,kth_fold) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s' %( node,filename,graph_type,t,kth_fold))
    time.sleep(5)
  pass

def run():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run=0

  #filenames=['emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50','toy100'] 
  filenames=['toy10']#,'toy50']
  n=0
  for filename in filenames:
    for graph_type in ['tree']:
      for t in range(0,101,20):
      #for t in [0]:	
        if t==0:
          t=1
        para_t="%d" % (t)
        for kth_fold in ['1','2','3','4','5']:
          node=cluster[n%len(cluster)]
          n+=1
          p=multiprocessing.Process(target=singleRSTA, args=(filename,graph_type,para_t,node,kth_fold,))
          jobs.append(p)
          p.start()
          time.sleep(2)
      time.sleep(10)
    time.sleep(600*is_main_run)

  for job in jobs:
    job.join()
  pass


run()


