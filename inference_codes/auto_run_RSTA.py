
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)



def singleRSTA(filename,graph_type,t,node,kth_fold,l_norm):
  try:
    with open("../outputs/%s_%s_%s_f%s_l%s_RSTAr.log" % (filename,graph_type,t,kth_fold,l_norm)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s' %( node,filename,graph_type,t,kth_fold,l_norm))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s' %( node,filename,graph_type,t,kth_fold,l_norm))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /home/group/urenzyme/workspace/colt2014/experiments/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_f%s_l%s_RSTAr' """ % (node,filename,graph_type,t,kth_fold,l_norm,filename,graph_type,t,kth_fold,l_norm) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s' %( node,filename,graph_type,t,kth_fold,l_norm))
    time.sleep(5)
  pass

def run():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run=0

  #filenames=['emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50','toy100'] 
  filenames=['toy10','toy50','emotions','yeast','scene']
  n=0
  for filename in filenames:
    for graph_type in ['tree']:
      for l_norm in ['1','2']:
        for t in range(0,61,20):
        #for t in [0]:	
          if t==0:
            t=1
          para_t="%d" % (t)
          for kth_fold in ['1','2','3','4','5']:
            node=cluster[n%len(cluster)]
            n+=1
            p=multiprocessing.Process(target=singleRSTA, args=(filename,graph_type,para_t,node,kth_fold,l_norm,))
            jobs.append(p)
            p.start()
            time.sleep(10) # fold
            pass
        time.sleep(300) # t
        pass
      time.sleep(300*is_main_run) # lnorm
      pass
    time.sleep(300*is_main_run) # tree
    for job in jobs:
      job.join()
      pass
    pass


run()


