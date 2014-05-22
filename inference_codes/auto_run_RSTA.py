
import os
import sys
import commands
sys.path.append('/cs/taatto/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)



def singleRSTA(filename,graph_type,t,node,kth_fold,l_norm,kappa):
  try:
    with open("../outputs/%s_%s_%s_f%s_l%s_k%s_RSTAr.log" % (filename,graph_type,t,kth_fold,l_norm,kappa)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /cs/taatto/group/urenzyme/workspace/colt2014/experiments/random_spanning_tree_approximation/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_f%s_l%s_k%s_RSTAr' """ % (node,filename,graph_type,t,kth_fold,l_norm,kappa,filename,graph_type,t,kth_fold,l_norm,kappa) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
    time.sleep(5)
  pass

def run():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run=1

  #filenames=['emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50','toy100'] 
  filenames=['toy10','emotions','yeast','medical','scene','enron','toy50','ArD10','ArD15','ArD20','ArD30']#,'fp','cancer','cal500']
  filenames=['ArD10','ArD15','ArD20','ArD30','toy10','toy50','emotions','yeast','medical','scene','enron']#'toy50','ArD10','ArD15','ArD20','ArD30']#,'fp','cancer','cal500']
  filenames=['cancer','ArD20','ArD30','toy10','toy50','emotions','yeast','medical','scene','enron','cal500']#,'fp','cancer']
  n=0
  for kth_fold in ['1']:#,'2','3','4','5']:
    for filename in filenames:
      graph_type = 'tree'
      for kappa in ['2','4','8','16','20']:
        for l_norm in ['2']:
          for t in range(0,41,5):
            if t==0:
              t=1
            para_t="%d" % (t)
            try:
              with open("../outputs/%s_%s_%s_f%s_l%s_k%s_RSTAr.log" % (filename,graph_type,para_t,kth_fold,l_norm,kappa)): pass
              continue
            except:
              node=cluster[n%len(cluster)]
              n+=1
              p=multiprocessing.Process(target=singleRSTA, args=(filename,graph_type,para_t,node,kth_fold,l_norm,kappa,))
              jobs.append(p)
              p.start()
              time.sleep(10*is_main_run) # fold
            pass
          time.sleep(10*is_main_run) # l norm
          pass
      time.sleep(10*is_main_run) # file
      pass
    time.sleep(60*is_main_run) # t
    pass
  for job in jobs:
    job.join()
    pass



run()


