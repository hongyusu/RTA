


import re
import os


names = ['cal500','toy10','toy50','emotions','scene','enron','yeast','ArD20','ArD30','medical','cancer','fp']
ts = range(0,41,5)
ts[0]=1
kappas = [2,4,8,16,20,24,32,40]
#ts=[30]
#kappas=[20]
fs = [1,2,3,4,5]
folders=[1,2,3,4,5,6,7,8,9,10,11,12]
suffixs=[1,2]
cs=['100','50','20','10','5','1','0.5','0.1','0.05','0.01']


for name in names:
  for f in fs:
    for t in ts:
      for kappa in kappas:
        for folder in folders:
          if folder==1:
            folderstr = 'outputs'
          if folder==2:
            folderstr = 'outputs/phase5'
          if folder==3:
            folderstr = 'outputs/phase6'
          if folder==4:
            folderstr = 'outputs/phase7'
          if folder==5:
            folderstr = 'outputs/phase8'
          if folder==6:
            folderstr = 'outputs/phase9'
          if folder==7:
            folderstr = 'outputs/phase10'
          if folder==8:
            folderstr = 'outputs/phase11'
          if folder==9:
            folderstr = 'outputs/phase12'
          if folder==10:
            folderstr = 'outputs/phase13'
          if folder==11:
            folderstr = 'outputs/phase14'
          if folder==12:
            folderstr = 'outputs/phase15'
          for suffix in suffixs:
            for c in cs:
              if suffix==1:
                suffixstr = 'RSTAs'
              if suffix==2:
                suffixstr = 'RSTAr'
              logfilename = '../%s/%s_tree_%d_f%d_l2_k%d_c%s_%s.log' % (folderstr,name,t,f,kappa,c,suffixstr)
              if not os.path.isfile(logfilename):
                #print logfilename
                continue
              #print logfilename
              fin = open(logfilename,'r')
              fbuff = fin.readlines()
              fbuff = re.sub(r'\(|\)','',fbuff[len(fbuff)-1].strip())
              fbuff = fbuff.split(' ')
              print name,f,folder,suffix,c,t,kappa,fbuff[5],fbuff[8],fbuff[11],fbuff[14]
              fin.close
              pass
            pass
          pass
        pass
      pass
    pass
  pass



