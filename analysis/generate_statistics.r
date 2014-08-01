
library(xtable)



options(width=150)

## previous results
res.prev.1 = c(94.7,95.8,77.6,80.0,90.2,93.6,86.3,89.7,84.7,97.4,95.3,94.3,  93.2,94.6,79.8,79.3,88.4,93.5,86.2,82.7,84.0,97.4,93.7,93.8,  94.8,95.7,79.9,78.3,81.6,93.8,86.3,89.5,85.4,97.9,97.4,98.5,  95.2,96.1,80.5,79.9,83.0,95.0,86.3,89.5,85.7,97.9,97.5,97.9)
res.prev.1 = matrix(res.prev.1,12,4)
res.prev.2 = c(45.1,36.9,22.2,14.1,52.8,0.40,0.00,1.00,43.1,8.20,71.1,30.2,  35.2,28.4,25.5,11.3,44.8,0.40,0.00,0.00,47.0,8.20,66.8,27.7,  46.6,36.4,28.7,7.00,27.8,7.30,0.00,0.40,36.9,36.2,79.7,61.2,  48.8,38.4,30.4,14.0,5.40,12.1,0.00,0.40,40.0,36.9,82.3,53.8)
res.prev.2 = matrix(res.prev.2,12,4)

std.prev.1 = c(0,0,1.9,0.7,0.3,0.2,0.3,0.3,0.7,0.0,0.9,0.5,  0,0,1.8,0.5,0.5,0.2,0.3,0.6,0.6,0.1,0.7,0.5,  0,0,0.9,0.6,0.3,0.2,0.3,0.6,1.3,0.1,0.3,0.3,  0,0,1.4,0.5,0.1,0.2,0.2,0.3,0.9,0.1,0.4,0.3)
std.prev.1 = matrix(std.prev.1,12,4)
std.prev.2 = c(0,0,3.4,2.8,1.4,0.3,0.0,0.7,1.3,2.1,3.8,2.0,  0,0,3.5,1.0,3.6,0.4,0.0,0.0,2.0,2.3,3.4,3.3,  0,0,3.1,1.5,1.2,2.8,0.0,0.5,1.4,3.3,2.1,4.5,  0,0,4.2,0.4,0.9,2.3,0.0,0.6,1.2,2.5,3.5,5.5)
std.prev.2 = matrix(std.prev.2,12,4)


data = as.matrix(read.table('tmpres_from_python'))
rownames(data)=data[,1]
data=data[,-1]
class(data) = 'numeric'
names = c('ArD20','ArD30','emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50')
preset_names = c('textsc{ArD20}','textsc{ArD30}','textsc{Emotions}','textsc{Yeast}','textsc{Scene}','textsc{Enron}','textsc{Cal500}','textsc{Fingerprint}','textsc{NCI60}','textsc{Medical}','textsc{Circle10}','textsc{Circle50}')


data.res = c()
std.res = c()
n_col = dim(data)[2]
n=0
for( name in names )
{
  n=n+1
  data.name = data[rownames(data)==name,]
  data.name.allf = c()
  for(f in c(1,2,3,4,5))
  {
    data.name.f = data.name[data.name[,1]==f,]
    data.name.f = round(rbind(data.name.f,data.name.f),1)
    data.name.f = data.name.f[order(data.name.f[,n_col-1],data.name.f[,n_col]),]
    if(length(data.name.f)==0){next}
    data.name.allf = rbind(data.name.allf, data.name.f[1,])
  }
  rownames(data.name.allf) = rep(preset_names[n],dim(data.name.allf)[1])
  #print(data.name.allf)
  data.name.avg = 100-round(colMeans(data.name.allf),1)
  data.res = rbind(data.res, data.name.avg[c((n_col-1):n_col)])
  #std
  data.name.avg = round(apply(data.name.allf,2,sd),1)
  std.res = rbind(std.res, data.name.avg[c((n_col-1):n_col)])
}

rownames(data.res) = preset_names
data.res = cbind(res.prev.1,data.res[,2],res.prev.2,data.res[,1])
colnames(data.res) = c('SVM','MTL','MMCRF','MAM','RSTA','SVM','MTL','MMCRF','MAM','RSTA')
data.res = 100-data.res
print(data.res)
data.res = xtable(data.res,digit=1)
print(data.res, sanitize.text.function=I) 

rownames(std.res) = preset_names
std.res = cbind(std.prev.1,std.res[,2],std.prev.2,std.res[,1])
colnames(std.res) = c('SVM','MTL','MMCRF','MAM','RSTA','SVM','MTL','MMCRF','MAM','RSTA')
std.res = std.res
print(std.res)
std.res = xtable(std.res,digit=1)
print(std.res, sanitize.text.function=I) 

