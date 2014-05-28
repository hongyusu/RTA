


library(gplots)



names = c('ArD20','ArD30'.'cal500','scene','toy10','toy50','emotions','yeast','medical','enron','cancer')
names = c('ArD20','ArD30')

for(name in names)
{
	trange = c('1','10','20','30','40')
}



for(name in names)
{

    krange = c('2','4','8','16','20')
    trange=c('1','5','10','15','20','25','30','35','40')
    if(name=='cancer')
    {krange = c('2','4','8','16','20','24','32','40','48','52','60')}

    pdf(sprintf('../results/phase4/%s.pdf',name),height=12,width=33)
    par(mfrow=c(5,9),mar=c(4,4,2,0),oma=c(0.5,0.5,0.5,0.5))
    overall=c()
    for(k in krange)
    {
        data_plot = c()
        posnames=c('Training 0/1 loss','Training microlabel loss','Median position of Y* over examples','Median position of Yi over examples','Per of examples with F(Yi)-F(Y*)>0','F(Yi)-F(Y*)','Relative duality gap','percentage of tree agreed over examples','percentage of example with update')
        for(pos in c(1,2,3,4,5,6,7,8,9))
        {
            pdata = c()
            for(t in trange)
            {
                cat(name, k,t,'\n')
                data = as.matrix(read.table(sprintf('../processed_outputs/phase4/%s_tree_%s_f1_l2_k%s_RSTAr.plog',name,t,k)))
                pdata = cbind(pdata,data[,pos])
                if(pos==1)
                {
                    data1 = as.matrix(read.table(sprintf('../processed_outputs/phase4/%s_tree_%s_f1_l2_k%s_RSTAr.tstr',name,t,k)))
                    #overall = rbind(overall,c(colMeans(data[c(46:50),c(1,2,5,7)]),data1[3:4]))
                    overall = rbind(overall,c(apply(data[c(5:6),c(1,2,5,7)],2,min),data1[3:4]))
                    #overall = rbind(overall,c(data[c(12),c(1,2,5,7)],data1[3:4]))

                }
            }
            for(i in c(1:length(trange)))
            {
                if(i==1)
                {
                    plot(pdata[,i],ylim = c(min(pdata),max(pdata)),type='l',col=i,xlab='Iteration',lwd=1.5,lty=i,main=sprintf("%s, K=%s",posnames[pos],k))
                    legend('topright',trange,col=1:9,lty=1:9,title='T')
                }
                else
                {
                    lines(pdata[,i],col=i,lty=i,lwd=1.5)
                }
            }
        }
    }
    par(mgp=c(2,1,0),mar=c(4,4,2,0),oma=c(1,1,1,1))
    layout(matrix(c(1,1,2,2,3,3,4,4,5,5,6,6,8,7,8,8),2,8,byrow=TRUE))
    errtypes=c('Training accuracy 0/1','Training accuracy microlabel','Per of examples with positive margin','Relative duality gap','Test accuracy 0/1','Test accuracy microlabel')
    for(i in c(1:6))
    {
        data =matrix(overall[,i],length(trange),length(krange))
        if(sum(i == c(1,2,4,5,6))){data = 100-round(data,1)}
        #print(data)
        image(data,col=colorpanel(5,'yellow','red'),xlab='T',ylab='K',xaxt='n',yaxt='n',main=errtypes[i])
        x1=1.2
        x2=0.25
        if(name=='cancer')
        {x1=1.2;x2=0.12}
        axis(1,seq(0,1.05,0.13),trange)
        axis(2,seq(0,x1,x2),krange)
    }
    image(matrix(c(1:10),1,10),col=colorpanel(10,'yellow','red'),axes=FALSE,ylab='Performance',mar=c(4,4,4,4),cex.lab=1.4)
    axis(2,axTicks(2),c('Low','','','','','High'),cex.axis=1.4)
    par(mfrow=c(5,9),mar=c(4,4,2,0),oma=c(0.5,0.5,0.5,0.5))
    #trange=c('1','5','10','15','20','25','30','35','40')
    for(t in trange)
    {
        data_plot = c()
        posnames=c('Training 0/1 loss','Training microlabel loss','Median position of Y* over examples','Median position of Yi over examples','Per of examples with F(Yi)-F(Y*)>0','F(Yi)-F(Y*)','Relative duality gap','percentage of tree agreed over examples','percentage of example with update')
        for(pos in c(1,2,3,4,5,6,7,8,9))
        {
            pdata = c()
            #krange=c('2','4','8','16','20')
            for(k in krange)
            {
                #if(t=='40' && k=='32' && name=='enron'){k='16'}
                data = as.matrix(read.table(sprintf('../processed_outputs/phase4/%s_tree_%s_f1_l2_k%s_RSTAr.plog',name,t,k)))
                pdata = cbind(pdata,data[,pos])
            }
            for(i in c(1:length(krange)))
            {
                if(i==1)
                {
                    plot(pdata[,i],ylim = c(min(pdata),max(pdata)),type='l',col=i,xlab='Iteration',lwd=1.5,lty=i,main=sprintf("%s, T=%s",posnames[pos],t))
                    legend('topright',krange,col=1:5,lty=1:5,title='K')
                }
                else
                {
                    lines(pdata[,i],col=i,lty=i,lwd=1.5)
                }
            }
        }
    }
    dev.off()

}
