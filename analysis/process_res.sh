

#for name in 'cal500' 'toy10' 'emotions' 'scene' 'toy50' 'enron' 'medical' 'yeast' 'ArD10' 'ArD15' 'ArD20' 'ArD30' 'cancer'
for name in 'cal500' 'toy10' 'emotions' 'scene' 'enron' 'yeast' 'ArD10' 'ArD15' 'ArD20' 'ArD30' 'medical' 'toy50'
do
for size in '1' '5' '10' '15' '20' '25' '30' '35' '40' 
do
for kfold in '1'
do
for lnorm in '2'
do
for kappa in '2' '4' '8' '16' '20' '24' '32' '40' '48' '52' '60'
do
	cat ../outputs/phase4/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_RSTAr.log |grep '%'| sed -e's/%//g'|cut -d' ' -f 8,11,16,21,29,30,35,37,38|sed -e'/Inf/d' -e'/ts/d' -e's/(//g' -e's/)//g' >../processed_outputs/phase4/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_RSTAr.plog
	cat ../outputs/phase4/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_RSTAr.log |grep -v '%'|cut -d' ' -f 6,9,12,15 |sed -e'/ts/d' -e's/(//g' -e's/)//g' >../processed_outputs/phase4/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_RSTAr.tstr
done
done
done
done
done
