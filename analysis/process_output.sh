

# bash script for postprocessing which analyzes *.log file and extracts numeric properties
# assume *.log locate in ../output/ folder, generated files will locate in ../processed_output/ folder
# usage example:
#	bash process_output.sh pf_phase1 pf_phase1 pfRSTAs


srcfolder=$1
desfolder=$2
suffix=$3


rm -rf ../processed_outputs/${desfolder}
mkdir ../processed_outputs/${desfolder}


for name in 'cal500' 'toy10' 'toy50' 'emotions' 'scene' 'enron' 'yeast' 'ArD20' 'ArD30' 'medical' 'cancer' 'fp'
do
for size in '1' '5' '10' '15' '20' '25' '30' '35' '40' 
do
for kfold in '1' '2' '3' '4' '5'
do
for lnorm in '2'
do
for kappa in '2' '4' '8' '16' '20' '24' '32' '40'
do
	if [ -f ../outputs/${srcfolder}/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_${suffix}.log ];
		then
		cat ../outputs/${srcfolder}/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_${suffix}.log |grep '%'| sed -e's/%//g'|cut -d' ' -f 8,11,15,16,17,18,20,21,22,23,29,30,31,33,35,40|sed -e'/Inf/d' -e'/ts/d' -e's/(//g' -e's/)//g' >../processed_outputs/${desfolder}/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_${suffix}.plog
		cat ../outputs/${srcfolder}/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_${suffix}.log |grep -v '%'|cut -d' ' -f 6,9,12,15 |sed -e'/ts/d' -e's/(//g' -e's/)//g' >../processed_outputs/${srcfolder}/${name}_tree_${size}_f${kfold}_l${lnorm}_k${kappa}_${suffix}.tstr
		else
			continue
		fi
done
done
done
done
done
