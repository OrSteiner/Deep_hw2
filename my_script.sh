
#exp 1.1
#for i in 32 64
#do
#	for j in 2 4 8 16
#	do
#		let x=${j}/2
#		srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_K${i}_L${j} --seed 42 --bs-train 128 --batches 100 --epochs 10 --early-stopping 3 --filters-per-layer ${i} --layers-per-block ${j} --pool-every ${x} --hidden-dims 100
#	done
#done


#exp 1.2
#for j in 2 4 8
#do
#	for i in 32 64 128 258
#	do
#		let x=${j}/2
#		srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2_L${j}_K${i} --seed 42 --bs-train 128 --batches 100 --epochs 10 --early-stopping 3 --filters-per-layer ${i} --layers-per-block ${j} --pool-every ${x} --hidden-dims 100
#	done
#done


#exp 1.3
#for j in 1 2 3 4
#do
#		let x=${j}
#		srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3_L${j}_K64-128-256 --seed 42 --bs-train 128 --batches 100 --epochs 10 --early-stopping 3 --filters-per-layer 64 128 256 --layers-per-block ${j} --pool-every ${x} --hidden-dims 100 
#done


#exp 2
#for j in 1 2 3 4
#do
	#	let x=2
	#	if ((j > 2))
	#	then
	#		let x=4
	#	fi
	#	srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp2_L${j}_K64-128-256-512 --seed 42 --bs-train 128 --batches 10000 --epochs 10 --early-stopping 3 --filters-per-layer 64 128 256 512 --layers-per-block ${j} --pool-every ${x} --hidden-dims 100 --ycn
#done

