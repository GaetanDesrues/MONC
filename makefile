EPOCHS = 10
MINIBATCH = 1
CROP_SIZE = 256
LEN_TRAIN = 0.8


train : train.py
	#  --epochs=10 --minib=1 --crop=100 --lentrain=0.1
	python3 -W ignore train.py -E $(EPOCHS) -m $(MINIBATCH) -c $(CROP_SIZE) -t $(LEN_TRAIN)

main : main.py
	python3 -W ignore main.py -E $(EPOCHS) -m $(MINIBATCH) -c $(CROP_SIZE) -t $(LEN_TRAIN)

tb :
	tensorboard --logdir output/runs

plaf :
	srun -p sirocco python3.6 -W ignore main.py -E $(EPOCHS) -m $(MINIBATCH) -c $(CROP_SIZE) -t $(LEN_TRAIN)

# mod :
# 	module load slurm
# 	module load language/python/3.6
