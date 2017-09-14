
## Training guide for Cycle-GAN

__
Note:
+ Implemented simple Alexnet structure only. you can try other archhitecture (e.g. 9-blocks resnet(G)).
+ DiscoGAN and CycleGAN are very similar. minor changes in loss function was implemented (Abs/MSE).
+ CycleGAN has only single cycle-consistency loss.
__


+ Download the image before you start training.
`bash ./datasets/download_dataset.sh <dataset_name>`

+ Change the data_train_root option in script/opts.lua


