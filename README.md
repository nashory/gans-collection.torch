# gans-collection.torch
Torch implementation of various types of GANs (e.g. DCGAN, ALI, Context-encoder, DiscoGAN, CycleGAN).

![image](https://camo.githubusercontent.com/45e147fc9dfcf6a8e5df2c9b985078258b9974e3/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a33394e6e6e695f6e685044614c7539416e544c6f57772e706e67)


## Contents
+ [DCGAN](https://arxiv.org/abs/1511.06434)
+ [Context-encoder](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Pathak_Context_Encoders_Feature_CVPR_2016_paper.html)
+ [ALI](https://arxiv.org/abs/1606.00704)
+ [DiscoGAN](https://arxiv.org/pdf/1703.05192.pdf)
+ [CycleGAN](https://arxiv.org/abs/1703.10593)


## Prerequisites
+ Torch7
+ python2.7
+ cuda
+ other torch packages (display, hdf5, image ...)

## Usage
1. download training data:
~~~ 
python download.py --datasets <dataset>
(e.g) python run.py --datasets celebA

---------------------------------------
The training data folder should look like : 
<train_data_root>
                |--classA
                        |--image1A
                        |--image2B ...
                |--classB
                        |--image1B
                        |--image2B ...
---------------------------------------
~~~

2. run GANs training:
__Note that you need to change parameter options in "script/opts.lua" for each GANs.__
~~~ 
python run.py --type <gan_type>
(e.g) python run.py --type dcgan
~~~

## Display GUI : How to see generated images in real-time?
step by step instruction:
~~~
1. set server-related options(ip, port, etc.) in "script.opts.lua"
2. run server (python server.py --type <gan_type>)
3. open web browser, and connect. (https://<server_ip>:<server_port>)
~~~

you will see like this:
![image](https://puu.sh/xyy5y/a12f6e9aa0.png)




## Results
will be updated soon.


## Acknowledgement
+ brought dataloader code from ([DCGAN](https://github.com/soumith/dcgan.torch))  
+ referenced the code from ([Context-encoder](https://github.com/pathak22/context-encoder))  



## Author
MinchulShin, [@nashory](https://github.com/nashory)  
__Will keep updating other types of GANs.__  
__Any insane bug reports or questions are welcome. (min.stellastra[at]gmail.com)  :-)__



