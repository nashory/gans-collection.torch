# gans-collection.torch
Collection of GAN models implemented in Torch7 (e.g. DCGAN, ALI, Context-encoder).


## Contents
+ [DCGAN](https://arxiv.org/abs/1511.06434)
+ [Context-encoder](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Pathak_Context_Encoders_Feature_CVPR_2016_paper.html)
+ [ALI](https://arxiv.org/abs/1606.00704)


## Prerequisites
+ Torch7
+ python2.7
+ cuda

## Usage
1. download training data:
~~~ 
python download.py --datasets <dataset>
(e.g) python run.py --datasets celebA
~~~
2. run training:
~~~ 
python run.py --type <gan_type>
(e.g) python run.py --type dcgan
~~~
3. run server (real-time visualization)
~~~
python server.py --type <gan_type>
(e.g) python server.py --type dcgan
~~~

## Results
will be updated soon.




## Author
MinchulShin, [@nashory](https://github.com/nashory)



