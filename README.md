# Sampling for fewshot

Sample efficiently in meta-training (not random-sampling, but sampling-method).
'''
file: `few_shot/core.py`
'''

This project is written in python 3.6 and Pytorch and assumes you have
a GPU.


# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

### Data
Edit the `DATA_PATH` variable in `config.py` to the location where
you store the miniImagenet datasets.

After acquiring the
data and running the setup scripts your folder structure should look
like
```
DATA_PATH/
    miniImageNet/
        images_background/
        images_evaluation/
```


**miniImageNet** dataset. Download files from
https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view,
place in `data/miniImageNet/images` and run `scripts/prepare_mini_imagenet.py`
I'm using the miniImagenet splits defined here: https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/

### Results

The file `experiments/matching_nets.py` contains the hyperparameters I
used to obtain the results given below.

### Inference

Run `inference/matching_infer.py` in the root directory to run
all tests.



# Reference
https://github.com/oscarknagg/few-shot

https://github.com/yaoyao-liu/mini-imagenet-tools