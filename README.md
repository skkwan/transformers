# README for transformers 

## Installation (do only once)

```
conda create -n "env-transformers" python=3.10
conda activate env-transformers

# pip install all the reqiurements in the hls4ml repo setup.cfg
pip install calmjs.parse
pip install h5py
pip install numpy
pip install onnx>=1.4.0
pip install pydigitalwavetools==1.1
pip install pyyaml
pip install qkeras
pip install six
pip install tabulate

# And a few more that are missing from setup.cfg as of June 9, 2023 (hls4ml commit 0599cca)
pip install tensorflow
pip install pyparsing

# For our usage
pip install jupyter

# And a few more packages that are not necessary to import hls4ml, but needed for pydotprint
pip install pydot
pip install graphviz
```

Slightly hacky way to get `hls4ml` code:
```
$ git clone https://github.com/skkwan/hls4ml.git
$ mv -r hls4ml/ hls4ml-repo/
$ cp hls4ml-repo/ hls4ml/ 
```



## Setup (start of each session)
```
$ conda activate env-transformers
```

In a different terminal on my own computer, set up tunneling
```
ssh -L 3996:localhost:3996 lxplus.cern.ch
```

Back in the working area:
```
$ jupyter notebook --port=3996 --no-browser
```