# README for transformers 

## Installation (do only once)

Package requirements: keras, tensorflow, jupyter

```
$ git clone git@github.com:skkwan/hls4ml.git
$ python3 -m venv env-transformer
$ source env-transformer/bin/activate
(env-transformer) $ pip install --upgrade pip
(env-transformer) $ pip install -r requirements.txt 
```


## Setup (start of each session)
```
$ source env-transformers/bin/activate
```

In a different terminal on my own computer, set up tunneling
```
ssh -L 3996:localhost:3996 lxplus.cern.ch
```

Back in the working area:
```
$ jupyter notebook --port=3996 --no-browser
```