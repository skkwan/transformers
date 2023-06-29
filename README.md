# README for transformers 

## Installation (do only once)

```bash
% conda create -n "env-transformers" python=3.10
% conda activate env-transformers
% # pip install all the reqiurements in the hls4ml repo setup.cfg
% (env-transformers) pip install calmjs.parse
% (env-transformers) pip install h5py
% (env-transformers) pip install numpy
% (env-transformers) pip install onnx==1.4.0
% (env-transformers) pip install pydigitalwavetools==1.1
% (env-transformers) pip install pyyaml
% (env-transformers) pip install qkeras
% (env-transformers) pip install six
% (env-transformers) pip install tabulate
% 
% # And a few more that are missing from setup.cfg as of June 9, 2023 (hls4ml commit 0599cca)
% (env-transformers) pip install tensorflow
% (env-transformers) pip install pyparsing
% 
% # For our usage
% (env-transformers) pip install jupyter
% 
% # And a few more packages that are not necessary to import hls4ml, but needed for pydotprint
% (env-transformers) pip install pydot
% (env-transformers) pip install graphviz
```

To get the `hls4ml` code:
```bash
$ git clone https://github.com/skkwan/hls4ml.git
$ mv hls4ml/ hls4ml-repo/
$ cp -r hls4ml-repo/hls4ml/ hls4ml/ 
```


## To open a Jupyter notebook on lxplus (start of each session)
```bash
$ conda activate env-transformers
```

In a different terminal on my own computer, set up tunneling
```bash
$ ssh -L 3996:localhost:3996 lxplus.cern.ch
```

Back in the working area:
```bash
$ jupyter notebook --port=3996 --no-browser
```

## To run Vivado HLS on cmstrigger02

I use the FPGA TAC-HEP module instructions (Lecture 3) https://drive.google.com/drive/folders/1dp8xqug21e1L4AdKKst8YV99aq6GmSAw to connect to `cmstrigger02` and set up a VNC Server. 
On my computer it's set up as:
```bash
$ ssh cmstrigger02-via-login
$ vncserver -localhost -geometry 1024x768 
```

On my own computer again:
```bash
# Replace 5901 with 5904, say, if the display number was 4 in the previous step
ssh skkwan@cmstrigger02-via-login -L5901:localhost:5901
```

In the VNC server,
```bash
$ source /opt/Xilinx/Vivado/2020.1/settings64.sh
$ vivado_hls
```

## Workflow

1. First use the `example-one-layer-GELU.py` in the top-level folder to generate a Vivado HLS project (e.g. called `dummy`). Since we call `hls::erf` or other functions supplied by Vivado HLS,
   we can't actually count on `hls4ml` compile or evaluate to work with the HLS version.
2. This will create a folder `dummy` which contains all the files that we need to make a Vivado HLS project.
3. What files to modify? 
    * The top-level function is in `dummy/firmware/myproject.cpp`.
    * From here on out, to change/develop the GELU activation function, you can just change it in `dummy/firmware/nnet_utils/nnet_activation.h`.
       -  The "proper" thing to do would be to modify the GELU activation function in the `hls4ml` code: `hls4ml/templates/vivado/nnet_utils/nnet_activation.h`, and then do 
        step 1 again, but all it does is dump the GELU definition in that header file into a 
        Vivado HLS project anyway, so I'm just changing the `.h` in the `dummy/` project folder.
    * The size of the inputs is set in `dummy/firmware/parameters.h`.
    * The test bench is in `dummy/firmware/myproject_tb.cpp`.
4. So far I have relied on the GUI to set things up:
    * Project location: `my-test-project/`
    * Simulation: 
        - TestBench Files: 
            - `myproject_tb.cpp`, with `CFLAGS` `-std=c++0x`
            - The folders `ap_types/`, `nnet_utils/`, `weights/`
    * Synthesis:
        - Top Function: `myproject`
        - Synthesis C/C++ Source Files:
            - `defines.h` with CFLAGS `-std=c++0x`
            - `myproject.cpp` with CFLAGS `-std=c++0x`
            - `myproject.h` with CFLAGS `--std=c++0x`
            - `parameters.h` with CFLAGS `--std=c++0x`