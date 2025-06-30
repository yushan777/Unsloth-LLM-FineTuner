## system-wide dependencies
```
sudo apt install cmake
sudo apt update
sudo apt install libcurl4-openssl-dev
```

##  create dir for unsloth
``` 
mkdir ~/ai/unsloth
cd ~/ai/unsloth
```

##  create venv and activate
```
python3 -m venv venv
source venv/bin/activate
```

##  install unsloth and tensorboard
```
pip install unsloth
pip install tensorboard
```

##  Build llama.cpp 
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release

cd ../..
```

##  create models dir
```
mkdir models
# download and place in unsloth/Meta-Llama-3.1-8B-Instruct
```
