### Setting

```
cd CS470
mkdir model_data
wget -O ~/CS470/model_data/answers_vqa.txt https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
wget -O ~/CS470/model_data/vocabulary_100k.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_100k.txt
wget -O ~/CS470/model_data/detectron_model.pth https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
wget -O ~/CS470/model_data/pythia.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth
wget -O ~/CS470/model_data/pythia.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.yml
wget -O ~/CS470/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget -O ~/CS470/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf ~/CS470/model_data/detectron_weights.tar.gz

pip install yacs cython matplotlib
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

git clone https://github.com/facebookresearch/mmf.git mmf
cd mmf
sed -i '/torch/d' requirements.txt
pip install -e .

cd
cd CS470
git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build
python setup.py develop
```

## Files
- screenshot.html: Basic html files for creating Web page
- script.js: JavaScript code to send data to the local server, receive results, and post them on the web page
- test.py: Python code for checking the operation of ajax and flask
- vqa.py: Python code to create a local server through flask. If it can be connected to script.js, it can receive data from script.js, make an reference through the Spatially Aware Multimodal Transformer, and send the results back to script.js.

## Explanation
- We tried to find a way to connect the JavaScript file and Python file, but it didn't work properly in the code we created.
