# CS470
CS470 final project for Team 32
Implementation of VQA and TextVQA for blind people
Details of each folder will be inside README of each folder

## API folder 
- Codes for flask api
- Infer VQA model

## web folder
- Codes for chrome extension api
- Not yet done with inferring model due to problem with onnx. Details will be in report.

## convert_pt2onnx.py
- Code to convert pytorch checkpoint to onnx
- Works well converting one model. Details with onnx will be in report.

## RunModel folder(failed)
- RunModel_modified_v00.py: Code to implement TextVQA demo (model inference)
- optimized_RunModel.py, optimized_RunModel_ver2.py: Code to implement optimization using TensorRT
- textvqa_dataset.py: modified textvqa_dataset.py to implement RunModel_modified_v00.py

## ajax-flask(failed)
- Codes to connect Javascript file and python file using ajax and flask
