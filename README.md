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

## TextVQA folder
- we refered to this git (https://github.com/yashkant/sam-textvqa/tree/main/data)
- you can download best_model.tar about our experiments from this link (https://drive.google.com/drive/folders/1seYWDxdAlwcFCU1j7W1EvdG4PoRX9KJX?usp=sharing)

## convert_pt2onnx.py
- Code to convert pytorch checkpoint to onnx
- Works well converting one model. Details with onnx will be in report.

## RunModel folder(failed)
- Code to implement TextVQA demo
- Need Google paid API to make OCR tokens

## ajax-flask(failed)
- Codes to connect Javascript file and python file using ajax and flask
