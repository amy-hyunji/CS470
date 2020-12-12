# README for Web folder 

### Setting
- change to `developer mode` in plugin install of chrome extension
- add this folder
- new chrome extension with name `Ask Your Question` will be shown. (icon of Magnifying Glass with yellow background)

### How to use
- if you click the extension, 
    - open image in new tab by `새 탭에서 이미지 열기`
    - click the icon
    - this will move to our page, `Ask Question`
    - you can ask any question related to the image
    - ** still in progress due to problem of input format on pretrained pythia

### Further work
- pretrain Pythia with simple input format (currently it makes a new object for input)
- convert Detectron, ResNet and Pythia model to onnx (have proved that it works)
- This will allow inferring the model by our chrome extension