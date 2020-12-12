## Files
- RunModel_modified_v00.py: Code to implement TextVQA demo (model inference)
- optimized_RunModel.py, optimized_RunModel_ver2.py: Code to implement optimization using TensorRT
- textvqa_dataset.py: modified textvqa_dataset.py to implement RunModel_modified_v00.py

## Explanation
- RunModel_modified_v00.py try to imitate the form of dataset that is used in Spatially Aware Multimodal Transformer.

## Limitation
- A problem occurred in the Dataloder section, so ㅈㄷ fixed both the functions in the original code and textvqa_dataset.py, but there was an error.
- It was found that for the reference, the dataset must be completely imitated. However, to do this, we need to create an OCR token, which requires a paid Google API.

## Further Work
- It is expected that a complete imitation of a dataset will enable a demonstration to be executed through TextVQA.
- Once the demonstration is complete, attempts to optimize the Spatially Aware Multimodal Transformer are also possible.
