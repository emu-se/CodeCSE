# CodeCSE
A simple pre-trained model for code and comment embeddings using contrastive learning. The pretrained model is hosted at https://huggingface.co/sjiang1/codecse Please check the inference script for how to download/use it.

## Environment
This model was trained and tested in Python 3.9. The dependencies are in the requirements.txt. This repository uses CodeBERT/GraphCodeBERT for data preparation. To initialize the submodule:
```sh
git submodule init
git submodule update
```

## Inference
Run the example script for inference:
```sh
GCB_PATH=./CodeBERT/GraphCodeBERT/codesearch \
PYTHONPATH=./CodeBERT/GraphCodeBERT/codesearch:./codecse:$PYTHONPATH \
python inference.py
```
_Note_: GraphCodeBERT is put at the beginning of `PATH` because Python has an internal 'parser' module, which conflicts with the package 'parser' in GraphCodeBERT/codesearch.

## Troubleshooting
### Error: 'parser/my-languages.so' (not a mach-o file)
The error message below means that the built file 'my-language.so' doesn't work on your machine.
```
OSError: dlopen(parser/my-languages.so, 0x0006): 
tried: 'parser/my-languages.so' (not a mach-o file), 
'/path/to/CodeCSE/CodeBERT/GraphCodeBERT/codesearch/parser/my-languages.so' (not a mach-o file)
```
To rebuild 'my-language.so', please follow the instructions in [GraphCodeBERT/codesearch#tree-sitter-optional](https://github.com/emu-se/CodeBERT/tree/91f1552235c4bfdbeb0a7d6dfe003233387a7db6/GraphCodeBERT/codesearch#tree-sitter-optional).

### Error: 'ValueError: Incompatible Language version XX. Must be between YY and ZZ'
The error message means that the tree-sitter package pip installed is not compatible with the built 'parser/my-languages.so'. Upgrade the tree-sitter will solve this problem.
```sh
python -m pip install tree-sitter --upgrade
```
