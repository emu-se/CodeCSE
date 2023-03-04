# public-CodeCSE
The public repository for CodeCSE.

## Environment
This model was trained and tested in Python 3.9. The dependencies are in the requirements.txt. This repository uses CodeBERT/GraphCodeBERT for data preparation. To initialize the submodule:
```sh
git submodule init
git submodule update
```

## Inference
Run the example script for inference:
```sh
GCB_PATH=./CodeBERT/GraphCodeBERT/codesearch PYTHONPATH=./CodeBERT/GraphCodeBERT/codesearch:./codecse:$PYTHONPATH python inference.py
```
_Note_: GraphCodeBERT is put at the beginning of `PATH` because Python has an internal 'parser' module, which conflicts with the package 'parser' in GraphCodeBERT/codesearch.
