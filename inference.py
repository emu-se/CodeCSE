import os
import torch
import json
from dataclasses import dataclass
from transformers import AutoTokenizer
from codecse import GraphCodeBERTForCL

"""
import GraphCodeBERT/codesearch
change the working directory to load the parser/my-languages.so properly
"""
# get the graphcodebert's path
graphcodebert_path = os.getenv('GCB_PATH')
assert os.path.isdir(graphcodebert_path), f"Cannot find the folder for GraphCodeBERT: {graphcodebert_path}"

# save the current cwd
cwd=os.getcwd()
# change the cwd so that 'parser/my-languages.so' can be loaded correctly
os.chdir(graphcodebert_path) 
# load the modules from graphcodebert_path/run.py
from run import convert_examples_to_features, TextDataset
# change the cwd back
os.chdir(cwd)


class TextDatasetForInference(TextDataset):
    """ A simplified TextDataset just for simple inference

    This dataset has only one example.

    This class inherits from TextData from GraphCodeBERT 
    to get the method for preparing input features
    """
    def __init__(self, args, example):
        self.args = args
        self.examples = [example]
    

@dataclass
class Args:
    """ The arguments to be passed for data preparation
    """
    device: torch.device
    lang: str
    code_length: int 
    data_flow_length: int
    nl_length: int


def load_codecse_model(device):
    """ Load the CodeCSE model

    The model uses a customized configuration file, 
    so it is not fully supported by .from_pretrained method 
    """
    model = GraphCodeBERTForCL.from_pretrained("sjiang1/codecse")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("sjiang1/codecse")
    return model, tokenizer


def prepare_inputs(input_json, tokenizer, args):
    """ Load the example from the file

    1. Convert the example to features
    2. Use __getitem__ from TextDataset (GraphCodeBERT/codesearch/run.py)
       to prepare the tensors
    3. Pack the tensors into a batch
    4. Put the batch to args.device
    """
    input_features = convert_examples_to_features((input_json, tokenizer, args))
    dataset = TextDatasetForInference(args, input_features)
    tensors = dataset[0]
    # tensors: (code_ids, attn_mask, position_idx, nl_ids)
    batches = []
    for tensor in tensors:
        batch = tensor.unsqueeze(0)
        batch = batch.to(args.device)
        batches.append(batch)
    return batches


def load_example(file_path):
    ''' Load the json file into an object
    '''
    with open(file_path) as f:
        js = json.load(f)
        return js


if __name__ == "__main__":
    print("hello main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model, tokenizer = load_codecse_model(device)
    model.eval()

    args = Args(
        device=device,
        lang='python', 
        code_length=model.config.gc_code_length, 
        data_flow_length=model.config.gc_data_flow_length, 
        nl_length=model.config.gc_nl_length)

    # Get the embedding of an NL example
    nl_json = load_example("example_nl.json")
    batch = prepare_inputs(nl_json, tokenizer, args)
    nl_inputs = batch[3]
    with torch.no_grad():        
        nl_vec = model(input_ids=nl_inputs, sent_emb="nl")[1] 
    
    nl_emb = nl_vec.cpu().numpy()
    print(f"Embedding of the code example:\n{nl_emb.shape}\n{nl_emb}")


    # Get the embedding of a code example
    code_json = load_example("example_code.json")
    batch = prepare_inputs(code_json, tokenizer, args)
    code_inputs = batch[0]
    attn_mask = batch[1]
    position_idx =batch[2]
    # ----- Code from GraphCodeBERT/codesearch/model.py
    nodes_mask=position_idx.eq(0)
    token_mask=position_idx.ge(2)
    nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
    nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
    # ------
    with torch.no_grad():
        code_vec= model(input_ids=code_inputs,
                        attn_mask=attn_mask,
                        position_idx=position_idx, 
                        nodes_mask=nodes_mask, 
                        nodes_to_token_mask=nodes_to_token_mask, 
                        sent_emb="code")[1]
        
    code_emb = code_vec.cpu().numpy()
    print(f"Embedding of the code example:\n{code_emb.shape}\n{code_emb}")