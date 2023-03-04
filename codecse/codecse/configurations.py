from transformers.models.roberta.configuration_roberta import RobertaConfig

class CLRobertaConfig(RobertaConfig):
    model_type = "roberta_for_cl"

    def __init__(self, 
        pad_token_id=1, 
        bos_token_id=0, 
        eos_token_id=2, 
        cl_pooler_type='cls', 
        cl_temp=0.05, 
        cl_mlp_layer=1, 
        cl_loss='asymmetric', 
        gc_code_length=-1,
        gc_data_flow_length=-1,
        gc_nl_length=-1,
        **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id, bos_token_id, eos_token_id, **kwargs)
        # for contrastive learning
        self.cl_pooler_type = cl_pooler_type
        self.cl_temp = cl_temp
        self.cl_mlp_layer = cl_mlp_layer
        self.cl_loss = cl_loss
        # for GraphCodeBERTForCL
        self.gc_code_length = gc_code_length
        self.gc_data_flow_length = gc_data_flow_length
        self.gc_nl_length = gc_nl_length
