from .base import (register_pruner,
                   PytorchBasePruner)
from ..schedulers import get_scheduler
from ..patterns import get_pattern
from ..criteria import get_criterion
from ..regs import get_reg
import transformers
from ..utils import logger, torch, nn
import gc
import math
@register_pruner('pt_sparse_gpt')
class SparseGPTPruner(PytorchBasePruner):
    """Pruning Pruner.
    The sparse_gpt pruner_class is derived from PytorchBasePruner.
    
    ###     Needs refinement   ###
        

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Attributes:
        pattern: A Pattern object that defines pruning weights' arrangements within space.
        criterion: A Criterion Object that defines which weights are to be pruned
        scheduler: A Scheduler object that defines how the model's sparsity changes as training/pruning proceeds.
        reg: A Reg object that defines regulization terms.
    """
    
    def __init__(self, config, modules, framework='pytorch'):
        """Initialize."""
        super(SparseGPTPruner, self).__init__(config, modules)
        
    def _init(self):
        """Initialize."""
        self.pattern = get_pattern(self.config, self.modules)
        
        # self.criterion = get_criterion(config=self.config, modules=self.modules)
        gc.collect()
        self.gpts = {}
        logger.warning("sparse_gpt pruner fixed the weights, Please DO NOT train or update gradients.")
        assert "1x1" in self.pattern.pattern or ":" in self.pattern.pattern, \
                "sparse_gpt pruner type only supports 1x1 and N:M patterns."
    
    class SparseGPT():
        def __init__(self, module):
            self.module = module
            self.dev = self.module.weight.device
            W = module.weight.data.clone()
            if isinstance(self.module, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(self.module, transformers.Conv1D):
                W = W.t()
            self.rows = W.shape[0]
            self.columns = W.shape[1]
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
            self.nsamples = 0

        def add_batch(self, inp, blocksize=1024):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            sample_num = inp.shape[0] # batchsize
            if isinstance(self.module , nn.Linear) or isinstance(self.module, transformers.Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()
            self.H *= self.nsamples / (self.nsamples + sample_num)
            self.nsamples += sample_num
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    def register_gpt_hook(self, op_names):
        self.gpts = {}
        def add_batch(gpt):
            def tmp(_, inp):
                gpt.add_batch(inp[0].data) # get layer-wise matrix, H = (XX> + λI)
            return tmp
        handles = []
        for name in op_names:
            module = self.modules[name]
            self.gpts[name] = self.SparseGPT(module)
            handles.append(module.register_forward_pre_hook(add_batch(self.gpts[name])))
        return handles
        
    @torch.no_grad()
    def fasterprune(self, op_names):
        for name in op_names:
            logger.info(f"module: {name}\t target ratio: {self.target_sparsity_ratio}")
            module = self.modules[name]
            self.pattern.fasterprune(self.gpts[name]) # is there necessary to add a hyperparameter of blocksize
