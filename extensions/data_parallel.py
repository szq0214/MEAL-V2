__author__ = "Hessam Bagherinezhad <hessam@xnor.ai>"

from torch import nn
from torch.nn.modules import loss


class DataParallel(nn.DataParallel):
    """An extension of nn.DataParallel.

    The only extensions are:
        1) If an attribute is missing in an object of this class, it will look
            for it in the wrapped module. This is useful for getting `LR_REGIME`
            of the wrapped module for example.
        2) state_dict() of this class calls the wrapped module's state_dict(),
            hence the weights can be transferred from a data parallel wrapped
            module to a single gpu module.
    """


    def __getattr__(self, name):
        # If attribute doesn't exist in the DataParallel object this method will
        # be called. Here we first ask the super class to get the attribute, if
        # couldn't find it, we ask the underlying module that is wrapped by this
        # DataParallel to get the attribute.
        try:
            return super().__getattr__(name)
        except AttributeError:
            underlying_module = super().__getattr__('module')
            return getattr(underlying_module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
