to_export = []

def register(mod):
    """decorate a torch.nn.Module to export a torchscript file"""
    to_export.append(mod)

def foo(_, i):
    print(i)