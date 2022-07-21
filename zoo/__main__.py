import fire

from .models import *
from .util import to_export, _export

def main():
    print(to_export)  
    for mod in to_export:
        _export(mod)

if __name__=='__main__':
    fire.Fire(main)