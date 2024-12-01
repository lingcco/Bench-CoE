import os
from typing import Dict

from .base import *
from ...utils import import_modules

def register_template(name):
    def register_template_cls(cls):
        if name in TEMPlATE_FACTORY:
            return TEMPlATE_FACTORY[name]

        TEMPlATE_FACTORY[name] = cls
        return cls

    return register_template_cls

TEMPlATE_FACTORY: Dict[str, Template] = {}

def TemplateFactory(version):
    template = TEMPlATE_FACTORY.get(version, None)
    #print("TEMPlATE_FACTORY:",TEMPlATE_FACTORY)
    assert template, f"{version} is not implmentation"
    return template


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.data.template")
