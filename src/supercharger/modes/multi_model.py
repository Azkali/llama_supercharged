from supercharger.multimodel import MultiModel
from argparse import ArgumentParser

def multi_model(yaml_file: str):
    multimodel = MultiModel(yaml_file=yaml_file)
    multimodel()
