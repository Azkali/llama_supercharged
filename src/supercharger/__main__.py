from supercharger.system.MEMCON.master import MEMCON
from argparse import ArgumentParser
import json

valid_types = [
    "MEMCON",
    "L0JSON",
]

def parser():
    parser = ArgumentParser()
    #parser.add_argument("-m", "--model", type=str, help="Model name")
    #parser.add_argument("-y", "--yaml_file", type=str, help="YAML file")
    #parser.add_argument("-j", "--json_file", type=str, nargs="+", help="JSON file")
    return parser.parse_args()

def main(model: str, data: []):
    instructions = []
    for i in data:
        with open(i) as f:
            instruction = json.load(f)
            if instruction.type not in valid_types:
                print("file \"{i}\" has invalid type.")
                exit(4321)
            instructions.append(json.load(f))




    if yaml_file:
        print(f"Loading YAML file... {yaml_file}")
        exit()
        #multi_model(yaml_file)
    elif json_file and model:
        print(f"Loading JSON file... {json_file}")
        exit()
        #single_model(model, json_file, messages)
    else:
        exit("Please provide either a JSON file AND a valid model name or a YAML file")

def run():
    args = parser()
    main(args.model, args.json_file, args.yaml_file)

if __name__ == "__main__":
    run()
