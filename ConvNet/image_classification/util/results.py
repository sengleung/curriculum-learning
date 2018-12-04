import json

def to_json_file(filename, results):
    with open(filename + '.json', 'w') as fp:
        json.dump(results, fp, indent=4)
