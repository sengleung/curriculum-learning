import json

def get(filepath, name):
    full = filepath + '/' + name + '.json'
    with open(full, 'r') as fp:
        results = json.load(fp)
    return results

def save(filepath, name, results):
    full = filepath + '/' + name + '.json'
    with open(full, 'w') as fp:
        json.dump(results, fp, indent=4)
