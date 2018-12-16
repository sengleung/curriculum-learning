import os
import json
import datetime

directory = os.fsencode("./data")

data = {}

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    with open('./data/' + filename, 'r') as fp:
        model_results = json.load(fp)
        model_name = model_results['name']
        data[model_name] = model_results

#Just to prevent accidentally overwriting all data we append datetime
with open('all_' + str(datetime.datetime.now()) +'.json', 'w') as fp:
    json.dump(data, fp, indent=4)
