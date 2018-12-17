import sys
distributions_1 = [
    #Tradeoff between previous and current
    #{ "Dp": 0.9 ,"Dc": 0.1, "Df": 0.0 },
    { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 },
    { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 },
    { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 },
    #{ "Dp": 0.1 ,"Dc": 0.9, "Df": 0.0 },

    #Tradeoff between forward and current
    #{ "Dp": 0.0, "Dc": 0.1, "Df": 0.9},
    { "Dp": 0.0, "Dc": 0.3, "Df": 0.7},
    { "Dp": 0.0, "Dc": 0.5, "Df": 0.5},
    { "Dp": 0.0, "Dc": 0.7, "Df": 0.3},
    #{ "Dp": 0.0, "Dc": 0.9, "Df": 0.1},

    #Pure current
    { "Dp": 0.0, "Dc": 1.0,"Df": 0.0},

    #Tradeoff between current and future + previous
    { "Dp":0.1, "Dc":0.8, "Df":0.1 },
    #{ "Dp":0.2, "Dc":0.6, "Df":0.2 },
    { "Dp":0.3, "Dc":0.4, "Df":0.3 },
    #{ "Dp":0.4, "Dc":0.2, "Df":0.4 },
    { "Dp":0.5, "Dc":0.0, "Df":0.5 },

    #Tradeoff between current and previous (with some fixed future)
    { "Dp":0.2, "Dc":0.7, "Df": 0.1},
    #{ "Dp":0.4, "Dc":0.5, "Df": 0.1},
    { "Dp":0.6, "Dc":0.3, "Df": 0.1},
    #{ "Dp":0.8, "Dc":0.1, "Df": 0.1},
    { "Dp":0.1, "Dc":0.6, "Df": 0.3},
    #{ "Dp":0.3, "Dc":0.4, "Df": 0.3},
    { "Dp":0.5, "Dc":0.2, "Df": 0.3},

    #Tradeoff between current and future (with some fixed previous)
    { "Dp": 0.1, "Dc":0.7, "Df": 0.2},
    #{ "Dp": 0.1, "Dc":0.5, "Df": 0.4},
    { "Dp": 0.1, "Dc":0.3, "Df": 0.6},
    #{ "Dp": 0.1, "Dc":0.1, "Df": 0.8},
    { "Dp": 0.3, "Dc":0.6, "Df": 0.1},
    #{ "Dp": 0.3, "Dc":0.4, "Df": 0.3},
    { "Dp": 0.3, "Dc":0.2, "Df": 0.5}
]

distributions_2 = [
    # #Tradeoff between previous and current
    { "Dp": 0.9 ,"Dc": 0.1, "Df": 0.0 },
    # { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 },
    # { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 },
    # { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 },
    { "Dp": 0.1 ,"Dc": 0.9, "Df": 0.0 },
    #
    # #Tradeoff between forward and current
    { "Dp": 0.0, "Dc": 0.1, "Df": 0.9},
    # { "Dp": 0.0, "Dc": 0.3, "Df": 0.7},
    # { "Dp": 0.0, "Dc": 0.5, "Df": 0.5},
    # { "Dp": 0.0, "Dc": 0.7, "Df": 0.3},
    { "Dp": 0.0, "Dc": 0.9, "Df": 0.1},
    #
    # #Pure current
    # { "Dp": 0.0, "Dc": 1.0,"Df": 0.0},
    #
    # #Tradeoff between current and future + previous
    # { "Dp":0.1, "Dc":0.8, "Df":0.1 },
    { "Dp":0.2, "Dc":0.6, "Df":0.2 },
    # { "Dp":0.3, "Dc":0.4, "Df":0.3 },
    { "Dp":0.4, "Dc":0.2, "Df":0.4 },
    # { "Dp":0.5, "Dc":0.0, "Df":0.5 },
    #
    # #Tradeoff between current and previous (with some fixed future)
    # { "Dp":0.2, "Dc":0.7, "Df": 0.1},
    { "Dp":0.4, "Dc":0.5, "Df": 0.1},
    # { "Dp":0.6, "Dc":0.3, "Df": 0.1},
    { "Dp":0.8, "Dc":0.1, "Df": 0.1},
    # { "Dp":0.1, "Dc":0.6, "Df": 0.3},
    { "Dp":0.3, "Dc":0.4, "Df": 0.3},
    # { "Dp":0.5, "Dc":0.2, "Df": 0.3},
    #
    # #Tradeoff between current and future (with some fixed previous)
    # { "Dp": 0.1, "Dc":0.7, "Df": 0.2},
    { "Dp": 0.1, "Dc":0.5, "Df": 0.4},
    # { "Dp": 0.1, "Dc":0.3, "Df": 0.6},
    { "Dp": 0.1, "Dc":0.1, "Df": 0.8},
    # { "Dp": 0.3, "Dc":0.6, "Df": 0.1},
    { "Dp": 0.3, "Dc":0.4, "Df": 0.3},
    # { "Dp": 0.3, "Dc":0.2, "Df": 0.5}
]

id = sys.argv[1]
task_amount = sys.argv[2]
group = sys.argv[3]
print("Running model " + id + " with a split of " + task_amount + " tasks on distribution group " + group)

model_ids = []
task_counts = []
distributions = []
if int(group) is 1:
    distributions = distributions_1
elif int(group) is 2:
    distributions = distributions_2

model_ids.append(int(id))
task_counts.append(int(task_amount))

model_configs = []
for model_id in model_ids:
    for task_count in task_counts:
        for distribution in distributions:
            name = "id{0}_t{1}_dp{2}_dc{3}_df{4}".format(
                model_id, task_count, distribution['Dp'], distribution['Dc'], distribution['Df']
            )
            model = {
                "name" : name,
                "id" : model_id,
                "task_count" : task_count,
                "distribution" : distribution
            }
            model_configs.append(model)

print("Model count = " + str(len(model_configs)))

import json
filepath = './models/'
filename = "configurations"

with open(filepath + filename + '.json', 'w') as fp:
    json.dump(model_configs, fp, indent=4)
