distributions = [
    #Tradeoff between previous and current
    { "Dp": 0.9 ,"Dc": 0.1, "Df": 0.0 },
    { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 },
    { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 },
    { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 },
    { "Dp": 0.1 ,"Dc": 0.9, "Df": 0.0 },

    #Tradeoff between forward and current
    { "Dp": 0.0, "Dc": 0.1, "Df": 0.9},
    { "Dp": 0.0, "Dc": 0.3, "Df": 0.7},
    { "Dp": 0.0, "Dc": 0.5, "Df": 0.5},
    { "Dp": 0.0, "Dc": 0.7, "Df": 0.3},
    { "Dp": 0.0, "Dc": 0.9, "Df": 0.1},

    #Pure current
    { "Dp": 0.0, "Dc": 1.0,"Df": 0.0},

    #Tradeoff between current and future + previous
    { "Dp":0.1, "Dc":0.8, "Df":0.1 },
    { "Dp":0.2, "Dc":0.6, "Df":0.2 },
    { "Dp":0.3, "Dc":0.4, "Df":0.3 },
    { "Dp":0.4, "Dc":0.2, "Df":0.4 },
    { "Dp":0.5, "Dc":0.0, "Df":0.5 },

    #Tradeoff between current and previous (with some fixed future)
    { "Dp":0.2, "Dc":0.7, "Df": 0.1},
    { "Dp":0.4, "Dc":0.5, "Df": 0.1},
    { "Dp":0.6, "Dc":0.3, "Df": 0.1},
    { "Dp":0.8, "Dc":0.1, "Df": 0.1},
    { "Dp":0.1, "Dc":0.6, "Df": 0.3},
    { "Dp":0.3, "Dc":0.4, "Df": 0.3},
    { "Dp":0.5, "Dc":0.2, "Df": 0.3},

    #Tradeoff between current and future (with some fixed previous)
    { "Dp": 0.1, "Dc":0.7, "Df": 0.2},
    { "Dp": 0.1, "Dc":0.5, "Df": 0.4},
    { "Dp": 0.1, "Dc":0.3, "Df": 0.6},
    { "Dp": 0.1, "Dc":0.1, "Df": 0.8},
    { "Dp": 0.3, "Dc":0.6, "Df": 0.1},
    { "Dp": 0.3, "Dc":0.4, "Df": 0.3},
    { "Dp": 0.3, "Dc":0.2, "Df": 0.5}
]

task_counts = [5, 10, 30]

model_ids = [0, 1, 2]

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

import json
filename = "configurations"

with open(filename + '.json', 'w') as fp:
    json.dump(model_configs, fp, indent=4)
