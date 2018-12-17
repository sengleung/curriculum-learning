import json
import results.results as results

results_filepath = './results/data'

distributions = {
    #Tradeoff between previous and current
    "group0" : [
        #{ "Dp": 0.9 ,"Dc": 0.1, "Df": 0.0 },
        { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 },
        { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 },
        { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 },
        #{ "Dp": 0.1 ,"Dc": 0.9, "Df": 0.0 }
    ],

    #Tradeoff between forward and current
    "group1" : [
        #{ "Dp": 0.0, "Dc": 0.1, "Df": 0.9},
        { "Dp": 0.0, "Dc": 0.3, "Df": 0.7},
        { "Dp": 0.0, "Dc": 0.5, "Df": 0.5},
        { "Dp": 0.0, "Dc": 0.7, "Df": 0.3},
        #{ "Dp": 0.0, "Dc": 0.9, "Df": 0.1}
    ],

    #Tradeoff between current and future + previous
    "group2" : [
        { "Dp":0.1, "Dc":0.8, "Df":0.1 },
        #{ "Dp":0.2, "Dc":0.6, "Df":0.2 },
        { "Dp":0.3, "Dc":0.4, "Df":0.3 },
        #{ "Dp":0.4, "Dc":0.2, "Df":0.4 },
        { "Dp":0.5, "Dc":0.0, "Df":0.5 }
    ],

    #Tradeoff between current and previous (with some fixed future)
    "group3" : [
        { "Dp":0.2, "Dc":0.7, "Df": 0.1},
        #{ "Dp":0.4, "Dc":0.5, "Df": 0.1},
        { "Dp":0.6, "Dc":0.3, "Df": 0.1},
        #{ "Dp":0.8, "Dc":0.1, "Df": 0.1},
        { "Dp":0.1, "Dc":0.6, "Df": 0.3},
        #{ "Dp":0.3, "Dc":0.4, "Df": 0.3},
        { "Dp":0.5, "Dc":0.2, "Df": 0.3}
    ],

    #Tradeoff between current and future (with some fixed previous)
    "group4" : [
        { "Dp": 0.1, "Dc":0.7, "Df": 0.2},
        #{ "Dp": 0.1, "Dc":0.5, "Df": 0.4},
        { "Dp": 0.1, "Dc":0.3, "Df": 0.6},
        #{ "Dp": 0.1, "Dc":0.1, "Df": 0.8},
        { "Dp": 0.3, "Dc":0.6, "Df": 0.1},
        #{ "Dp": 0.3, "Dc":0.4, "Df": 0.3},
        { "Dp": 0.3, "Dc":0.2, "Df": 0.5}
    ]

    #Pure current
    "groupPureCurrent" : [
        { "Dp": 0.0, "Dc": 1.0,"Df": 0.0}
    ],

}

models = [0]
task_counts = [5, 10, 25]

def model_name(id, tasks, distribution):
    name = "id{0}_t{1}_dp{2}_dc{3}_df{4}".format(
        id, tasks, distribution['Dp'], distribution['Dc'], distribution['Df']
    )
    return name

def accuracy_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seen'])
        ys.append(result_point['categorial_accuracy'])
    return xs, ys

def top2_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seen'])
        ys.append(result_point['top_2_accuracy'])
    return xs, ys

def loss_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seen'])
        ys.append(result_point['loss'])
    return xs, ys

def model_average(tasks, distribution):

    for model in models:

#Graph group0 together

#Graph group1 together

#Graph group2 together

#Graph group3 together

#Graph group4 together
