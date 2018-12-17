import json
import results.results as results
import matplotlib.pyplot as plt

results_filepath = './results/data'

distributions = {
    #Tradeoff between previous and current
    "group0" : [
        { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 },
        { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 },
        { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 },
    ],

    #Tradeoff between forward and current
    "group1" : [
        { "Dp": 0.0, "Dc": 0.3, "Df": 0.7},
        { "Dp": 0.0, "Dc": 0.5, "Df": 0.5},
        { "Dp": 0.0, "Dc": 0.7, "Df": 0.3},
    ],

    #Tradeoff between current and future + previous
    "group2" : [
        { "Dp":0.1, "Dc":0.8, "Df":0.1 },
        { "Dp":0.3, "Dc":0.4, "Df":0.3 },
        { "Dp":0.5, "Dc":0.0, "Df":0.5 }
    ],

    #Tradeoff between current and previous (with some fixed future)
    "group3" : [
        { "Dp":0.2, "Dc":0.7, "Df": 0.1},
        { "Dp":0.6, "Dc":0.3, "Df": 0.1},
        { "Dp":0.1, "Dc":0.6, "Df": 0.3},
        { "Dp":0.5, "Dc":0.2, "Df": 0.3}
    ],

    #Tradeoff between current and future (with some fixed previous)
    "group4" : [
        { "Dp": 0.1, "Dc":0.7, "Df": 0.2},
        { "Dp": 0.1, "Dc":0.3, "Df": 0.6},
        { "Dp": 0.3, "Dc":0.6, "Df": 0.1},
        { "Dp": 0.3, "Dc":0.2, "Df": 0.5}
    ],

    #Pure current
    "groupPureCurrent" : [
        { "Dp": 0.0, "Dc": 1.0,"Df": 0.0}
    ]
}

models = [0, 1, 2]
task_counts = [5, 10, 25]
plot_directory = './plots/'

def model_name(id, tasks, distribution):
    name = ""
    if distribution == "unsorted":
        name = "id{0}_t{1}_unsorted".format(id, tasks)
    elif distribution == "sorted":
        name = "id{0}_t{1}_sorted".format(id, tasks)
    else:
        name = "id{0}_t{1}_dp{2}_dc{3}_df{4}".format(
            id, tasks, distribution['Dp'], distribution['Dc'], distribution['Df']
        )
    return name

def accuracy_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seens'])
        ys.append(result_point['categorial_accuracy'])
    return xs, ys

def top2_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seens'])
        ys.append(result_point['top_2_accuracy'])
    return xs, ys

def loss_vs_samples_seen(model_results):
    xs = []
    ys = []
    for result_point in model_results['results']:
        xs.append(result_point['samples_seens'])
        ys.append(result_point['loss'])
    return xs, ys

def average(lists):
    list_amount = float(len(lists))
    list_size = len(lists[0])
    average = [0] * list_size
    for i in range(0, list_size):
        sum = 0
        for l in lists:
            sum += l[i]
        average[i] = sum / list_amount
    return average

def get_model_results(id, tasks, distribution):
    mname = model_name(id, tasks, distribution)
    model_results = results.get(results_filepath, mname)
    return model_results

#models specify which models to get results from
def get_loss_average(models, task, distribution):
    models_losses = []
    models_samples_seen = []
    for model in models:
        model_results = get_model_results(model, task, distribution)
        losses, samples_seen = loss_vs_samples_seen(model_results)
        models_losses.append(losses)
        models_samples_seen.append(samples_seen)
    return average(models_samples_seen),  average(models_losses)

def get_accuracy_average(models, task, distribution):
    models_accuracies = []
    models_samples_seen = []
    for model in models:
        model_results = get_model_results(model, task, distribulosstion)
        accuracies, samples_seen = accuracy_vs_samples_seen(model_results)
        models_accuracies.append(accuracies)
        models_samples_seen.append(samples_seen)
    return average(models_samples_seen), average(models_accuracies)

def get_top2_average(models, task, distribution):
    models_top2s = []
    models_samples_seen = []
    for model in models:
        model_results = get_model_results(model, task, distribution)
        top2s, samples_seen = top2_vs_samples_seen(model_results)
        models_top2s.append(top2s)
        models_samples_seen.append(samples_seen)
    return average(models_samples_seen), average(models_top2s)

def get_all_averages(models, task, distribution):
    loss, samples_seen = get_loss_average(models, task, distribution)
    accuracy, samples_seen = get_accuracy_average(models, task, distribution)
    top2, samples_seen = get_top2_average(models, task, distribution)
    return (loss, accuracy, top2, samples_seen)

def loss(averages_collection):
    return averages_collection[0]

def accuracy(averages_collection):
    return averages_collection[1]

def top2(averages_collection):
    return averages_collection[2]

def samples_seen(averages_collection):
    return averages_collection[3]

#models = [(id, task, distribution)]
def plot_loss_results(models):
    plt.xlim(0, 300000)
    for model in models:
        model_results = get_model_results(model[0], model[1], model[2])
        xs, ys = loss_vs_samples_seen(model_results)
        plt.plot(xs, ys)
    plt.show()

all_models = [0,1,2]

def plot_baseline_comparison_between_tasks():
    unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
    unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
    unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
    sorted_t5 = get_all_averages(all_models, 5, "sorted")
    sorted_t10 = get_all_averages(all_models, 10, "sorted")
    sorted_t25 = get_all_averages(all_models, 25, "sorted")
    pure_current_t5 = get_all_averages(all_models, 5, { "Dp": 0.0, "Dc": 1.0,"Df": 0.0})
    pure_current_t10 = get_all_averages(all_models, 10, { "Dp": 0.0, "Dc": 1.0,"Df": 0.0})
    pure_current_t25 = get_all_averages(all_models, 25, { "Dp": 0.0, "Dc": 1.0,"Df": 0.0})
    fig, axs = plt.subplots(3,1)

    #loss axs[0]
    axs[0].set_title("Loss vs Samples Seen")
    axs[0].set_xlim(0,300000)
    axs[0].set_ylabel("Loss")

    axs[0].plot(samples_seen(unsorted_t5), loss(unsorted_t5), label='unsorted 5 tasks', color="b", marker='s')
    axs[0].plot(samples_seen(unsorted_t10), loss(unsorted_t10), label='unsorted 10 tasks', color="b", marker='o')
    axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25), label='unsorted 25 tasks', color="b", marker='.')

    axs[0].plot(samples_seen(sorted_t5), loss(sorted_t5), label='sorted 5 tasks', color="r", marker='s')
    axs[0].plot(samples_seen(sorted_t10), loss(sorted_t10), label='sorted 10 tasks', color="r", marker='o')
    axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25), label='sorted 25 tasks', color="r", marker='.')

    axs[0].plot(samples_seen(pure_current_t5), loss(pure_current_t5), label='single task 5 tasks', color="g", marker='s')
    axs[0].plot(samples_seen(pure_current_t10), loss(pure_current_t10), label='single task 10 tasks', color="g", marker='o')
    axs[0].plot(samples_seen(pure_current_t25), loss(pure_current_t25), label='single task 25 tasks', color="g", marker='.')

    axs[0].legend()

    #accuracy axs[1]
    axs[1].set_title("Accuracy vs Samples Seen")
    axs[1].set_xlim(0,300000)
    axs[1].set_ylim(0,1.0)
    axs[1].set_ylabel("Accuracy")

    axs[1].plot(samples_seen(unsorted_t5), accuracy(unsorted_t5), label='unsorted 5 tasks', color="b", marker='s')
    axs[1].plot(samples_seen(unsorted_t10), accuracy(unsorted_t10), label='unsorted 10 tasks', color="b", marker='o')
    axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25), label='unsorted 25 tasks', color="b", marker='.')

    axs[1].plot(samples_seen(sorted_t5), accuracy(sorted_t5), label='sorted 5 tasks', color="r", marker='s')
    axs[1].plot(samples_seen(sorted_t10), accuracy(sorted_t10), label='sorted 10 tasks', color="r", marker='d')
    axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25), label='sorted 25 tasks', color="r", marker='d')

    axs[1].plot(samples_seen(pure_current_t5), accuracy(pure_current_t5), label='single task 5 tasks', color="g", marker='s')
    axs[1].plot(samples_seen(pure_current_t10), accuracy(pure_current_t10), label='single task 10 tasks', color="g", marker='o')
    axs[1].plot(samples_seen(pure_current_t25), accuracy(pure_current_t25), label='single task 25 tasks', color="g", marker='.')

    axs[1].legend()

    #top2 axs[2]
    axs[2].set_title("Top 2 Accuracy vs Samples Seen")
    axs[2].set_xlim(0,300000)
    axs[2].set_ylim(0,1.0)
    axs[2].set_ylabel("Top 2 Accuracy")

    axs[2].plot(samples_seen(unsorted_t5), top2(unsorted_t5), label='unsorted 5 tasks', color="b", marker='s')
    axs[2].plot(samples_seen(unsorted_t10), top2(unsorted_t10), label='unsorted 10 tasks', color="b", marker='o')
    axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25), label='unsorted 25 tasks', color="b", marker='.')

    axs[2].plot(samples_seen(sorted_t5), top2(sorted_t5), label='sorted 5 tasks', color="r", marker='s')
    axs[2].plot(samples_seen(sorted_t10), top2(sorted_t10), label='sorted 10 tasks', color="r", marker='d')
    axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25), label='sorted 25 tasks', color="r", marker='d')

    axs[2].plot(samples_seen(pure_current_t5), top2(pure_current_t5), label='single task 5 tasks', color="g", marker='s')
    axs[2].plot(samples_seen(pure_current_t10), top2(pure_current_t10), label='single task 10 tasks', color="g", marker='o')
    axs[2].plot(samples_seen(pure_current_t25), top2(pure_current_t25), label='single task 25 tasks', color="g", marker='.')

    axs[2].legend()
    plt.savefig(plot_directory + 'Baselines.png')
    plt.show()

def plot_group0_comparisons():
    #5 tasks
    d1_t5 = get_all_averages(all_models, 5, { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 })
    d2_t5 = get_all_averages(all_models, 5, { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 })
    d3_t5 = get_all_averages(all_models, 5, { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 })
    unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
    sorted_t5 = get_all_averages(all_models, 5, "sorted")

    #10 tasks
    d1_t10 = get_all_averages(all_models, 10, { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 })
    d2_t10 = get_all_averages(all_models, 10, { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 })
    d3_t10 = get_all_averages(all_models, 10, { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 })
    unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
    sorted_t10 = get_all_averages(all_models, 10, "sorted")

    #25 tasks
    d1_t25 = get_all_averages(all_models, 25, { "Dp": 0.7 ,"Dc": 0.3, "Df": 0.0 })
    d2_t25 = get_all_averages(all_models, 25, { "Dp": 0.5 ,"Dc": 0.5, "Df": 0.0 })
    d3_t25 = get_all_averages(all_models, 25, { "Dp": 0.3 ,"Dc": 0.7, "Df": 0.0 })
    unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
    sorted_t25 = get_all_averages(all_models, 25, "sorted")

    fig, axs = plt.subplots(3,1)

    #loss axs[0]
    axs[0].set_title("Loss vs Samples Seen")
    axs[0].set_xlim(0,300000)
    axs[0].set_ylabel("Loss")

    axs[0].plot(samples_seen(d1_t5), loss(d1_t5), label='Prev: 70% | Cur: 30% 5 Tasks', color='y', marker='s')
    axs[0].plot(samples_seen(d1_t10), loss(d1_t10), label='Prev: 70% | Cur: 30% 10 Tasks', color='y', marker='o')
    axs[0].plot(samples_seen(d1_t25), loss(d1_t25), label='Prev: 70% | Cur: 30% 25 Tasks', color='y', marker='.')

    axs[0].plot(samples_seen(d2_t5), loss(d2_t5), label='Prev: 50% | Cur: 50% 5 Tasks', color='c', marker='s')
    axs[0].plot(samples_seen(d2_t10), loss(d2_t10), label='Prev: 50% | Cur: 50% 10 Tasks', color='c', marker='o')
    axs[0].plot(samples_seen(d2_t25), loss(d2_t25), label='Prev: 50% | Cur: 50% 25 Tasks', color='c', marker='.')

    axs[0].plot(samples_seen(d3_t5), loss(d3_t5), label='Prev: 30% | Cur: 70% 5 Tasks', color='g', marker='s')
    axs[0].plot(samples_seen(d3_t10), loss(d3_t10), label='Prev: 30% | Cur: 70% 10 Tasks', color='g', marker='o')
    axs[0].plot(samples_seen(d3_t25), loss(d3_t25), label='Prev: 30% | Cur: 70% 25 Tasks', color='g', marker='.')

    axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25) label='unsorted', color='b')
    axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25) label='sorted', color='r')

    axs[0].legend()

    #accuracy axs[0]
    axs[1].set_title("Accuracy vs Samples Seen")
    axs[1].set_xlim(0,300000)
    axs[1].set_ylim(0,1.0)

    axs[1].plot(samples_seen(d1_t5), accuracy(d1_t5), label='Prev: 70% | Cur: 30% 5 Tasks', color='y', marker='s')
    axs[1].plot(samples_seen(d1_t10), accuracy(d1_t10), label='Prev: 70% | Cur: 30% 10 Tasks', color='y', marker='o')
    axs[1].plot(samples_seen(d1_t25), accuracy(d1_t25), label='Prev: 70% | Cur: 30% 25 Tasks', color='y', marker='.')

    axs[1].plot(samples_seen(d2_t5), accuracy(d2_t5), label='Prev: 50% | Cur: 50% 5 Tasks', color='c', marker='s')
    axs[1].plot(samples_seen(d2_t10), accuracy(d2_t10), label='Prev: 50% | Cur: 50% 10 Tasks', color='c', marker='o')
    axs[1].plot(samples_seen(d2_t25), accuracy(d2_t25), label='Prev: 50% | Cur: 50% 25 Tasks', color='c', marker='.')

    axs[1].plot(samples_seen(d3_t5), accuracy(d3_t5), label='Prev: 30% | Cur: 70% 5 Tasks', color='g', marker='s')
    axs[1].plot(samples_seen(d3_t10), accuracy(d3_t10), label='Prev: 30% | Cur: 70% 10 Tasks', color='g', marker='o')
    axs[1].plot(samples_seen(d3_t25), accuracy(d3_t25), label='Prev: 30% | Cur: 70% 25 Tasks', color='g', marker='.')

    axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25) label='unsorted', color='b')
    axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25) label='sorted', color='r')

    axs[1].legend()

    #top2 axs[0]
    axs[2].set_title("Top 2 Accuracy vs Samples Seen")
    axs[2].set_xlim(0,300000)
    axs[2].set_ylim(0,1.0)
    axs[2].set_ylabel("Top 2 Accuracy")

    axs[2].plot(samples_seen(d1_t5), top2(d1_t5), label='Prev: 70% | Cur: 30% 5 Tasks', color='y', marker='s')
    axs[2].plot(samples_seen(d1_t10), top2(d1_t10), label='Prev: 70% | Cur: 30% 10 Tasks', color='y', marker='o')
    axs[2].plot(samples_seen(d1_t25), top2(d1_t25), label='Prev: 70% | Cur: 30% 25 Tasks', color='y', marker='.')

    axs[2].plot(samples_seen(d2_t5), top2(d2_t5), label='Prev: 50% | Cur: 50% 5 Tasks', color='c', marker='s')
    axs[2].plot(samples_seen(d2_t10), top2(d2_t10), label='Prev: 50% | Cur: 50% 10 Tasks', color='c', marker='o')
    axs[2].plot(samples_seen(d2_t25), top2(d2_t25), label='Prev: 50% | Cur: 50% 25 Tasks', color='c', marker='.')

    axs[2].plot(samples_seen(d3_t5), top2(d3_t5), label='Prev: 30% | Cur: 70% 5 Tasks', color='g', marker='s')
    axs[2].plot(samples_seen(d3_t10), top2(d3_t10), label='Prev: 30% | Cur: 70% 10 Tasks', color='g', marker='o')
    axs[2].plot(samples_seen(d3_t25), top2(d3_t25), label='Prev: 30% | Cur: 70% 25 Tasks', color='g', marker='.')

    axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25) label='unsorted', color='b')
    axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25) label='sorted', color='r')

    axs[2].legend()
    plt.savefig(plot_directory + 'Group0.png')
    plt.show()

def plot_group1_comparisons():
    #5 tasks
    d1_t5 = get_all_averages(all_models, 5, { "Dp": 0.0, "Dc": 0.3, "Df": 0.7})
    d2_t5 = get_all_averages(all_models, 5, { "Dp": 0.0, "Dc": 0.5, "Df": 0.5})
    d3_t5 = get_all_averages(all_models, 5, { "Dp": 0.0, "Dc": 0.7, "Df": 0.3})
    unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
    sorted_t5 = get_all_averages(all_models, 5, "sorted")

    #10 tasks
    d1_t10 = get_all_averages(all_models, 10, { "Dp": 0.0, "Dc": 0.3, "Df": 0.7})
    d2_t10 = get_all_averages(all_models, 10, { "Dp": 0.0, "Dc": 0.5, "Df": 0.5})
    d3_t10 = get_all_averages(all_models, 10, { "Dp": 0.0, "Dc": 0.7, "Df": 0.3})
    unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
    sorted_t10 = get_all_averages(all_models, 10, "sorted")

    #25 tasks
    d1_t25 = get_all_averages(all_models, 25, { "Dp": 0.0, "Dc": 0.3, "Df": 0.7})
    d2_t25 = get_all_averages(all_models, 25, { "Dp": 0.0, "Dc": 0.5, "Df": 0.5})
    d3_t25 = get_all_averages(all_models, 25, { "Dp": 0.0, "Dc": 0.7, "Df": 0.3})
    unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
    sorted_t25 = get_all_averages(all_models, 25, "sorted")

    fig, axs = plt.subplots(3,1)

    #loss axs[0]
    axs[0].set_title("Loss vs Samples Seen")
    axs[0].set_xlim(0,300000)
    axs[0].set_ylabel("Loss")

    axs[0].plot(samples_seen(d1_t5), loss(d1_t5), label='Cur: 30% | Fut: 70% 5 Tasks', color='y', marker='s')
    axs[0].plot(samples_seen(d1_t10), loss(d1_t10), label='Cur: 30% | Fut: 70% 10 Tasks', color='y', marker='o')
    axs[0].plot(samples_seen(d1_t25), loss(d1_t25), label='Cur: 30% | Fut: 70% 25 Tasks', color='y', marker='.')

    axs[0].plot(samples_seen(d2_t5), loss(d2_t5), label='Cur: 50% | Fut: 50% 5 Tasks', color='c', marker='s')
    axs[0].plot(samples_seen(d2_t10), loss(d2_t10), label='Cur: 50% | Fut: 50% 10 Tasks', color='c', marker='o')
    axs[0].plot(samples_seen(d2_t25), loss(d2_t25), label='Cur: 50% | Fut: 50% 25 Tasks', color='c', marker='.')

    axs[0].plot(samples_seen(d3_t5), loss(d3_t5), label='Cur: 70% | Fut: 30% 5 Tasks', color='g', marker='s')
    axs[0].plot(samples_seen(d3_t10), loss(d3_t10), label='Cur: 70% | Fut: 30% 10 Tasks', color='g', marker='o')
    axs[0].plot(samples_seen(d3_t25), loss(d3_t25), label='Cur: 70% | Fut: 30% 25 Tasks', color='g', marker='.')

    axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25) label='unsorted', color='b')
    axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25) label='sorted', color='r')

    axs[0].legend()

    #accuracy axs[0]
    axs[1].set_title("Accuracy vs Samples Seen")
    axs[1].set_xlim(0,300000)
    axs[1].set_ylim(0,1.0)

    axs[1].plot(samples_seen(d1_t5), accuracy(d1_t5), label='Cur: 30% | Fut: 70% 5 Tasks', color='y', marker='s')
    axs[1].plot(samples_seen(d1_t10), accuracy(d1_t10), label='Cur: 30% | Fut: 70% 10 Tasks', color='y', marker='o')
    axs[1].plot(samples_seen(d1_t25), accuracy(d1_t25), label='Cur: 30% | Fut: 70% 25 Tasks', color='y', marker='.')

    axs[1].plot(samples_seen(d2_t5), accuracy(d2_t5), label='Cur: 50% | Fut: 50% 5 Tasks', color='c', marker='s')
    axs[1].plot(samples_seen(d2_t10), accuracy(d2_t10), label='Cur: 50% | Fut: 50% 10 Tasks', color='c', marker='o')
    axs[1].plot(samples_seen(d2_t25), accuracy(d2_t25), label='Cur: 50% | Fut: 50% 25 Tasks', color='c', marker='.')

    axs[1].plot(samples_seen(d3_t5), accuracy(d3_t5), label='Cur: 70% | Fut: 30% 5 Tasks', color='g', marker='s')
    axs[1].plot(samples_seen(d3_t10), accuracy(d3_t10), label='Cur: 70% | Fut: 30% 10 Tasks', color='g', marker='o')
    axs[1].plot(samples_seen(d3_t25), accuracy(d3_t25), label='Cur: 70% | Fut: 30% 25 Tasks', color='g', marker='.')

    axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25) label='unsorted', color='b')
    axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25) label='sorted', color='r')

    axs[1].legend()

    #top2 axs[0]
    axs[2].set_title("Top 2 Accuracy vs Samples Seen")
    axs[2].set_xlim(0,300000)
    axs[2].set_ylim(0,1.0)
    axs[2].set_ylabel("Top 2 Accuracy")

    axs[2].plot(samples_seen(d1_t5), top2(d1_t5), label='Cur: 30% | Fut: 70% 5 Tasks', color='y', marker='s')
    axs[2].plot(samples_seen(d1_t10), top2(d1_t10), label='Cur: 30% | Fut: 70% 10 Tasks', color='y', marker='o')
    axs[2].plot(samples_seen(d1_t25), top2(d1_t25), label='Cur: 30% | Fut: 70% 25 Tasks', color='y', marker='.')

    axs[2].plot(samples_seen(d2_t5), top2(d2_t5), label='Cur: 50% | Fut: 50% 5 Tasks', color='c', marker='s')
    axs[2].plot(samples_seen(d2_t10), top2(d2_t10), label='Cur: 50% | Fut: 50% 10 Tasks', color='c', marker='o')
    axs[2].plot(samples_seen(d2_t25), top2(d2_t25), label='Cur: 50% | Fut: 50% 25 Tasks', color='c', marker='.')

    axs[2].plot(samples_seen(d3_t5), top2(d3_t5), label='Cur: 70% | Fut: 30% 5 Tasks', color='g', marker='s')
    axs[2].plot(samples_seen(d3_t10), top2(d3_t10), label='Cur: 70% | Fut: 30% 10 Tasks', color='g', marker='o')
    axs[2].plot(samples_seen(d3_t25), top2(d3_t25), label='Cur: 70% | Fut: 30% 25 Tasks', color='g', marker='.')

    axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25) label='unsorted', color='b')
    axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25) label='sorted', color='r')

    axs[2].legend()
    plt.savefig(plot_directory + 'Group1.png')
    plt.show()

def plot_group2_comparisons():
    #5 tasks
    d1_t5 = get_all_averages(all_models, 5, { "Dp":0.1, "Dc":0.8, "Df":0.1 })
    d2_t5 = get_all_averages(all_models, 5, { "Dp":0.3, "Dc":0.4, "Df":0.3 })
    d3_t5 = get_all_averages(all_models, 5, { "Dp":0.5, "Dc":0.0, "Df":0.5 })
    unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
    sorted_t5 = get_all_averages(all_models, 5, "sorted")

    #10 tasks
    d1_t10 = get_all_averages(all_models, 10, { "Dp":0.1, "Dc":0.8, "Df":0.1 })
    d2_t10 = get_all_averages(all_models, 10, { "Dp":0.3, "Dc":0.4, "Df":0.3 })
    d3_t10 = get_all_averages(all_models, 10, { "Dp":0.5, "Dc":0.0, "Df":0.5 })
    unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
    sorted_t10 = get_all_averages(all_models, 10, "sorted")

    #25 tasks
    d1_t25 = get_all_averages(all_models, 25, { "Dp":0.1, "Dc":0.8, "Df":0.1 })
    d2_t25 = get_all_averages(all_models, 25, { "Dp":0.3, "Dc":0.4, "Df":0.3 })
    d3_t25 = get_all_averages(all_models, 25, { "Dp":0.5, "Dc":0.0, "Df":0.5 })
    unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
    sorted_t25 = get_all_averages(all_models, 25, "sorted")

    fig, axs = plt.subplots(3,1)

    #loss axs[0]
    axs[0].set_title("Loss vs Samples Seen")
    axs[0].set_xlim(0,300000)
    axs[0].set_ylabel("Loss")

    axs[0].plot(samples_seen(d1_t5), loss(d1_t5), label='P:10% | C:80% | F:10% 5 Tasks', color='y', marker='s')
    axs[0].plot(samples_seen(d1_t10), loss(d1_t10), label='P:10% | C:80% | F:10% 10 Tasks', color='y', marker='o')
    axs[0].plot(samples_seen(d1_t25), loss(d1_t25), label='P:10% | C:80% | F:10% 25 Tasks', color='y', marker='.')

    axs[0].plot(samples_seen(d2_t5), loss(d2_t5), label='P:30% | C:40% | F:30% 5 Tasks', color='c', marker='s')
    axs[0].plot(samples_seen(d2_t10), loss(d2_t10), label='P:30% | C:40% | F:30% 10 Tasks', color='c', marker='o')
    axs[0].plot(samples_seen(d2_t25), loss(d2_t25), label='P:30% | C:40% | F:30% 25 Tasks', color='c', marker='.')

    axs[0].plot(samples_seen(d3_t5), loss(d3_t5), label='P:50% | C:0% | F:50% 5 Tasks', color='g', marker='s')
    axs[0].plot(samples_seen(d3_t10), loss(d3_t10), label='P:50% | C:0% | F:50% 10 Tasks', color='g', marker='o')
    axs[0].plot(samples_seen(d3_t25), loss(d3_t25), label='P:50% | C:0% | F:50% 25 Tasks', color='g', marker='.')

    axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25) label='unsorted', color='b')
    axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25) label='sorted', color='r')

    axs[0].legend()

    #accuracy axs[0]
    axs[1].set_title("Accuracy vs Samples Seen")
    axs[1].set_xlim(0,300000)
    axs[1].set_ylim(0,1.0)

    axs[1].plot(samples_seen(d1_t5), accuracy(d1_t5), label='P:10% | C:80% | F:10% 5 Tasks', color='y', marker='s')
    axs[1].plot(samples_seen(d1_t10), accuracy(d1_t10), label='P:10% | C:80% | F:10% 10 Tasks', color='y', marker='o')
    axs[1].plot(samples_seen(d1_t25), accuracy(d1_t25), label='P:10% | C:80% | F:10% 25 Tasks', color='y', marker='.')

    axs[1].plot(samples_seen(d2_t5), accuracy(d2_t5), label='P:30% | C:40% | F:30% 5 Tasks', color='c', marker='s')
    axs[1].plot(samples_seen(d2_t10), accuracy(d2_t10), label='P:30% | C:40% | F:30% 10 Tasks', color='c', marker='o')
    axs[1].plot(samples_seen(d2_t25), accuracy(d2_t25), label='P:30% | C:40% | F:30% 25 Tasks', color='c', marker='.')

    axs[1].plot(samples_seen(d3_t5), accuracy(d3_t5), label='P:50% | C:0% | F:50% 5 Tasks', color='g', marker='s')
    axs[1].plot(samples_seen(d3_t10), accuracy(d3_t10), label='P:50% | C:0% | F:50% 10 Tasks', color='g', marker='o')
    axs[1].plot(samples_seen(d3_t25), accuracy(d3_t25), label='P:50% | C:0% | F:50% 25 Tasks', color='g', marker='.')

    axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25) label='unsorted', color='b')
    axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25) label='sorted', color='r')

    axs[1].legend()

    #top2 axs[0]
    axs[2].set_title("Top 2 Accuracy vs Samples Seen")
    axs[2].set_xlim(0,300000)
    axs[2].set_ylim(0,1.0)
    axs[2].set_ylabel("Top 2 Accuracy")

    axs[2].plot(samples_seen(d1_t5), top2(d1_t5), label='P:10% | C:80% | F:10% 5 Tasks', color='y', marker='s')
    axs[2].plot(samples_seen(d1_t10), top2(d1_t10), label='P:10% | C:80% | F:10% 10 Tasks', color='y', marker='o')
    axs[2].plot(samples_seen(d1_t25), top2(d1_t25), label='P:10% | C:80% | F:10% 25 Tasks', color='y', marker='.')

    axs[2].plot(samples_seen(d2_t5), top2(d2_t5), label='P:30% | C:40% | F:30% 5 Tasks', color='c', marker='s')
    axs[2].plot(samples_seen(d2_t10), top2(d2_t10), label='P:30% | C:40% | F:30% 10 Tasks', color='c', marker='o')
    axs[2].plot(samples_seen(d2_t25), top2(d2_t25), label='P:30% | C:40% | F:30% 25 Tasks', color='c', marker='.')

    axs[2].plot(samples_seen(d3_t5), top2(d3_t5), label='P:50% | C:0% | F:50% 5 Tasks', color='g', marker='s')
    axs[2].plot(samples_seen(d3_t10), top2(d3_t10), label='P:50% | C:0% | F:50% 10 Tasks', color='g', marker='o')
    axs[2].plot(samples_seen(d3_t25), top2(d3_t25), label='P:50% | C:0% | F:50% 25 Tasks', color='g', marker='.')

    axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25) label='unsorted', color='b')
    axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25) label='sorted', color='r')

    axs[2].legend()
    plt.savefig(plot_directory + 'Group2.png')
    plt.show()

def plot_group3_comparisons():
    #5 tasks
    d1_t5 = get_all_averages(all_models, 5, { "Dp":0.2, "Dc":0.7, "Df": 0.1})
    d2_t5 = get_all_averages(all_models, 5, { "Dp":0.6, "Dc":0.3, "Df": 0.1})
    d3_t5 = get_all_averages(all_models, 5, { "Dp":0.1, "Dc":0.6, "Df": 0.3})
    d4_t5 = get_all_averages(all_models, 5, { "Dp":0.5, "Dc":0.2, "Df": 0.3})
    unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
    sorted_t5 = get_all_averages(all_models, 5, "sorted")

    #10 tasks
    d1_t10 = get_all_averages(all_models, 10, { "Dp":0.2, "Dc":0.7, "Df": 0.1})
    d2_t10 = get_all_averages(all_models, 10, { "Dp":0.6, "Dc":0.3, "Df": 0.1})
    d3_t10 = get_all_averages(all_models, 10, { "Dp":0.1, "Dc":0.6, "Df": 0.3})
    d4_t10 = get_all_averages(all_models, 10, { "Dp":0.5, "Dc":0.2, "Df": 0.3})
    unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
    sorted_t10 = get_all_averages(all_models, 10, "sorted")

    #25 tasks
    d1_t25 = get_all_averages(all_models, 25, { "Dp":0.2, "Dc":0.7, "Df": 0.1})
    d2_t25 = get_all_averages(all_models, 25, { "Dp":0.6, "Dc":0.3, "Df": 0.1})
    d3_t25 = get_all_averages(all_models, 25, { "Dp":0.1, "Dc":0.6, "Df": 0.3})
    d4_t25 = get_all_averages(all_models, 25, { "Dp":0.5, "Dc":0.2, "Df": 0.3})
    unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
    sorted_t25 = get_all_averages(all_models, 25, "sorted")

    fig, axs = plt.subplots(3,1)

    #loss axs[0]
    axs[0].set_title("Loss vs Samples Seen")
    axs[0].set_xlim(0,300000)
    axs[0].set_ylabel("Loss")

    axs[0].plot(samples_seen(d1_t5), loss(d1_t5), label='P:20% | C:70% | F:10% 5 Tasks', color='y', marker='s')
    axs[0].plot(samples_seen(d1_t10), loss(d1_t10), label='P:20% | C:70% | F:10% 10 Tasks', color='y', marker='o')
    axs[0].plot(samples_seen(d1_t25), loss(d1_t25), label='P:20% | C:70% | F:10% 25 Tasks', color='y', marker='.')

    axs[0].plot(samples_seen(d2_t5), loss(d2_t5), label='P:60% | C:30% | F:10% 5 Tasks', color='c', marker='s')
    axs[0].plot(samples_seen(d2_t10), loss(d2_t10), label='P:60% | C:30% | F:10% 10 Tasks', color='c', marker='o')
    axs[0].plot(samples_seen(d2_t25), loss(d2_t25), label='P:60% | C:30% | F:10% 25 Tasks', color='c', marker='.')

    axs[0].plot(samples_seen(d3_t5), loss(d3_t5), label='P:10% | C:60% | F:30% 5 Tasks', color='g', marker='s')
    axs[0].plot(samples_seen(d3_t10), loss(d3_t10), label='P:10% | C:60% | F:30% 10 Tasks', color='g', marker='o')
    axs[0].plot(samples_seen(d3_t25), loss(d3_t25), label='P:10% | C:60% | F:30% 25 Tasks', color='g', marker='.')

    axs[0].plot(samples_seen(d4_t5), loss(d4_t5), label='P:50% | C:20% | F:30% 5 Tasks', color='m', marker='s')
    axs[0].plot(samples_seen(d4_t10), loss(d4_t10), label='P:50% | C:20% | F:30% 10 Tasks', color='m', marker='o')
    axs[0].plot(samples_seen(d4_t25), loss(d4_t25), label='P:50% | C:20% | F:30% 25 Tasks', color='m', marker='.')

    axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25) label='unsorted', color='b')
    axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25) label='sorted', color='r')

    axs[0].legend()

    #accuracy axs[0]
    axs[1].set_title("Accuracy vs Samples Seen")
    axs[1].set_xlim(0,300000)
    axs[1].set_ylim(0,1.0)

    axs[1].plot(samples_seen(d1_t5), accuracy(d1_t5), label='P:20% | C:70% | F:10% 5 Tasks', color='y', marker='s')
    axs[1].plot(samples_seen(d1_t10), accuracy(d1_t10), label='P:20% | C:70% | F:10% 10 Tasks', color='y', marker='o')
    axs[1].plot(samples_seen(d1_t25), accuracy(d1_t25), label='P:20% | C:70% | F:10% 25 Tasks', color='y', marker='.')

    axs[1].plot(samples_seen(d2_t5), accuracy(d2_t5), label='P:60% | C:30% | F:10% 5 Tasks', color='c', marker='s')
    axs[1].plot(samples_seen(d2_t10), accuracy(d2_t10), label='P:60% | C:30% | F:10% 10 Tasks', color='c', marker='o')
    axs[1].plot(samples_seen(d2_t25), accuracy(d2_t25), label='P:60% | C:30% | F:10% 25 Tasks', color='c', marker='.')

    axs[1].plot(samples_seen(d3_t5), accuracy(d3_t5), label='P:10% | C:60% | F:30% 5 Tasks', color='g', marker='s')
    axs[1].plot(samples_seen(d3_t10), accuracy(d3_t10), label='P:10% | C:60% | F:30% 10 Tasks', color='g', marker='o')
    axs[1].plot(samples_seen(d3_t25), accuracy(d3_t25), label='P:10% | C:60% | F:30% 25 Tasks', color='g', marker='.')

    axs[1].plot(samples_seen(d4_t5), accuracy(d4_t5), label='P:50% | C:20% | F:30% 5 Tasks', color='m', marker='s')
    axs[1].plot(samples_seen(d4_t10), accuracy(d4_t10), label='P:50% | C:20% | F:30% 10 Tasks', color='m', marker='o')
    axs[1].plot(samples_seen(d4_t25), accuracy(d4_t25), label='P:50% | C:20% | F:30% 25 Tasks', color='m', marker='.')

    axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25) label='unsorted', color='b')
    axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25) label='sorted', color='r')

    axs[1].legend()

    #top2 axs[0]
    axs[2].set_title("Top 2 Accuracy vs Samples Seen")
    axs[2].set_xlim(0,300000)
    axs[2].set_ylim(0,1.0)
    axs[2].set_ylabel("Top 2 Accuracy")

    axs[2].plot(samples_seen(d1_t5), top2(d1_t5), label='P:20% | C:70% | F:10% 5 Tasks', color='y', marker='s')
    axs[2].plot(samples_seen(d1_t10), top2(d1_t10), label='P:20% | C:70% | F:10% 10 Tasks', color='y', marker='o')
    axs[2].plot(samples_seen(d1_t25), top2(d1_t25), label='P:20% | C:70% | F:10% 25 Tasks', color='y', marker='.')

    axs[2].plot(samples_seen(d2_t5), top2(d2_t5), label='P:60% | C:30% | F:10% 5 Tasks', color='c', marker='s')
    axs[2].plot(samples_seen(d2_t10), top2(d2_t10), label='P:60% | C:30% | F:10% 10 Tasks', color='c', marker='o')
    axs[2].plot(samples_seen(d2_t25), top2(d2_t25), label='P:60% | C:30% | F:10% 25 Tasks', color='c', marker='.')

    axs[2].plot(samples_seen(d3_t5), top2(d3_t5), label='P:10% | C:60% | F:30% 5 Tasks', color='g', marker='s')
    axs[2].plot(samples_seen(d3_t10), top2(d3_t10), label='P:10% | C:60% | F:30% 10 Tasks', color='g', marker='o')
    axs[2].plot(samples_seen(d3_t25), top2(d3_t25), label='P:10% | C:60% | F:30% 25 Tasks', color='g', marker='.')

    axs[2].plot(samples_seen(d4_t5), top2(d4_t5), label='P:50% | C:20% | F:30% 5 Tasks', color='m', marker='s')
    axs[2].plot(samples_seen(d4_t10), top2(d4_t10), label='P:50% | C:20% | F:30% 10 Tasks', color='m', marker='o')
    axs[2].plot(samples_seen(d4_t25), top2(d4_t25), label='P:50% | C:20% | F:30% 25 Tasks', color='m', marker='.')

    axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25) label='unsorted', color='b')
    axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25) label='sorted', color='r')

    axs[2].legend()
    plt.savefig(plot_directory + 'Group3.png')
    plt.show()

def plot_group4_comparisons():
        #5 tasks
        d1_t5 = get_all_averages(all_models, 5, { "Dp": 0.1, "Dc":0.7, "Df": 0.2})
        d2_t5 = get_all_averages(all_models, 5, { "Dp": 0.1, "Dc":0.3, "Df": 0.6})
        d3_t5 = get_all_averages(all_models, 5, { "Dp": 0.3, "Dc":0.6, "Df": 0.1})
        d4_t5 = get_all_averages(all_models, 5, { "Dp": 0.3, "Dc":0.2, "Df": 0.5})
        unsorted_t5 = get_all_averages(all_models, 5, "unsorted")
        sorted_t5 = get_all_averages(all_models, 5, "sorted")

        #10 tasks
        d1_t10 = get_all_averages(all_models, 10, { "Dp": 0.1, "Dc":0.7, "Df": 0.2})
        d2_t10 = get_all_averages(all_models, 10, { "Dp": 0.1, "Dc":0.3, "Df": 0.6})
        d3_t10 = get_all_averages(all_models, 10, { "Dp": 0.3, "Dc":0.6, "Df": 0.1})
        d4_t10 = get_all_averages(all_models, 10, { "Dp": 0.3, "Dc":0.2, "Df": 0.5})
        unsorted_t10 = get_all_averages(all_models, 10, "unsorted")
        sorted_t10 = get_all_averages(all_models, 10, "sorted")

        #25 tasks
        d1_t5 = get_all_averages(all_models, 25, { "Dp": 0.1, "Dc":0.7, "Df": 0.2})
        d2_t5 = get_all_averages(all_models, 25, { "Dp": 0.1, "Dc":0.3, "Df": 0.6})
        d3_t5 = get_all_averages(all_models, 25, { "Dp": 0.3, "Dc":0.6, "Df": 0.1})
        d4_t5 = get_all_averages(all_models, 25, { "Dp": 0.3, "Dc":0.2, "Df": 0.5})
        unsorted_t25 = get_all_averages(all_models, 25, "unsorted")
        sorted_t25 = get_all_averages(all_models, 25, "sorted")

        fig, axs = plt.subplots(3,1)

        #loss axs[0]
        axs[0].set_title("Loss vs Samples Seen")
        axs[0].set_xlim(0,300000)
        axs[0].set_ylabel("Loss")

        axs[0].plot(samples_seen(d1_t5), loss(d1_t5), label='P:10% | C:70% | F:20% 5 Tasks', color='y', marker='s')
        axs[0].plot(samples_seen(d1_t10), loss(d1_t10), label='P:10% | C:70% | F:20% 10 Tasks', color='y', marker='o')
        axs[0].plot(samples_seen(d1_t25), loss(d1_t25), label='P:10% | C:70% | F:20% 25 Tasks', color='y', marker='.')

        axs[0].plot(samples_seen(d2_t5), loss(d2_t5), label='P:10% | C:30% | F:60% 5 Tasks', color='c', marker='s')
        axs[0].plot(samples_seen(d2_t10), loss(d2_t10), label='P:10% | C:30% | F:60% 10 Tasks', color='c', marker='o')
        axs[0].plot(samples_seen(d2_t25), loss(d2_t25), label='P:10% | C:30% | F:60% 25 Tasks', color='c', marker='.')

        axs[0].plot(samples_seen(d3_t5), loss(d3_t5), label='P:30% | C:60% | F:10% 5 Tasks', color='g', marker='s')
        axs[0].plot(samples_seen(d3_t10), loss(d3_t10), label='P:30% | C:60% | F:10% 10 Tasks', color='g', marker='o')
        axs[0].plot(samples_seen(d3_t25), loss(d3_t25), label='P:30% | C:60% | F:10% 25 Tasks', color='g', marker='.')

        axs[0].plot(samples_seen(d4_t5), loss(d4_t5), label='P:30% | C:20% | F:50% 5 Tasks', color='m', marker='s')
        axs[0].plot(samples_seen(d4_t10), loss(d4_t10), label='P:30% | C:20% | F:50% 10 Tasks', color='m', marker='o')
        axs[0].plot(samples_seen(d4_t25), loss(d4_t25), label='P:30% | C:20% | F:50% 25 Tasks', color='m', marker='.')

        axs[0].plot(samples_seen(unsorted_t25), loss(unsorted_t25) label='unsorted', color='b')
        axs[0].plot(samples_seen(sorted_t25), loss(sorted_t25) label='sorted', color='r')

        axs[0].legend()

        #accuracy axs[0]
        axs[1].set_title("Accuracy vs Samples Seen")
        axs[1].set_xlim(0,300000)
        axs[1].set_ylim(0,1.0)

        axs[1].plot(samples_seen(d1_t5), accuracy(d1_t5), label='P:10% | C:70% | F:20% 5 Tasks', color='y', marker='s')
        axs[1].plot(samples_seen(d1_t10), accuracy(d1_t10), label='P:10% | C:70% | F:20% 10 Tasks', color='y', marker='o')
        axs[1].plot(samples_seen(d1_t25), accuracy(d1_t25), label='P:10% | C:70% | F:20% 25 Tasks', color='y', marker='.')

        axs[1].plot(samples_seen(d2_t5), accuracy(d2_t5), label='P:10% | C:30% | F:60% 5 Tasks', color='c', marker='s')
        axs[1].plot(samples_seen(d2_t10), accuracy(d2_t10), label='P:10% | C:30% | F:60% 10 Tasks', color='c', marker='o')
        axs[1].plot(samples_seen(d2_t25), accuracy(d2_t25), label='P:10% | C:30% | F:60% 25 Tasks', color='c', marker='.')

        axs[1].plot(samples_seen(d3_t5), accuracy(d3_t5), label='P:30% | C:60% | F:10% 5 Tasks', color='g', marker='s')
        axs[1].plot(samples_seen(d3_t10), accuracy(d3_t10), label='P:30% | C:60% | F:10% 10 Tasks', color='g', marker='o')
        axs[1].plot(samples_seen(d3_t25), accuracy(d3_t25), label='P:30% | C:60% | F:10% 25 Tasks', color='g', marker='.')

        axs[1].plot(samples_seen(d4_t5), accuracy(d4_t5), label='P:30% | C:20% | F:50% 5 Tasks', color='m', marker='s')
        axs[1].plot(samples_seen(d4_t10), accuracy(d4_t10), label='P:30% | C:20% | F:50% 10 Tasks', color='m', marker='o')
        axs[1].plot(samples_seen(d4_t25), accuracy(d4_t25), label='P:30% | C:20% | F:50% 25 Tasks', color='m', marker='.')

        axs[1].plot(samples_seen(unsorted_t25), accuracy(unsorted_t25) label='unsorted', color='b')
        axs[1].plot(samples_seen(sorted_t25), accuracy(sorted_t25) label='sorted', color='r')

        axs[1].legend()

        #top2 axs[0]
        axs[2].set_title("Top 2 Accuracy vs Samples Seen")
        axs[2].set_xlim(0,300000)
        axs[2].set_ylim(0,1.0)
        axs[2].set_ylabel("Top 2 Accuracy")

        axs[2].plot(samples_seen(d1_t5), top2(d1_t5), label='P:10% | C:70% | F:20% 5 Tasks', color='y', marker='s')
        axs[2].plot(samples_seen(d1_t10), top2(d1_t10), label='P:10% | C:70% | F:20% 10 Tasks', color='y', marker='o')
        axs[2].plot(samples_seen(d1_t25), top2(d1_t25), label='P:10% | C:70% | F:20% 25 Tasks', color='y', marker='.')

        axs[2].plot(samples_seen(d2_t5), top2(d2_t5), label='P:10% | C:30% | F:60% 5 Tasks', color='c', marker='s')
        axs[2].plot(samples_seen(d2_t10), top2(d2_t10), label='P:10% | C:30% | F:60% 10 Tasks', color='c', marker='o')
        axs[2].plot(samples_seen(d2_t25), top2(d2_t25), label='P:10% | C:30% | F:60% 25 Tasks', color='c', marker='.')

        axs[2].plot(samples_seen(d3_t5), top2(d3_t5), label='P:30% | C:60% | F:10% 5 Tasks', color='g', marker='s')
        axs[2].plot(samples_seen(d3_t10), top2(d3_t10), label='P:30% | C:60% | F:10% 10 Tasks', color='g', marker='o')
        axs[2].plot(samples_seen(d3_t25), top2(d3_t25), label='P:30% | C:60% | F:10% 25 Tasks', color='g', marker='.')

        axs[2].plot(samples_seen(d4_t5), top2(d4_t5), label='P:30% | C:20% | F:50% 5 Tasks', color='m', marker='s')
        axs[2].plot(samples_seen(d4_t10), top2(d4_t10), label='P:30% | C:20% | F:50% 10 Tasks', color='m', marker='o')
        axs[2].plot(samples_seen(d4_t25), top2(d4_t25), label='P:30% | C:20% | F:50% 25 Tasks', color='m', marker='.')

        axs[2].plot(samples_seen(unsorted_t25), top2(unsorted_t25) label='unsorted', color='b')
        axs[2].plot(samples_seen(sorted_t25), top2(sorted_t25) label='sorted', color='r')

        axs[2].legend()
        plt.savefig(plot_directory + 'Group4.png')
        plt.show()
