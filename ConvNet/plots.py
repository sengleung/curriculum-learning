import json
import matplotlib.pyplot as plt

def b(dict):
    return dict['Balanced10k']

def lb(dict):
    return dict['LookBack10k']

def lf(dict):
    return dict['LookForward10k']

def s(dict):
    return dict['Sorted10k']

def st(dict):
    return dict['Stationary10k']

def u(dict):
    return dict['Unsorted10k']

def v(dict):
    return dict['validation']

def e(dict):
    return dict['evaluation']

def el(dict):
    return dict['evaluation_loss']

def ea(dict):
    return dict['evaluation_accuracy']

def vl(dict):
    return dict['validation_loss']

def va(dict):
    return dict['validation_accuracy']

def sla(dict):
    return dict['samples_looked_at']

result_files = {
    'Balanced10k' : './results/BalancedFull.json',
    'LookBack10k' : './results/LookBackFull.json',
    'LookForward10k' : './results/LookForwardFull.json',
    'Sorted10k' : './results/SortedFull.json',
    'Stationary10k' : './results/StationaryFull.json',
    'Unsorted10k' : './results/UnsortedFull.json',
}

results = dict()
for key, value in result_files.items():

    with open(value) as f:
        data = json.load(f)
        results[key] = data

#Evaluation Accuracies
unsorted_eval_accuracies = ea(e(u(results)))
sorted_eval_accuracies = ea(e(s(results)))
stationary_eval_accuracies = ea(e(st(results)))
lookback_eval_accuracies = ea(e(lb(results)))
lookforward_eval_accuracies = ea(e(lf(results)))
balanced_eval_accuracies = ea(e(b(results)))

#Evaluation Loss
unsorted_eval_loss = el(e(u(results)))
sorted_eval_loss = el(e(s(results)))
stationary_eval_loss = el(e(st(results)))
lookback_eval_loss = el(e(lb(results)))
lookforward_eval_loss = el(e(lf(results)))
balanced_eval_loss = el(e(b(results)))

unsorted_eval_sla = sla(e(u(results)))
sorted_eval_sla = sla(e(s(results)))
stationary_eval_sla = sla(e(st(results)))
lookback_eval_sla = sla(e(lb(results)))
lookforward_eval_sla = sla(e(lf(results)))
balanced_eval_sla = sla(e(b(results)))

plt.plot(unsorted_eval_sla, unsorted_eval_accuracies)
plt.plot(sorted_eval_sla, sorted_eval_accuracies)
plt.plot(stationary_eval_sla, stationary_eval_accuracies)
plt.plot(lookback_eval_sla, lookback_eval_accuracies)
plt.plot(lookforward_eval_sla, lookforward_eval_accuracies)
plt.plot(balanced_eval_sla, balanced_eval_accuracies)
plt.legend(['Unsorted', 'Sorted', 'Stationary', 'Lookback', 'Lookforward', 'Balanced'])
plt.title("Accuracies")
plt.show()

# plt.plot(unsorted_eval_sla, unsorted_eval_loss)
# plt.plot(sorted_eval_sla, sorted_eval_loss)
# plt.plot(stationary_eval_sla, stationary_eval_loss)
# plt.plot(lookback_eval_sla, lookback_eval_loss)
# plt.plot(lookforward_eval_sla, lookforward_eval_loss)
# plt.plot(balanced_eval_sla, balanced_eval_loss)
# plt.legend(['Unsorted', 'Sorted', 'Stationary', 'Lookback', 'Lookforward', 'Balanced'])
# plt.title("Loss")
# plt.show()
