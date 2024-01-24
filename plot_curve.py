import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import statistics
from matplotlib.cm import get_cmap

color_dict= {
    4: 0,
    6: 1,
    8: 2,
    10: 3,
    16: 4,
    24: 5 ,
}

cmap = get_cmap('tab10')

def load_data(pattern):
    files = glob.glob(pattern)
    err_list = []
    ptr_nb_list = []
    for filename in files:
        with open(filename, 'rb') as f:
            x = pickle.load(f)
            err = (100 - x['best']['acc']) / 100
            err_list.append(err)
            ptr_nb_list.append(x['args'].ptr)
    ptr_nb_list, err_list = zip(*sorted(zip(ptr_nb_list, err_list)))
    return ptr_nb_list, err_list

def average(l):
    return sum(l) / len(l)

# Define nhead and scaleup_dim values
layer_list = [2, 3]
m_list = [4, 6, 8, 10,  16, 24]
res1 = {}
# Load data for each set of files and plot
plt.figure(figsize=(10, 6))

for num_layers in layer_list:
    for m in m_list:
        # print(f"num_layers = {num_layers}, m = {m}")
        num_features = m 

        layer_nb = 12 
        nhead = 4
        scaleup_dim = nhead * m
        dim_feedforward = 4 * scaleup_dim
        list_of_dicts = []
        try:
            for seed_init in [1, 2, 3, 4]:
                file_path = f"/home/guazhang/MasterThesisEPFL/pickles/num_features_{num_features}_num_layers_{num_layers}_ptr_*_net_layers_{layer_nb}_nhead_{nhead}_dim_feedforward_{dim_feedforward}_scaleup_dim_{scaleup_dim}_seed_init_{seed_init}.pkl"
                ptr_nb_list, err_list = load_data(file_path)
                zip_list = dict(zip(ptr_nb_list, err_list))

                list_of_dicts.append(zip_list)

            # print(list_of_dicts)
            collected = {}
            for d in list_of_dicts:
                for key in d:
                    if key in collected:
                        collected[key].append(d[key])
                    else:
                        collected[key] = [d[key]]


            for key in collected:
                collected[key] = average(collected[key])
            ptr_nb_list, err_list = zip(*sorted(collected.items()))
            # print(f'v={m} L={num_layers}', ptr_nb_list, err_list)
            # sort_indices = np.argsort(err_list)
            ptr_nb_list = ptr_nb_list[::-1]
            err_list = err_list[::-1]
            res1[f'v={m} L={num_layers}'] = np.interp(0.1,np.array(err_list), np.array(ptr_nb_list), )
            if num_layers == 2:
                plt.plot(ptr_nb_list, err_list, linestyle='--', marker='o', label=f'v={m} L={num_layers}', markersize=10, color=cmap(color_dict[m]))
            else:
                plt.plot(ptr_nb_list, err_list, linestyle='--', marker='x', label=f'v={m} L={num_layers}', markersize=10, color = cmap(color_dict[m]))
        except:
            # print(f"num_features_{num_features}_num_layers_{num_layers}_ptr_*_net_layers_{layer_nb}_nhead_{nhead}_dim_feedforward_{dim_feedforward}_scaleup_dim_{scaleup_dim}_seed_init_{seed_init}.pkl not found")
            pass
print(f"res1 = {res1}")


# m_list = [6, 7, 8]
# res2 = []
# for m in m_list:
#     num_features = m 
#     num_layers = 2

#     layer_nb = 12 
#     nhead = 4
#     scaleup_dim = nhead * m
#     dim_feedforward = 4 * scaleup_dim
#     file_path = f"/home/guazhang/MasterThesisEPFL/pickles/num_features_{num_features}_num_layers_3_ptr_*_net_layers_{layer_nb}_nhead_{nhead}_dim_feedforward_{dim_feedforward}_scaleup_dim_{scaleup_dim}.pkl"
#     ptr_nb_list, err_list = load_data(file_path)

#     sort_indices = np.argsort(err_list)
#     res2.append(np.interp(0.1,np.array(err_list)[sort_indices], np.array(ptr_nb_list)[sort_indices], ))

#     plt.plot(ptr_nb_list, err_list, linestyle='--', marker='x', label=f'v={m} L=3', markersize=10)

# print(f"res2 = {res2}")


plt.xlabel('P=nb of training points', fontsize=13)
plt.ylabel('Test error', fontsize=13)
plt.title('Test error v.s. number of training points', fontsize=16)
# plt.grid(True)
plt.xscale('log')

# Format major tick labels in scientific notation
def scientific_formatter(value, pos):
    if value < 1:
        return f"{value:.1e}"
    else:
        return fr"$10^{int(np.log10(value))}$"

plt.gca().xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
# Show minor ticks
plt.minorticks_on()
# Set the number of minor tick divisions (optional)
plt.grid(b=True, which='minor', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('err_ptr_final.pdf')
plt.show()