from matplotlib import pyplot as plt
from cal_e_DINS import *
import pickle
def store_data(var, path):
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(var, file)

def load_data(path):    
    with open(path+'.pkl', 'rb') as file:
        var = pickle.load(file)
    return var

def get_box_plot_data_for_DINS_e_DINS(number_of_docs, dataset, gamma):
    
    plot_data1 = []
    plot_data2 = []
    
    for num in number_of_docs:
        data_for_doc1 = []
        data_for_doc2 = []
        for i in range(0, 25):
            dct_for_dins = {}
            lst = list(dataset.keys())

            #pick random 30 documents from num
            for j in range(30):
                docid = lst[random.randint(1, num) - 1]
                dct_for_dins.update({docid : dataset[docid]})
            
            vanilla_DINS = calculate_vanilla_D_INS(dct_for_dins)
            PF = 1 - calculate_penalty_factor(dct_for_dins)
            e_DINS = vanilla_DINS + (gamma * PF)
            
            data_for_doc1.append(vanilla_DINS)
            data_for_doc2.append(e_DINS)

        plot_data1.append(data_for_doc1)
        plot_data2.append(data_for_doc2)
        
    return plot_data1, plot_data2
    


# Creating plot

def box_plot1(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       

    return bp

# Creating axes instance

fig, ax = plt.subplots()
ax = fig.add_axes([0, 0, 1, 1])
# ax.set_xticklabels(number_of_docs)

number_of_docs = [500, 1000, 1500, 2000, 2500, 3000, 3500]
user_distb = load_data('/home/sourishd/rahul/data_with_distb_v3_using_doc_text_v2/data_with_distb_v3_roberta')
path = '/home/sourishd/rahul/data_with_distb_NAML/data_with_distb_NAML_roberta' #'/home/sourishd/rahul/data_with_distb_v3_using_doc_text/data_with_distb_v3_roberta' #'/home/sourishd/rahul/data_with_distb2'
tmp= load_data(path)
plot_data1, plot_data2 = get_box_plot_data_for_DINS_e_DINS(number_of_docs, tmp, 1)

bp1 = box_plot1(plot_data1, 'red', 'tan')
bp2 = box_plot1(plot_data2, 'blue', 'cyan')
plt.xticks([1, 2, 3, 4, 5, 6, 7], number_of_docs)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['D_INS in range [0, 1]', 'e-D_INS in range [0, 2]'])

plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.xlabel('Number of documents', fontsize = 8)
plt.ylabel('DINS and e-DINS score for PENS dataset (Xiang Ao et.al.: Microsoft Research, 2021)', fontsize = 8)
plt.title("Score on summaries generate using NAML user embedding in PENS model", fontsize = 11)

# show plot
# plt.show()
plt.savefig('NAML.png')