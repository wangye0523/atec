

source_path = "../data/atec_nlp_sim_train.csv"
from matplotlib import pyplot as plt

def count():
    f = open(source_path)
    label_count_dict = {}
    length_count_dict = {}
    for i, line in enumerate(f):
        line = line.strip("\n").split("\t")
        label= line[-1]
        text1 = line[1]
        text2 = line[2]
        label_count_dict[label] = label_count_dict.get(label, 0) + 1
        l1 = len(text1)
        l2 = len(text2)
        length_count_dict[l1] = length_count_dict.get(l1, 0) + 1
        length_count_dict[l2] = length_count_dict.get(l2, 0) + 2
    print(label_count_dict)

    print(length_count_dict)
    length_count_dict = sorted(length_count_dict.items(), key= lambda x: x[0])
    leng_list = [x[0] for x in length_count_dict]
    count_list = [x[1] for x in length_count_dict]
    plt.plot(leng_list, count_list)
    plt.show()



count()

