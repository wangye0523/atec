

path = "../data/atec_nlp_sim_train.csv"

import jieba
jieba.load_userdict("../data/user_dict")

def create_seg_data():
    f = open(path)
    outf = open("../data/seg_result.txt", 'w')
    for i, line in enumerate(f):
        line = line.strip("\n").split("\t")
        print(line)
        text_a = line[1]
        text_b = line[2]

        words_a = jieba.cut(text_a)
        words_b = jieba.cut(text_b)
        outf.write(" ".join(words_a)+"\n")
        outf.write(" ".join(words_b)+"\n")



if __name__ == '__main__':
    create_seg_data()