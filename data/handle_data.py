#/usr/bin/env python
#-*-conding:utf-8-*-


source_data = "atec_nlp_sim_train.csv"

def create_all_sent():
    f = open(source_data)
    outf = open("all_sent.txt", 'w')
    for _, line in enumerate(f):
        line = line.strip("\n").strip()
        line = line.split("\t")
        text_a = line[1]
        text_b = line[2]
        outf.write(text_a+"\n")
        outf.write(text_b+"\n")


if __name__ == '__main__':
    create_all_sent()
