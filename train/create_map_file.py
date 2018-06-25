


vec_path = "../data/vectors.txt"
map_path = "../data/word_dict.txt"


def create_map_file():
    f = open(vec_path)
    outf = open(map_path, 'w')
    word_map = {}
    print(f.readline())
    for i, line in enumerate(f):
        word = line.split()[0]
        if word not in word_map:
            word_map[word] = len(word_map)
    for k, v in word_map.items():
        outf.write(k + " "+ str(v)+"\n")



if __name__ == '__main__':
    create_map_file()
