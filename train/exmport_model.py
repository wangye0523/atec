

import tensorflow as tf
from tensorflow.python.platform import gfile

def save_to_binary(checkpoints_path, out_model_path):
    checkpoint_dir = checkpoints_path

    graph = tf.Graph()
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint_file)

    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def,
            output_node_names=['predict',"prob"]
        )
        with tf.gfile.FastGFile(out_model_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


import pickle
def save_vocab_to_text(path, outf_path):
    outf = open(outf_path, 'w')
    with gfile.Open(path, 'rb') as f:
        data = pickle.loads(f.read())
    print(type(data.vocabulary_._mapping))
    word_map = data.vocabulary_._mapping
    for k, v in word_map.items():
        outf.write("\t".join([k, str(v)])+"\n")



if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="checkpoint dir", default="/home/rocky/dl/atec-nlp-sim/results/mvlstm")
    parser.add_argument("--out_dir", required=True, help="out dir ", default="/home/rocky/dl/atec-nlp-sim/results")
    parser.add_argument("--prefix", required=False, help="preffix for out model", default="model")
    args = parser.parse_args()
    print(args)

    print("export model from: %s, save to :%s"%(args.checkpoint_dir, args.out_dir) )
    save_to_binary(args.checkpoint_dir, os.path.join(args.out_dir, "%s.pb"%(args.prefix)))