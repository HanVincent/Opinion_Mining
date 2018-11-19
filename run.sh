# mkdir models
# mkdir logs
# mkdir dataset
# cd dataset
# wget http://www.cs.cornell.edu/~oirsoy/files/mpqa-drnt/mpqa-opexps.tar.gz

####


# source, hidden_dim, learning_rate, num_layers, bidirectional, dropout, batch_size = sys.argv[1:]

# echo 1
# time python3 Stacked_RNN.py word2vec 100 0.03 3 0 0 80 > logs/0.log
