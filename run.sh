# source, hidden_dim, learning_rate, num_layers, bidirectional, dropout, batch_size = sys.argv[1:]

# echo 1
# time python3 Stacked_RNN.py word2vec 100 0.03 3 0 0 80 > logs/0.log

# echo 2
# time python3 Stacked_RNN.py word2vec 100 0.03 3 1 0 80 > logs/1.log

# echo 3
# time python3 Stacked_RNN.py glove 100 0.03 3 0 0 80 > logs/2_.log

# echo 4
# time python3 Stacked_RNN.py glove 100 0.03 3 1 0 80 > logs/3_.log

echo 5
time python3 Stacked_RNN.py word2vec 100 0.03 3 1 0 40 > logs/4_.log

echo 6
time python3 Stacked_RNN.py word2vec 100 0.03 3 1 0 20 > logs/5_.log

echo 7
time python3 Stacked_RNN.py word2vec 100 0.03 3 1 0.3 80 > logs/6_.log

echo 8
time python3 Stacked_RNN.py word2vec 100 0.03 3 1 0.5 80 > logs/7_.log