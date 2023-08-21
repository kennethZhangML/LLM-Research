import numpy as np 
import torch 
import torch.nn as nn 

# Citation:
# Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). 
# Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.

# Word2Vec predicts surrounding words from target word
# maximize probability of predicting words given a center word
class Word2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.WT = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        x = self.W(X)
        x = self.WT(x)
        return x 
    
# Random batch sampling from the skip grams 
# we sample a random skip gram from a random index 
# and we retrun it is a diagonalizable matrix with skip_gram[i] on the diagonals
def random_batch(skip_grams, batch_size, vocab_size):
    random_input = []
    random_target = []

    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace = False)

    for i in random_index:
        input_temp = np.eye(vocab_size)[skip_grams[i][0]]
        target_temp = skip_grams[i][1]

        random_input.append(input_temp)
        random_target.append(target_temp)
    return random_input, random_target

# Skips grams Algorithm: appending the word as the target, and the context as the input
# context representing word before and word after the word, and the word in the middle as the target
def skip_grams(word_sequence: list, word_dict : dict):
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        word = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

        for w in context:
            skip_grams.append([word, w])
    return skip_grams

if __name__ == "__main__":
    epochs = 5000
    batch_size = 2
    embedding_size = 2

    sentences = ["tiger monkey animal", "beef pork meat", "black yellow color", "crime black bad"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w:i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    skip_grams = skip_grams(word_sequence, word_dict)

    model = Word2Vec(voc_size, embedding_size, voc_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    user_exec = input('Start Training? (y/n): ')

    if user_exec == 'y' or user_exec == 'Y':
        for epoch in range(epochs):
            input_batch, target_batch = random_batch(skip_grams, batch_size, voc_size)
            input_batch = torch.Tensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            output = model(input_batch)
            loss = criterion(output, target_batch)

            if (epoch + 1) % 1000 == 0:
                print('Epoch: ', '%40d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
            
            loss.backward()
            optimizer.step()

        else:
            user_exec = input('PLease select y/n: ')
    
    predict = model(input_batch).data.max(1, keepdim = True)
    print(input_batch, '->', model(input_batch).data.max(1, keepdim = True))





    