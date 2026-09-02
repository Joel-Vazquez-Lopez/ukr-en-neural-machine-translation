
from __future__ import unicode_literals, print_function, division

"""
we train the model sentence by sentence, i.e., setting the batch_size = 1
"""

"""
IMPORTS AND USEFUL VARIABLES
"""



import argparse
import logging
import random
import time
from io import open
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 60
teacher_forcing_ratio = 0.5

######################################################################

class Vocab:
    
    """ 
    This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

######################################################################


def split_lines(input_file):

    
    logging.info("Reading lines of %s...", input_file)
    
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):

    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """
    creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):

    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the enoder RNN"""

    def __init__(self, input_size, hidden_size):
        # input_size: src_side vocabulary size
        # hidden_size: hidden state dimension
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # TODO 1: Initilize your word embedding, encoder rnn
        
        # we save our data of the embedded vectors: 
        self.emb = nn.Embedding(input_size, hidden_size)
        
        # Then we define GRU / LSTM depending on the model we want to use: 
       
        # I think this is the most clean way to do it 
        self.rnn = nn.GRU(input_size = hidden_size, hidden_size = hidden_size)
        # I think another way would be to use hidden size for both arguments
        # as we are interested on the space that we are created, and for it
        # to be acceptable by GRU 

        # in the case to use the standard RNN 
        # self.rnn = nn.RNN(input_size, hidden_size)
        # self.rnn = nn.LSTM(input_size, hidden_size)
        
        #self.linear = this one is redundant here, so no need for a linear layer for the encoder 

    def forward(self, input, hidden):
        """
        runs the forward pass of the encoder
        returns the output and the hidden state
        """
        # TODO 2: complete the forward computation, given the input and the previous hidden state
        #  return the output and the hidden state
        #  hidden 1 * 1 * 256
        
        # we transform tokens into the embedded values 
        embed_input = self.emb(input)

        """
        We are reshaping below the hidden, but wouldnt that cause the input 
        to be missmatched on size? 
        """
        # we just apply the same structure as we do with hidden size, but now we
        # also add the embedded input 
        shaped_emb = torch.reshape(embed_input, (1,1, self.hidden_size))

        # We store the values learned  by the RNN model and return them, with 
        # what the model had learned and updated. 
        # we will send this values to the decoder 
        output, hidden = self.rnn(shaped_emb, hidden)
        return output, hidden


    def get_initial_hidden_state(self): 
        
        # NOTE: you need to change here if you use LSTM as the rnn unit 
        # Maybe running an if/else would be helpful to get the space properly 
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    
    """
    the class for the decoder 
    """

    def __init__(self, hidden_size, output_size):
        # hidden_size: hidden state dimension
        # output_size: trg_side vocabulary size
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # TODO 3: Initilize your word embedding, decoder rnn, output layer, softmax layer
        
        # we initialize the values to be embedded 
        self.emb = nn.Embedding(output_size, hidden_size)

        """
        We cannot pass the output as it does: 
        TypeError: __init__() got an unexpected keyword argument 'output_size'
        self.rnn = nn.GRU(output_size = hidden_size, hidden_size = hidden_size)
        my code -->
        self.rnn = nn.GRU(output_size = hidden_size, hidden_size = hidden_size)

        obviously GRU requires of Input_size, for that we need I guess, should 
        work just to put hidden_size on both 
        """
        self.rnn = nn.GRU(hidden_size, hidden_size)
        
        # this is our linear layer, that we are initiating 
        # it help us to map the hidden values updated by GRU 
        # into "scores" for each word 
        self.generator = nn.Linear(hidden_size, output_size)
        
        # we also initialize the log softmax, which will be the essential 
        # part to transform the values into probabilities 
        self.log_softmax = nn.LogSoftmax(dim=-1)



    def forward(self, input, hidden):
        """
        runs the forward pass of the decoder
        returns the log_softmax, hidden state
        """

        # TODO 4: complete the forward computation, given the input and the previous hidden state
        # return the following variables
        # log_softmax: the output after applying LogSoftmax function
        # and hidden: hidden states
        # similar to TODO 2, difference: compute the prob over target-side vocabulary given the output
        
        """
        we follow a similar structure as we did with the encoder 
        now the main difference is that we will use the layer and the 
        logsoftmax, here we need it because the decoder needs to predict
        probabilities for the words, in comparison with encoder, that 
        doesnt need such
        """

        embed_input = self.emb(input)

        """
        Issue with dimensionality: 
          raise RuntimeError(
        RuntimeError: For unbatched 2-D input, hx should also be 2-D but got 3-D tensor
        """
        # we fix the issue as we did in our encoder, 
        # matching the same structure as the hidden state 
        shaped_emb = torch.reshape(embed_input, (1,1, self.hidden_size))

        output, hidden = self.rnn(shaped_emb, hidden)
        
        # we only pass the output onto the layer, as for this we
        # do not need the hidden
        result = self.generator(output)
        
        # then we get the probability of the value based on how probable 
        # is the next word 
        log_softmax = self.log_softmax(result)
        
        # we return the result of the probability on the shape expected
        # by the loss
        return log_softmax.view(1,-1), hidden

    def get_initial_hidden_state(self):
       
       # NOTE: you need to change here if you use LSTM as the rnn unit
        
        return torch.zeros(1, 1, self.hidden_size, device=device)
       

######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion):
    # input tensor: seq_length * 1
    encoder_hidden = encoder.get_initial_hidden_state()
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()


    input_length = input_tensor.size(0) # 10
    target_length = target_tensor.size(0)   # 9

    loss = 0
    # encoder-side forward computation
    for ei in range(input_length):
        
        # TODO 5: feed each input to the encoder, and get the output
        # we pass each word in a loop through the encoder so it
        # builds the hidden representations of each sentences
        output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # dev 0.40
    #  set the first input to the decoder is the symbol "SOS"
    decoder_input = torch.tensor([[SOS_index]], device=device)
    
    # TODO 5: initialize the decoder with the last encoder hidden state
    
    """
    As dumb as it sounds we are just passing the hidden values in encodded
    to decoder, I mistakenly did: decoder_hidden = encoder_hidden[:decoder_hidden]
    which doesnt make sense, but I was trying to mimic the TODO. 
    """

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    optimizer.zero_grad()
    # target-side generation
    for di in range(target_length):
        
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        loss += criterion(decoder_output , target_tensor[di].view(-1))

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_index:
                break

    loss.backward()
    optimizer.step()

    loss = loss.item() / target_length  # average of all the steps
    return loss

######################################################################
# SECTION 3: TRANSLATION PROCESS
######################################################################

# This is where the whole work of translation happen 
def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    
    """ 
    runs translation, returns the output
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        max_length = min(max_length, int(input_length * 1.5 + 10))
        encoder_hidden = encoder.get_initial_hidden_state()

        for ei in range(input_length):
            
            output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        #  set the first input to the decoder is the symbol "SOS"
        decoder_input = torch.tensor([[SOS_index]], device=device)
        
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
        
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()  # detach from history as input

        return decoded_words

######################################################################


# Translate (dev/test)set takes in a list of sentences and writes out their translates
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, max_length=max_length)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--status_every', default=500, type=int,
                    help='print how many examples have been learned ')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=None, default=None,
                    help='checkpoint file to start from')
    ap.add_argument('--inference', default=False, action='store_true')
    ap.add_argument('--seed', default=1004, type=int)
    ap.add_argument('--checkpoint_dir', default='results/seq2seq_bpe30000/checkpoints')
    ap.add_argument('--max_length', default=60, type=int,
                help='maximum output length during translation')

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logging.info(str(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint, map_location=device)
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    # TODO 0: initialize the encoder and the decoder here
    # we pass the information required for the encoder / decoder to be able 
    # to fully run the seq2seq: 
    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = DecoderRNN(args.hidden_size, tgt_vocab.n_words).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)  # 8701
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    if args.load_checkpoint is not None and args.inference:
        translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab, max_length=args.max_length)

        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_file, "w", encoding="utf-8") as f:
            for sent in translated_sentences:
                f.write(clean(sent) + "\n")

        references = [[clean(pair[1]).split(), ] for pair in test_pairs[:len(translated_sentences)]]
        candidates = [clean(sent).split() for sent in translated_sentences]
        test_bleu = corpus_bleu(references, candidates) * 100
        logging.info('Test BLEU score: %.2f', test_bleu)
        return

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    # start training
    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0] # 10 * 1
        target_tensor = training_pair[1]    # 9 * 1

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.status_every == 0:
            logging.info('has learnt %d examples', iter_num)
        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = os.path.join(args.checkpoint_dir, 'state_%010d.pt' % iter_num)
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)

            translated_sentences = translate_sentences(
                encoder, decoder, dev_pairs, src_vocab, tgt_vocab,
                max_num_sentences=500,
                max_length=args.max_length
            )


            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates) * 100
            logging.info('Dev BLEU score: %.2f', dev_bleu)

if __name__ == '__main__':
    main()

######################################################################