from __future__ import unicode_literals, print_function, division

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 60
teacher_forcing_ratio = 0.5


class Vocab:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            if word:
                self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def split_lines(input_file):
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding="utf-8").read().strip().split("\n")
    pairs = [l.split("|||") for l in lines]
    pairs = [p for p in pairs if len(p) == 2]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info("%s (src) vocab size: %s", src_vocab.lang_code, src_vocab.n_words)
    logging.info("%s (tgt) vocab size: %s", tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


def tensor_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.split():
        if word in vocab.word2index:
            indexes.append(vocab.word2index[word])
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_token, hidden):
        embedded = self.dropout(self.emb(input_token)).view(1, 1, self.hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    """
    Luong-style dot-product attention decoder.
    The decoder attends over all encoder outputs at every target step.
    """

    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.emb = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(hidden_size, hidden_size)

        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.generator = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.dropout(self.emb(input_token)).view(1, 1, self.hidden_size)

        rnn_output, hidden = self.rnn(embedded, hidden)
        # rnn_output: 1 x 1 x hidden
        # encoder_outputs: input_len x hidden

        decoder_state = rnn_output.squeeze(0)  # 1 x hidden

        # Dot-product attention scores over encoder outputs.
        # scores: 1 x input_len
        scores = torch.mm(decoder_state, encoder_outputs.transpose(0, 1))
        attn_weights = F.softmax(scores, dim=1)

        # context: 1 x hidden
        context = torch.mm(attn_weights, encoder_outputs)

        combined = torch.cat((decoder_state, context), dim=1)
        combined = torch.tanh(self.attn_combine(combined))

        logits = self.generator(combined)
        log_probs = self.log_softmax(logits)

        return log_probs, hidden, attn_weights


def encode_sentence(encoder, input_tensor):
    encoder_hidden = encoder.get_initial_hidden_state()
    input_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    return encoder_outputs, encoder_hidden


def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()

    target_length = target_tensor.size(0)

    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encode_sentence(encoder, input_tensor)

    decoder_input = torch.tensor([[SOS_index]], device=device)
    decoder_hidden = encoder_hidden

    loss = 0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):
        decoder_output, decoder_hidden, _ = decoder(
            decoder_input,
            decoder_hidden,
            encoder_outputs
        )

        loss += criterion(decoder_output, target_tensor[di].view(-1))

        if use_teacher_forcing:
            decoder_input = target_tensor[di]
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()
            if decoder_input.item() == EOS_index:
                break

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=5.0
    )

    optimizer.step()

    return loss.item() / max(1, target_length)


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size(0)
        max_length = min(max_length, int(input_length * 1.5 + 10))

        encoder_outputs, encoder_hidden = encode_sentence(encoder, input_tensor)

        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

            topv, topi = decoder_output.topk(1)

            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word.get(topi.item(), "<UNK>"))

            decoder_input = topi.detach()

        return decoded_words


def beam_translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH, beam_size=5):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size(0)
        max_length = min(max_length, int(input_length * 1.5 + 10))

        encoder_outputs, encoder_hidden = encode_sentence(encoder, input_tensor)

        start_input = torch.tensor([[SOS_index]], device=device)
        beams = [(0.0, [], start_input, encoder_hidden, False)]

        for _ in range(max_length):
            new_beams = []

            for score, token_ids, decoder_input, decoder_hidden, finished in beams:
                if finished:
                    new_beams.append((score, token_ids, decoder_input, decoder_hidden, finished))
                    continue

                decoder_output, next_hidden, _ = decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs
                )

                log_probs, top_indices = decoder_output.topk(beam_size)

                for log_prob, top_idx in zip(log_probs[0], top_indices[0]):
                    token_id = int(top_idx.item())
                    new_score = score + float(log_prob.item())
                    new_token_ids = token_ids + [token_id]
                    new_input = torch.tensor([[token_id]], device=device)
                    new_finished = token_id == EOS_index
                    new_beams.append((new_score, new_token_ids, new_input, next_hidden, new_finished))

            def rank_key(beam):
                score, token_ids, _, _, _ = beam
                length = max(1, len(token_ids))
                return score / (length ** 0.7)

            beams = sorted(new_beams, key=rank_key, reverse=True)[:beam_size]

            if all(b[-1] for b in beams):
                break

        best = sorted(
            beams,
            key=lambda b: b[0] / (max(1, len(b[1])) ** 0.7),
            reverse=True
        )[0]

        decoded_words = []
        for token_id in best[1]:
            if token_id == EOS_index:
                decoded_words.append(EOS_token)
                break
            decoded_words.append(tgt_vocab.index2word.get(token_id, "<UNK>"))

        return decoded_words


def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab,
                        max_num_sentences=None, max_length=MAX_LENGTH, beam_size=1):
    output_sentences = []

    subset = pairs[:max_num_sentences] if max_num_sentences else pairs

    for pair in subset:
        if beam_size and beam_size > 1:
            output_words = beam_translate(
                encoder, decoder, pair[0], src_vocab, tgt_vocab,
                max_length=max_length,
                beam_size=beam_size
            )
        else:
            output_words = translate(
                encoder, decoder, pair[0], src_vocab, tgt_vocab,
                max_length=max_length
            )

        output_sentence = " ".join(output_words)
        output_sentences.append(output_sentence)

    return output_sentences


def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for _ in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def clean(strx):
    return " ".join(strx.replace("@@ ", "").replace(EOS_token, "").strip().split())


def save_checkpoint(path, iter_num, encoder, decoder, optimizer, src_vocab, tgt_vocab, best_dev_bleu):
    state = {
        "iter_num": iter_num,
        "enc_state": encoder.state_dict(),
        "dec_state": decoder.state_dict(),
        "opt_state": optimizer.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "best_dev_bleu": best_dev_bleu,
    }
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_size", default=256, type=int)
    ap.add_argument("--n_iters", default=100000, type=int)
    ap.add_argument("--print_every", default=5000, type=int)
    ap.add_argument("--status_every", default=500, type=int)
    ap.add_argument("--checkpoint_every", default=10000, type=int)
    ap.add_argument("--initial_learning_rate", default=0.001, type=float)
    ap.add_argument("--src_lang", default="uk")
    ap.add_argument("--tgt_lang", default="en")
    ap.add_argument("--train_file", default="data/train.bpe")
    ap.add_argument("--dev_file", default="data/valid.bpe")
    ap.add_argument("--test_file", default="data/test.bpe")
    ap.add_argument("--out_file", default="out.txt")
    ap.add_argument("--load_checkpoint", default=None)
    ap.add_argument("--inference", default=False, action="store_true")
    ap.add_argument("--seed", default=1004, type=int)
    ap.add_argument("--checkpoint_dir", default="results/seq2seq_attention/checkpoints")
    ap.add_argument("--max_length", default=60, type=int)
    ap.add_argument("--beam_size", default=1, type=int)
    ap.add_argument("--dev_eval_size", default=500, type=int)
    ap.add_argument("--best_checkpoint_name", default="best_model_attention.pt")
    ap.add_argument("--dropout", default=0.1, type=float)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.info(str(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint: %s", args.load_checkpoint)
        state = torch.load(args.load_checkpoint, map_location=device)
        iter_num = state["iter_num"]
        src_vocab = state["src_vocab"]
        tgt_vocab = state["tgt_vocab"]
        best_dev_bleu = state.get("best_dev_bleu", -1.0)
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang, args.train_file)
        best_dev_bleu = -1.0

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size, dropout=args.dropout).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout=args.dropout).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    if args.load_checkpoint is not None:
        encoder.load_state_dict(state["enc_state"])
        decoder.load_state_dict(state["dec_state"])
        optimizer.load_state_dict(state["opt_state"])

    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    if args.inference:
        translated_sentences = translate_sentences(
            encoder, decoder, test_pairs, src_vocab, tgt_vocab,
            max_length=args.max_length,
            beam_size=args.beam_size
        )

        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_file, "w", encoding="utf-8") as f:
            for sent in translated_sentences:
                f.write(clean(sent) + "\n")

        references = [[clean(pair[1]).split()] for pair in test_pairs[:len(translated_sentences)]]
        candidates = [clean(sent).split() for sent in translated_sentences]
        test_bleu = corpus_bleu(references, candidates) * 100
        logging.info("Test BLEU score: %.2f", test_bleu)
        return

    start = time.time()
    print_loss_total = 0

    logging.info("Starting training from iteration %d to %d", iter_num, args.n_iters)

    while iter_num < args.n_iters:
        iter_num += 1

        pair = random.choice(train_pairs)
        input_tensor, target_tensor = tensors_from_pair(src_vocab, tgt_vocab, pair)

        loss = train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.status_every == 0:
            logging.info("has learnt %d examples", iter_num)

        if iter_num % args.checkpoint_every == 0:
            filename = os.path.join(args.checkpoint_dir, "state_%010d.pt" % iter_num)
            save_checkpoint(filename, iter_num, encoder, decoder, optimizer, src_vocab, tgt_vocab, best_dev_bleu)
            logging.debug("wrote checkpoint to %s", filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0

            logging.info(
                "time since start:%s (iter:%d iter/n_iters:%.2f%%) loss_avg:%.4f",
                time.time() - start,
                iter_num,
                iter_num / args.n_iters * 100,
                print_loss_avg
            )

            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)

            translated_sentences = translate_sentences(
                encoder, decoder, dev_pairs, src_vocab, tgt_vocab,
                max_num_sentences=args.dev_eval_size,
                max_length=args.max_length,
                beam_size=args.beam_size
            )

            references = [[clean(pair[1]).split()] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates) * 100
            logging.info("Dev BLEU score: %.2f", dev_bleu)

            if dev_bleu > best_dev_bleu:
                best_dev_bleu = dev_bleu
                best_path = os.path.join(args.checkpoint_dir, args.best_checkpoint_name)
                save_checkpoint(best_path, iter_num, encoder, decoder, optimizer, src_vocab, tgt_vocab, best_dev_bleu)
                logging.info("New best dev BLEU %.2f; saved checkpoint to %s", best_dev_bleu, best_path)


if __name__ == "__main__":
    main()
