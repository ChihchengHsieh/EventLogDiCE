import numpy as np
import tensorflow as tf
from typing import List


class Constants():
    PAD_VOCAB = '<PAD>'
    UNK_VOCAB = '<UNK>'
    SOS_VOCAB = '<SOS>'
    EOS_VOCAB = '<EOS>'


class VocabDict:
    def __init__(self, vocabs) -> None:
        self.vocabs: List[int] = vocabs

    def index_to_vocab(self, index: int) -> str:
        return self.vocabs[index]

    def vocab_to_index(self, vocab: str) -> int:
        return self.vocabs.index(vocab)

    def list_of_index_to_vocab(self, list_of_index: List[int]):
        return [self.index_to_vocab(i) for i in list_of_index]

    def list_of_vocab_to_index(self, list_of_vocab: List[str]):
        return [self.vocab_to_index(v) for v in list_of_vocab]

    def list_of_vocab_to_index_2d(self, list_of_vocab_2d):
        return [self.list_of_vocab_to_index(l) for l in list_of_vocab_2d]

    def list_of_index_to_vocab_2d(self, list_of_index_2d):
        return [self.list_of_index_to_vocab(l) for l in list_of_index_2d]

    def vocab_size(self) -> int:
        '''
        Include <START>, <END> and <PAD> tokens. So, if you the actual number of activities,
        you have to minus 3.
        '''
        return len(self.vocabs)

    def padding_index(self):
        return self.vocab_to_index(Constants.PAD_VOCAB)

    def tags_vocab(self):
        return [Constants.PAD_VOCAB, Constants.EOS_VOCAB, Constants.SOS_VOCAB]

    def tags_idx(self):
        return self.list_of_vocab_to_index(self.tags_vocab())

    def tranform_to_input_data_from_seq_idx_with_caseid(self, seq_list: List[List[int]], caseids: List[str] = None):
        '''
        Calculate the lengths for reach trace, so we can use padding.
        '''
        seq_lens = np.array([len(s)for s in seq_list])
        sorted_len_index = np.flip(np.argsort(seq_lens))
        sorted_seq_lens = [seq_lens[idx] for idx in sorted_len_index]
        sorted_seq_list = [tf.constant(seq_list[idx])
                           for idx in sorted_len_index]

        if (caseids):
            sorted_caseids = [caseids[i] for i in sorted_len_index]
        else:
            sorted_caseids = None

        return sorted_caseids, tf.constant(tf.keras.preprocessing.sequence.pad_sequences(sorted_seq_list, padding='post', value=0)), tf.constant(sorted_seq_lens)

    def __len__(self):
        return self.vocab_size()

    def sos_idx(self):
        return self.vocab_to_index(self.sos_vocab())

    def eos_idx(self):
        return self.vocab_to_index(self.eos_vocab())

    def pad_idx(self):
        return self.vocab_to_index(self.pad_vocab())

    def pad_vocab(self):
        return Constants.PAD_VOCAB

    def eos_vocab(self):
        return Constants.EOS_VOCAB

    def sos_vocab(self):
        return Constants.SOS_VOCAB
