import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
import argparse
from torch import nn
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

from text.cleaners import vietnamese_cleaner
from text import cleaned_text_to_sequence
from text.symbols import symbols
from text.symbols import _punctuation
import phonemizer

#<yymmdd>_<hhmmss>
string_time = datetime.now().strftime("%y%m%d_%H%M%S")

# en_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
en_phonemizer = None
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text, sid, dictionary, phonemize=True):
    if phonemize:
        clean_text = vietnamese_cleaner(text, en_phonemizer)
    else:
        clean_text = text.lower()
    clean_text = '# ' + clean_text + ' ~'

    for i in range(len(dictionary)):
        # clean_text = clean_text.replace('ɪmpɹuːv ', 'ɪmpɹuː ')
        clean_text = clean_text.replace(dictionary['Original'][i], dictionary['Substitute'][i])
   
    sequence = cleaned_text_to_sequence(clean_text, "Vi")
    
    text_norm = commons.intersperse(sequence, 0)
    text_norm = torch.LongTensor(text_norm)
    # text_norm = torch.LongTensor(sequence)

    return text_norm, clean_text

def synthesize(sentences, speakers, net_g, hps, filename=None, play_audio=False, dict_path=None, print_dur=False, noise_scale_w=1,length_scale=0.8, phonemize=True):
    dictionary = pd.read_csv(dict_path, delimiter='|')
    # print(dictionary)
    
    
    texts_norm = [text_to_sequence(sentences[s], speakers[s], dictionary, phonemize=phonemize)[0] for s in range(len(sentences))]
    clean_texts = [text_to_sequence(sentences[s], speakers[s], dictionary, phonemize=phonemize)[1] for s in range(len(sentences))]
    max_text_len = max([len(x) for x in texts_norm])
    
    text_padded = torch.LongTensor(len(sentences), max_text_len)
    text_lengths = torch.LongTensor(len(sentences))
    
    sid = torch.LongTensor(speakers)
    speaker_text = [s if s != 2 else 3 for s in speakers]
    sid_text = torch.LongTensor(speaker_text)

    text_padded.zero_()
    for i in range(len(sentences)):
        text_padded[i, :len(texts_norm[i])] = texts_norm[i]
        text_lengths[i] = len(texts_norm[i])

    # print("text_padded: ", text_padded)
    with torch.no_grad():
        x_tst = text_padded.cuda()
        x_tst_lengths = text_lengths.cuda()
        
        sid = sid.cuda()
        sid_text = sid_text.cuda()
        start = time.time()
        # print("x_tst shape: ", x_tst.shape)
        # print("x_tst: ", x_tst)
        result = net_g.infer_new(clean_texts, x_tst, x_tst_lengths, sid=sid, sid_text = sid_text, noise_scale=.0, noise_scale_w=noise_scale_w, length_scale=length_scale, print_dur=print_dur)
        end = time.time()
        print('time predict: ', (end-start)*1000)
        
        max_len_dur = max([dur.sum() for dur in result[1]])
        
        for i in range(len(speakers)):
            if speakers[i] == 0: sp = 'vf0'
            elif speakers[i] == 1: sp = 'vm2' 
            elif speakers[i] == 2: sp = 'vm3' 
            elif speakers[i] == 3: sp = 'vf4' 
                
            dur = result[1][i].cpu()
            audio = result[0][i,0].data.cpu().float().numpy()
            
            start_wav = int(audio.shape[0] * dur[0][:50].sum() / max_len_dur.cpu().sum())
            end_wav = int(audio.shape[0] * dur.sum() / max_len_dur.cpu().sum())

            print('dur: ', dur.shape)
            print("start_wav: ", start_wav)
            print("end_wav: ", end_wav)
  
            if filename:
                # write(filename[i] + f'_{sp}.wav', hps.data.sampling_rate, audio[:end_wav])
                write(filename[i], hps.data.sampling_rate, audio[:end_wav])
            end = time.time()
            print(clean_texts[i])
            if play_audio:
                # ipd.display(ipd.Audio(audio[start_wav:end_wav], rate=hps.data.sampling_rate, normalize=True))
                ipd.display(ipd.Audio(audio[:end_wav], rate=hps.data.sampling_rate, normalize=True))

    return result

def plot_dur(result):
    dict = result[2]

    keys = []
    values = []
    
    for v in dict:
        if v[0] == '_':
            continue
        keys.append(v[0])
        values.append(v[1])
    plt.bar(range(len(values)), height=values)
    plt.xticks(range(len(values)), keys)
    plt.axhline(y=2.0, color='r', linestyle='-')
    
    plt.show() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_dataset', type=str, default="./file/file.txt",
                        help='path to sentences file')
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default="./checkpoints/checkpoint.pt",
                        help='path to checkpoint')
    parser.add_argument('-o', '--output_wav_path', type=str, required=True,
                        help='path to save path')
  
    args = parser.parse_args()

    config = args.config
    checkpoint_path = args.checkpoint
    output_wav_path = args.output_wav_path
    test_dataset = args.test_dataset

    print(config)
    print(checkpoint_path)
    print(output_wav_path)
    print(test_dataset)

    hps = utils.get_hparams_from_file(config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)

    
    df = pd.read_csv(test_dataset, delimiter='|', header=None)
    # os.makedirs(os.path.join(output_wav_path, string_time + '_character' + checkpoint_path.split('/')[-1][1:-4]))
    # os.makedirs(os.path.join(output_wav_path))

    sentences = []
    sid = []
    filename = []
    
    for i in tqdm(range(len(df))):
        # for j in range(4):
        #     sentences.append(df[1][i])
        #     sid.append(j)
        #     filename.append(os.path.join(output_wav_path, string_time + '_character' + checkpoint_path.split('/')[-1][1:-4], df[0][i]))
        
        sentences = [df[1][i]]
        sid = [int(df[0][i])]
        
        if df[0][i] == 0: str_id = 'vf0'
        elif df[0][i] == 1: str_id = 'vm2'
        elif df[0][i] == 2: str_id = 'vm3'
        elif df[0][i] == 3: str_id = 'vf4'
            
        filename = [os.path.join(output_wav_path, f's{i}_{str_id}.wav')]
        result = synthesize(
            sentences, 
            sid, 
            net_g, hps, 
            filename=filename,
            dict_path='/data/luonghc/vits_final/zalotts-v2/zalo_tts/vits/dictionary/dictionary.csv')

if __name__ == "__main__":
    main()