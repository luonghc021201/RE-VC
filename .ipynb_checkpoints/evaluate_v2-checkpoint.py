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

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift_interface import TextToSpeechNLPService
from thrift_interface.ttypes import TNlpParam
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
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

host = '10.40.34.32'
port = '8131'
socket = TSocket.TSocket(host=host, port=port)
transport = TTransport.TFramedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = TextToSpeechNLPService.Client(protocol)

transport.open()

en_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text):
    text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' :' , ':').replace(' ;', ';')
    text = text.replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?').replace(':' , ' :').replace(';', ' ;')
    text = convert_standard_text(text)

    clean_text = vietnamese_cleaner(text, en_phonemizer)
    clean_text = '# ' + clean_text

    sequence = cleaned_text_to_sequence(clean_text, "Vi")
    text_norm = commons.intersperse(sequence, 0)
    text_norm = torch.LongTensor(text_norm)

    return text_norm, clean_text

def synthesize(sentences, speakers, net_g, hps, filename=None, play_audio=False):
    for sid in speakers:
        text_norm, cleaned_text = text_to_sequence(sentences)
    
        stn_tst = text_norm
        with torch.no_grad():
            x_tst = text_norm.unsqueeze(0).cuda()
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            # print(x_tst_lengths.shape)
            sid = torch.LongTensor([sid]).cuda()
            sid_text = torch.LongTensor([3]).cuda()
            sid_dur = torch.LongTensor([3]).cuda()
            result = net_g.infer_new(cleaned_text, x_tst, x_tst_lengths, sid=sid, sid_text=sid_text, noise_scale=.0, noise_scale_w=0.8, length_scale=1)
            audio = result[0][0,0].data.cpu().float().numpy()

        sp = 'vf0'
        if sid == 1: sp = 'vm2' 
        elif sid == 2: sp = 'vm3' 
        elif sid == 3: sp = 'vf4' 
          
        if filename:
          write(filename + f'_{sp}.wav', hps.data.sampling_rate, audio)
        end = time.time()
        
        if play_audio:
          ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

    return result, cleaned_text

def convert_standard_text(sentence):
    new_input = TNlpParam()
    new_input.text = sentence
    
    new_input.max_L = 101000
    new_input.requestId = 5555
    new_input.speakerId = 0
    new_input.extra_options = {
        "mode": "platform"
    }
    speech_output = client.normalizeAndSplit(new_input)
    
    sentence = sentence.split(' ')
    cmu_sentence = speech_output.result[0].text.split(' ')
    temp = []
    for i in range(len(cmu_sentence)):
        if cmu_sentence[i].startswith('@'): 
            temp.append('<ENG>' + sentence[i] + '</ENG>')
        else: 
            temp.append(sentence[i])
    return ' '.join(temp)
  
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
    os.makedirs(os.path.join(output_wav_path, string_time + '_character_no_cs' + checkpoint_path.split('/')[-1][1:-4]))
    for i in tqdm(range(len(df))):
        result, cleaned_text = synthesize(
            df[1][i], 
            [0,1,2,3], 
            net_g, hps, 
            os.path.join(output_wav_path, string_time + '_character_no_cs' + checkpoint_path.split('/')[-1][1:-4], df[0][i]))

if __name__ == "__main__":
    main()