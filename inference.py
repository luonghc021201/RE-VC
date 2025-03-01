import matplotlib.pyplot as plt
import IPython.display as ipd
from wav2vec2 import Wav2vec2
from augmentation.aug import Augment
import os
import IPython.display as ipd
import os
import pandas as pd
from scipy.io.wavfile import write
import time
import librosa
import utils
import numpy as np
from models import SynthesizerTrn
from torch.nn import functional as F
from mel_processing import mel_spectrogram_torch
# from evaluate_v2_test import synthesize, text_to_sequence
from wavlm import WavLM, WavLMConfig
import torch
torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"]="3"

hps2 = utils.get_hparams_from_file("./configs/vits_vc_vctk.json")


aug = Augment(hps2)
checkpoint = torch.load('wavlm/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
cmodel = WavLM(cfg)
cmodel.load_state_dict(checkpoint['model'])
cmodel.eval()
net_g = SynthesizerTrn(
    hps2.data.filter_length // 2 + 1,
    hps2.train.segment_size // hps2.data.hop_length,
    n_speakers=hps2.data.n_speakers,
    **hps2.model)
_ = net_g.eval()
_ = utils.load_checkpoint("ckpt/ckpt.pth", net_g, None)

wav_src, sr = librosa.load('audio_paper/FUS1.wav', sr=16000)
wav_src = wav_src / abs(wav_src).max() * 0.999
wav_torch = torch.from_numpy(wav_src).unsqueeze(0)
x = utils.get_content(cmodel, wav_torch)

wav, sr = librosa.load(f'audio_paper/FUS2.wav', sr=16000)
wav = wav / abs(wav).max() * 0.999

mel = mel_spectrogram_torch(
    torch.from_numpy(wav).unsqueeze(0),
    hps2.data.filter_length, 
    hps2.data.n_mel_channels, 
    hps2.data.sampling_rate,
    hps2.data.hop_length,
    hps2.data.win_length,
    hps2.data.mel_fmin, 
    hps2.data.mel_fmax
)

result = net_g.infer(x, torch.tensor([x.size(2)]), mel, noise_scale=00)[0,0].data.cpu().float().numpy()
print("converted: ")

write(f"audio_paper/FUS1_to_FUS2.wav", 16000, result) 