import librosa
import librosa.filters
import numpy as np 
from scipy import signal
# import tensorflow as tf 
from scipy.io import wavfile

_min_samples = 3000

def trim_silence(wav, n_fft, sample_rate, hop_size, frame_shift_ms, threshold_db=25):
  '''Trims silence from the ends of the wav'''
  splits = librosa.effects.split(wav, top_db=threshold_db, frame_length=n_fft, hop_length=get_hop_size(sample_rate, hop_size, frame_shift_ms))
  return wav[_find_start(splits):]


def _find_start(splits):
  for split_start, split_end in splits:
    return max(0, split_start - _min_samples)
  return 0


def _find_end(splits, num_samples):
  for split_start, split_end in reversed(splits):
    if split_end - split_start > _min_samples:
      return min(num_samples, split_end + _min_samples)
  return num_samples

def get_hop_size(sample_rate, hop_size, frame_shift_ms):
	if hop_size is None:
		assert frame_shift_ms is not None
		hop_size = int(frame_shift_ms / 1000 * sample_rate)
	return hop_size