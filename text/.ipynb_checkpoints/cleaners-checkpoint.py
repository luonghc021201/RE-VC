""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
import time
from unidecode import unidecode
from phonemizer import phonemize
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift_interface import TextToSpeechNLPService
from thrift_interface.ttypes import TNlpParam

host = '10.40.34.32'
port = '8131'
socket = TSocket.TSocket(host=host, port=port)
transport = TTransport.TFramedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = TextToSpeechNLPService.Client(protocol)

transport.open()

def convert_standard_text(sentence):
    sentence = sentence.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' :' , ':').replace(' ;', ';')
    sentence = sentence.replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?').replace(':' , ' :').replace(';', ' ;')
    
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
    # print(cmu_sentence)
    temp = []
    for i in range(len(cmu_sentence)):
        if cmu_sentence[i].startswith('@'): 
            temp.append('<ENG>' + sentence[i] + '</ENG>')
        else: 
            temp.append(sentence[i])
    return ' '.join(temp)
  
# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

def vietnamese_cleaner(text, en_phonemizer):
  text = collapse_whitespace(text)
  text = convert_standard_text(text).replace('</ENG></ENG>', '</ENG>').replace('<ENG><ENG>', '<ENG>')
  text = lowercase(text)

  if '<eng>' not in text:
    return text
  else:
    temp = text.split(' ')
    result = []
    for t in temp:
      if '<eng>' not in t:
        result.append(t)
      else:
        # print(t[5:-6])
        start = time.time()
        phonemes = en_phonemizer.phonemize([t[5:-6]], strip=True, njobs=1)[0]
        end = time.time()
        print(t[5:-6] + ': ', end-start)
        result.append('@' + phonemes)
    phonemes = ' '.join(result) 
  return phonemes
