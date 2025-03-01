import matplotlib.pylab as plt
import IPython.display as ipd

def generate_character_segments(sentence, attn, audio):
    new_text_norm = '0'
    for i in range(len(sentence)):
        new_text_norm += sentence[i] 
        new_text_norm += '0'
    # print(len(new_text_norm))
    # print(new_text_norm)
    dictionary = []
    sentence = sentence.replace('<ENG>', '').replace('</ENG>', '').split(' ')
    list_text_norm = new_text_norm.split(' ')
    for i in range(len(list_text_norm)):
        # print(list_text_norm[i])
        start_index = new_text_norm.index(list_text_norm[i])
        end_index = new_text_norm.index(list_text_norm[i]) + len(list_text_norm[i])-2
    
        start_time = attn[start_index]
        end_time = attn[end_index]
    
        t = dict()
        t['word'] = sentence[i]
        t['start_time'] = int(audio.shape[0]*start_time/attn[-1])
        t['end_time'] = int(audio.shape[0]*end_time/attn[-1])
        dictionary.append(t)
    return dictionary

def plot_dur(result, sentence):
    attn = result[2]
    attn = attn.sum(2).flatten()
    for i in range(1, len(attn)):
        attn[i] += attn[i-1]

    audio = result[0][0,0].data.cpu().float().numpy()
    
    dictionary = generate_character_segments(sentence, attn, audio)
    
    plt.figure(figsize=(30,8))
    plt.plot(audio)
    plt.ylabel("Amplitude")
    for d in dictionary:
        plt.axvspan(d['start_time'],d['end_time'], alpha=0.1, color="red")
        plt.annotate(d['word'], (d['start_time'] ,0.22), color='red', fontsize=12, weight='bold')
    plt.xlabel("Time")
    plt.show()


def cut_audio(result, sentence):
    attn = result[1]
    attn = attn.sum(2).flatten()
    for i in range(1, len(attn)):
        attn[i] += attn[i-1]

    audio = result[0][0,0].data.cpu().float().numpy()
    dictionary = generate_character_segments(sentence, attn, audio)
    
    for d in dictionary:
        print(d['word'])
        ipd.display(ipd.Audio(audio[d['start_time']:d['end_time']], rate=16000, normalize=False))