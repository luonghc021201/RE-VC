import transformers
import torch
class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=12): 
        super().__init__() 
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("nguyenvulebinh/wav2vec2-large-vi")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer
        
    @torch.no_grad()
    def forward(self, x): 
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]    
        
        return y.permute((0, 2, 1))  