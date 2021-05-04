import torch
from torch import nn

import os
from os import path

import pickle

from constants import device, source_vocab_length, target_vocab_length

english_lang_pkl = open('./saved/english_lang.pkl', 'rb')
english_lang = pickle.load(english_lang_pkl)

tamil_lang_pkl = open('./saved/tamil_lang.pkl', 'rb')
tamil_lang = pickle.load(tamil_lang_pkl)


class MyTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu",source_vocab_length: int = 60000,target_vocab_length: int = 60000) -> None:
        super(MyTransformer, self).__init__()

        #embeddings
        self.source_embedding = nn.Embedding(source_vocab_length, d_model)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model)
        
        #initialize positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        #initialize the encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        #layer normalization
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        #initialize the decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        #layer normalization
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        #linear layer
        self.out = nn.Linear(512, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask= None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask= None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        #convert source language into embedding
        src = self.source_embedding(src)
        #add the positional encoding 
        src = self.pos_encoder(src)
        #pass it to the encoder
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #convert target langauge into embedding
        tgt = self.target_embedding(tgt)
        #add the positional encoding 
        tgt = self.pos_encoder(tgt)
        #pass the target embedding and encoder representation to decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        #output layer
        output = self.out(output)
        return output

model = MyTransformer(
    source_vocab_length=source_vocab_length,
    target_vocab_length=target_vocab_length
).to(device)

saved_weights = torch.load(
    path.join(os.getcwd(), './saved/transformer_model.h5')
)

model.load_state_dict(saved_weights)

model.eval()

