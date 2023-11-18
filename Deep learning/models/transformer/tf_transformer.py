import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import tensorflow as tf



# https://www.tensorflow.org/text/tutorials/transformer


# input shape: (B,5,2,128,12) output shape: (B,1,2,128,12)
'''
$$
PE_{\left( pos,2i \right)}=\sin \left( pos/10000^{2i/d_{model}} \right) 
\\
PE_{\left( pos,2i+1 \right)}=\cos \left( pos/10000^{2i/d_{model}} \right) 
$$

'''
#length = 5
#depth = 32


# positional_encoding
def positional_encoding(length, depth):
    # length: seq_len
    # depth : d_model
    # if depth % 2 != 0:
    #     raise ValueError("Cannot use sin/cos positional encoding with "
    #                      "odd dim (got dim={:d})".format(depth))
    pos_encoding = np.zeros((length, depth))
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
        # depths for `^{2i/d_{model}}`
    #print(depths) #
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (seq, depth)
    #print(np.sin(angle_rads)) # (5, 16)
    #print(np.cos(angle_rads)) # (5, 16)

    pos_encoding[:,0::2] = np.sin(angle_rads)
    pos_encoding[:,1::2] = np.cos(angle_rads)
    return tf.cast(tf.convert_to_tensor(pos_encoding), dtype=tf.float32)
        # shape:(length,depth) which means for each i-length and every j-depth also adapt pe!
# a = positional_encoding(length, depth)
# print(a)
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_features, seq_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = layers.Dense(d_model) #linear embedding
        self.pos_encoding = positional_encoding(length=seq_len,depth=d_model)
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis,:length,:]
        return x #





#causal mask
def create_causal_mask(seq_length):
    # Create a lower triangular matrix with ones below the diagonal
    mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    # Expand the mask to the batch dimension
    #mask = tf.expand_dims(mask, 0)
    # Duplicate the mask for each example in the batch
    #mask = tf.tile(mask, [batch_size, 1, 1])
    #mask = tf.tile(mask[tf.newaxis,:,:], [batch_size, 1, 1])
    return mask


# Add and normalize
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
# CrossAttention Layer
class CrossAttention(BaseAttention):

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query = x,
            key = context,
            value = context,
            return_attention_scores = True
        )
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x







# Attention: The `query` is what you're trying to find.
            #The `key` is what sort of information the dictionary has
            #The `value` is that information
# Gloabl self-attention
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x




# The causal self-attention layer
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        B, seq_len, _ = x.shape
        attn_output = self.mha(
            query = x,
            value = x,
            key =x,
            attention_mask = create_causal_mask(seq_length=seq_len),
            #=None,
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x




# The feed forward network

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x



# The encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate,
        )
        self.ffn = FeedForward(d_model, dff)
    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


# The encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, num_features,src_seq_len,d_model,num_heads,dff,dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(num_features=num_features,seq_len= src_seq_len,d_model=d_model) #linear embedding
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        # x shape: (batch_size,src_seq_len, num_features)
        x = self.pos_embedding(x) # (batch_size,src_seq_len, d_model)
        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x # shape:(batch_size,src_seq_len, d_model)




# encoderInput = keras.Input(shape=(src_seq_len,num_features))
# encoderOutput = sample_encoder(encoderInput)
# encoder_model = tf.keras.Model(inputs=encoderInput, outputs=encoderOutput)
# print(encoder_model.summary())
# Total params: 43,586,048


# The decoder layer

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention =CausalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate,
        )

        self.cross_attention = CrossAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate,
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x,context=context)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)

        return x


# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, num_features, d_model, num_heads, dff, tgt_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(num_features=num_features,seq_len= tgt_seq_len,d_model=d_model) #linear embedding
        self.dropout = layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff = dff,
                         dropout_rate=dropout_rate,
                         )
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        # x shape: (batch_size,tgt_seq_len, num_features)
        x = self.pos_embedding(x)  # (batch_size,tgt_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # the shape of x is (batch_size, tgt_seq_len, d_model)

        return x



# Transformer
# Put them together

class Transformer(tf.keras.Model):
    def __init__(self,  *, num_layers, num_features, d_model, num_heads, dff, src_seq_len, tgt_seq_len, valid_len,dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               num_features=num_features,
                               src_seq_len=src_seq_len,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers,
                               num_features=num_features,
                               tgt_seq_len=tgt_seq_len,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               dropout_rate=dropout_rate)
        self.final_layer = layers.Dense(num_features)
        self.valid_len = valid_len
    #@tf.function
    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        context, x = inputs
        #context = tf.reshape(context, shape=(context.shape[0], context.shape[1], -1))
        #x = tf.reshape(x, shape=(x.shape[0], x.shape[1], -1))
            # context shape:(batch_size, src_seq_len,2,128,12)
            # x shape: (batch_size, tgt_seq_len,num_features)
        context = self.encoder(context) # (batch_size, src_seq_len,d_model)
        x = self.decoder(x=x, context=context)
        out = self.final_layer(x) # shape: # (batch_size, tgt_seq_len,num_features)
        out = out [:,-self.valid_len:,]
        #out = tf.reshape(out, shape=(out.shape[0], out.shape[1],2,128,12))
        # try:
        #     # Drop the keras mask, so it doesn't scale the losses/metrics.
        #     # b/250038731
        #     del out._keras_mask
        # except AttributeError:
        #     pass
        return out


# transformer_output = transformer((ein, din))
# print(ein.shape) # (batch_size, src_seq_len, num_features)
# print(din.shape) # (batch_size, tgt_seq_len, num_features)
# print(transformer_output.shape)# (batch_size, tgt_seq_len, num_features)
#
# attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
# print(attn_scores.shape)#(batch_size, num_heads, tgt_seq_len, src_seq_len)
# exit()


if __name__ == "__main__":
    obs_window = 5
    pred_window = 1
    transision_window = 2
    batch_size = 32
    Nt = 32
    Nr = 4
    Ns = 12
    num_features = Nt * Nr * 2
    src_seq_len = obs_window
    tgt_seq_len = pred_window + transision_window

    embed_ein = PositionalEmbedding(num_features=256, seq_len=src_seq_len, d_model=512)
    embed_din = PositionalEmbedding(num_features=256, seq_len=tgt_seq_len, d_model=512)
    ein = tf.random.normal(shape=(batch_size, src_seq_len, num_features))
    ein_emb = embed_ein(ein)  # from (batch_size,src_seq_len,num_features) -> (batch_size,src_seq_len,d_model)
    print(ein_emb.shape)  # (batch_size,src_seq_len,d_model)
    din = tf.random.normal(shape=(batch_size, tgt_seq_len, num_features))
    din_emb = embed_din(din)  # from (batch_size,tgt_seq_len,num_features) -> (batch_size,tgt_seq_len,d_model)
    print(din_emb.shape)  # (batch_size,tgt_seq_len,d_model)

    sample_ca = CrossAttention(num_heads=2, key_dim=512)
    print(sample_ca(din_emb, ein_emb).shape)  # (batch_size,tgt_seq_len,d_model) attention not change tensor shape

    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
    print(sample_gsa(ein_emb).shape)  # (batch_size,src_seq_len,d_model)

    sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
    print(sample_csa(ein_emb).shape)  # (batch_size,src_seq_len,d_model)

    out1 = sample_csa(embed_ein(ein[:, :2]))
    out2 = sample_csa(embed_ein(ein))[:, :2]
    print(tf.reduce_max(abs(out1 - out2)).numpy())  # check if accurate mask

    sample_ffn = FeedForward(512, 2048)
    print(sample_ffn(ein_emb).shape)  # (batch_size,src_seq_len,d_model)

    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
    print(sample_encoder_layer(ein_emb).shape)  # (batch_size,src_seq_len# ,d_model)

    sample_encoder = Encoder(num_layers=4, d_model=512, num_features=num_features, src_seq_len=src_seq_len, num_heads=8,
                             dff=2048)

    sample_encoder_output = sample_encoder(ein, training=False)
    print(sample_encoder_output.shape)  # (batch_size,src_seq_len, d_model)

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

    sample_decoder_layer_output = sample_decoder_layer(x=din_emb, context=ein_emb)
    print(sample_decoder_layer_output.shape)  # (batch_size,tgt_seq_len, d_model)

    sample_decoder = Decoder(num_layers=4,
                             num_features=num_features,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             tgt_seq_len=tgt_seq_len)
    decoder_output = sample_decoder(
        x=din,
        context=ein_emb
    )
    print("dcoder shape test")
    print(din.shape)  # (batch_size, tgt_seq_len, num_features)
    print(ein_emb.shape)  # (batch_size, src_seq_len, num_features)
    print(decoder_output.shape)  # (batch_size, tgt_seq_len, num_features)

    print(sample_decoder.last_attn_scores.shape)  # # (batch, heads, target_seq, input_seq)

    num_layers = 4
    d_model = 512
    dff = 2048
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        num_features=num_features,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        dropout_rate=0.1,
        valid_len= pred_window,
    )

    transformer_src_Input = keras.Input(shape=(src_seq_len,num_features), name='source_input')
    transformer_tgt_Input = keras.Input(shape=(tgt_seq_len,num_features), name='target_input')
    transformer_output = transformer((transformer_tgt_Input, transformer_src_Input))

    transformer_model = tf.keras.Model(inputs=[transformer_src_Input, transformer_tgt_Input], outputs=transformer_output)
    print(transformer_model.summary())