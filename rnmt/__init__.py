# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, Linear, TransformerEncoder
from fairseq.modules import (
    LayerNorm,
    TransformerDecoderLayer,
    MultiheadAttention,
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('rnmt')
class HybridTransformerRNNModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument(
            "--decoder-reduced-attention-dim",
            type=int,
            metavar="N",
            help="if specified, computes attention with this dimensionality "
            "(instead of using encoder output dims)",
        )
        parser.add_argument(
            "--decoder-lstm-units",
            type=int,
            metavar="N",
            help="num LSTM units for each decoder layer",
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return HybridRNNDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

class HybridRNNDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        self.embed_tokens = embed_tokens

        self.lstm_units = args.decoder_lstm_units
        self.num_layers = args.decoder_layers
        self.initial_input_dim = embed_dim

        self.encoder_output_dim = args.encoder_embed_dim
        if args.decoder_reduced_attention_dim is None:
            self.attention_dim = self.encoder_output_dim
        else:
            self.attention_dim = args.decoder_reduced_attention_dim
        self.input_dim = self.lstm_units + self.attention_dim

        self.num_attention_heads = args.decoder_attention_heads
        self.bottleneck_dim = args.decoder_out_embed_dim


        self.initial_rnn_layer = nn.LSTM(
            input_size=self.initial_input_dim, hidden_size=self.lstm_units
        )
        self.initial_layernorm = LayerNorm(self.lstm_units)

        self.proj_encoder_layer = None
        if self.attention_dim != self.encoder_output_dim:
            self.proj_encoder_layer = Linear(
                self.encoder_output_dim, self.attention_dim
            )

        self.proj_layer = None
        if self.lstm_units != self.attention_dim:
            self.proj_layer = Linear(
                self.lstm_units, self.attention_dim
            )

        self.attention = MultiheadAttention(
            self.attention_dim,
            self.num_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.extra_rnn_layers = nn.ModuleList([])
        self.extra_layernorms = nn.ModuleList([])
        for _ in range(self.num_layers - 1):
            self.extra_rnn_layers.append(
                nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_units)
            )
            self.extra_layernorms.append(
                LayerNorm(self.lstm_units)
            )

        self.bottleneck_layer = None
        if self.bottleneck_dim is not None:
            self.out_embed_dim = self.bottleneck_dim
            self.bottleneck_layer = Linear(
                self.input_dim, self.out_embed_dim
            )
        else:
            self.out_embed_dim = self.input_dim

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.out_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.out_embed_dim ** -0.5)
        else:
            assert self.bottleneck_dim == args.decoder_embed_dim, (self.bottleneck_dim, args.decoder_embed_dim)

    def _unpack_encoder_out(self, encoder_out):
        """ Allow taking encoder_out from different architecture which
        may have different formats.
        """
        # return encoder_out['encoder_out'], encoder_out['encoder_padding_mask']
        return encoder_out.encoder_out, encoder_out.encoder_padding_mask

    def _init_hidden(self, encoder_out, batch_size):
        """ Initialize with latent code if available otherwise zeros."""
        return torch.zeros([1, batch_size, self.lstm_units])

    def _concat_latent_code(self, x, encoder_out):
        """ Concat latent code, if available in encoder_out, which is the
        case in subclass.
        """
        return x

    def _embed_prev_outputs(self, prev_output_tokens, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return x, prev_output_tokens

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
    ):
        x, prev_output_tokens = self._embed_prev_outputs(
            prev_output_tokens=prev_output_tokens, incremental_state=incremental_state
        )
        return self._forward_given_embeddings(
            embed_out=x,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            possible_translation_tokens=possible_translation_tokens,
            timestep=timestep,
        )

    def _forward_given_embeddings(
        self,
        embed_out,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        possible_translation_tokens=None,
        timestep=None,
    ):
        x = embed_out
        (encoder_x, encoder_padding_mask) = self._unpack_encoder_out(encoder_out)
        bsz, seqlen = prev_output_tokens.size()

        state_outputs = []
        if incremental_state is not None:
            prev_states = utils.get_incremental_state(
                self, incremental_state, "cached_state"
            )
            if prev_states is None:
                prev_states = self._init_prev_states(encoder_out)

            # final 2 states of list are projected key and value
            saved_state = {"prev_key": prev_states[-2], "prev_value": prev_states[-1]}
            self.attention._set_input_buffer(incremental_state, saved_state)

        if incremental_state is not None:
            # first num_layers pairs of states are (prev_hidden, prev_cell)
            # for each layer
            h_prev = prev_states[0]
            c_prev = prev_states[1]
        else:
            h_prev = self._init_hidden(encoder_out, bsz).type_as(x)
            c_prev = torch.zeros([1, bsz, self.lstm_units]).type_as(x)

        x = self._concat_latent_code(x, encoder_out)
        x, (h_next, c_next) = self.initial_rnn_layer(x, (h_prev, c_prev))
        x = self.initial_layernorm(x)
        if incremental_state is not None:
            state_outputs.extend([h_next, c_next])

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.proj_encoder_layer is not None:
            encoder_x = self.proj_encoder_layer(encoder_x)

        attention_in = x
        if self.proj_layer is not None:
            attention_in = self.proj_layer(x)

        attention_out, attention_weights = self.attention(
            query=attention_in,
            key=encoder_x,
            value=encoder_x,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=(not self.training),
        )

        for i, layer in enumerate(self.extra_rnn_layers):
            residual = x
            rnn_input = torch.cat([x, attention_out], dim=2)
            rnn_input = self._concat_latent_code(rnn_input, encoder_out)

            if incremental_state is not None:
                # first num_layers pairs of states are (prev_hidden, prev_cell)
                # for each layer
                h_prev = prev_states[2 * i + 2]
                c_prev = prev_states[2 * i + 3]
            else:
                h_prev = self._init_hidden(encoder_out, bsz).type_as(x)
                c_prev = torch.zeros([1, bsz, self.lstm_units]).type_as(x)

            x, (h_next, c_next) = layer(rnn_input, (h_prev, c_prev))
            if incremental_state is not None:
                state_outputs.extend([h_next, c_next])
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
            x = self.extra_layernorms[i](x)

        x = torch.cat([x, attention_out], dim=2)
        x = self._concat_latent_code(x, encoder_out)
        if self.bottleneck_layer is not None:
            x = self.bottleneck_layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.share_input_output_embed:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = F.linear(x, self.embed_out)

        if incremental_state is not None:
            # encoder projections can be reused at each incremental step
            state_outputs.extend([prev_states[-2], prev_states[-1]])
            utils.set_incremental_state(
                self, incremental_state, "cached_state", state_outputs
            )

        return logits, attention_weights

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1024)  # an arbitrary large number

    def _init_prev_states(self, encoder_out):
        """
        Initial (hidden, cell) values for LSTM layers are zero.

        For encoder-decoder attention, key and value are computed once from
        the encoder outputs and stay the same throughout decoding.
        """
        (encoder_x, encoder_padding_mask) = self._unpack_encoder_out(encoder_out)
        batch_size = torch.onnx.operators.shape_as_tensor(encoder_x)[1]

        if self.proj_encoder_layer is not None:
            encoder_x = self.proj_encoder_layer(encoder_x)

        states = []
        for _ in range(self.num_layers):
            hidden = self._init_hidden(encoder_out, batch_size).type_as(encoder_x)
            cell = torch.zeros([1, batch_size, self.lstm_units]).type_as(encoder_x)
            states.extend([hidden, cell])

        # (key, value) for encoder-decoder attention computed from encoder
        # output and remain the same throughout decoding
        key = self.attention.k_proj(encoder_x)
        value = self.attention.v_proj(encoder_x)

        # (key, value) kept in shape (bsz, num_heads, seq_len, head_dim)
        # to avoid repeated transpose operations
        seq_len, batch_size_int, _ = encoder_x.shape
        num_heads = self.attention.num_heads
        head_dim = self.attention.head_dim
        key = (
            key.view(seq_len, batch_size_int * num_heads, head_dim)
            .transpose(0, 1)
            .view(batch_size_int, num_heads, seq_len, head_dim)
        )
        value = (
            value.view(seq_len, batch_size_int * num_heads, head_dim)
            .transpose(0, 1)
            .view(batch_size_int, num_heads, seq_len, head_dim)
        )
        states.extend([key, value])

        return states

    def reorder_incremental_state(self, incremental_state, new_order):
        # parent reorders attention model
        super().reorder_incremental_state(incremental_state, new_order)

        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        # Last 2 elements of prev_states are encoder projections
        # used for ONNX export
        for i, state in enumerate(cached_state[:-2]):
            cached_state[i] = state.index_select(1, new_order)
        for i in [-2, -1]:
            cached_state[i] = cached_state[i].index_select(0, new_order)

        utils.set_incremental_state(
            self, incremental_state, "cached_state", cached_state
        )

@register_model_architecture('rnmt', 'rnmt')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 8)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_reduced_attention_dim = getattr(args, "decoder_reduced_attention_dim", None)
    args.decoder_lstm_units = getattr(args, "decoder_lstm_units", 1024)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1024)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.3)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

@register_model_architecture('rnmt', 'rnmt2')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 8)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_reduced_attention_dim = getattr(args, "decoder_reduced_attention_dim", None)
    args.decoder_lstm_units = getattr(args, "decoder_lstm_units", 1024)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1024)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.3)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
