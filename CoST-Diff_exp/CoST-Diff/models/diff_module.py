# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from copy import deepcopy
# import copy
# from models.cond_film import CondFiLMStatic


# def Conv1d_with_init(in_channels, out_channels, kernel_size, padding=0):
#     layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
#     nn.init.kaiming_normal_(layer.weight)
#     return layer

# class DiffusionEmbedding(nn.Module):
#     def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
#         super().__init__()
#         if projection_dim is None:
#             projection_dim = embedding_dim
#         self.register_buffer(
#             "embedding",
#             self._build_embedding(num_steps, embedding_dim / 2),
#             persistent=False,
#         )
#         self.projection1 = nn.Linear(embedding_dim, projection_dim)
#         self.projection2 = nn.Linear(projection_dim, projection_dim)

#     def forward(self, diffusion_step):
#         x = self.embedding[diffusion_step]
#         x = self.projection1(x)
#         x = F.silu(x)
#         x = self.projection2(x)
#         x = F.silu(x)
#         return x

#     def _build_embedding(self, num_steps, dim=64):
#         steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
#         frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
#         table = steps * frequencies  # (T,dim)
#         table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
#         return table

# class diff_CSDI(nn.Module):
#     def __init__(self, channels, inputdim, num_steps, embedding_dim, rdcl, args):
#         super().__init__()
#         self.channels = channels
#         self.pre_dim = args.pre_trained_dim
#         self.embedding_hour_dim = args.embedding_hour_dim # 时间 encoder 的维度
        
#         self.rdcl = rdcl
#         self.diffusion_embedding = DiffusionEmbedding(
#             num_steps,
#             embedding_dim,
#         )

#         self.input_projection = Conv1d_with_init(self.pre_dim + self.embedding_hour_dim, self.channels, 1)
#         self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
#         self.output_projection2 = Conv1d_with_init(self.channels, self.channels, 1)
#         self.output_layer = nn.Linear(self.channels, self.pre_dim + self.embedding_hour_dim)
        
#         nn.init.zeros_(self.output_projection2.weight)

#         self.residual_layers = nn.ModuleList(
#             [
#                 Residual_block(
#                     res_channels=self.channels, skip_channels=self.channels, 
#                     dilation=2 ** (n % 4),
#                     diffusion_step_embed_dim_out=embedding_dim,
#                 )
#                 for n in range(self.rdcl)
#             ]
#         )
#         self.rnn = nn.GRU(inputdim, inputdim)
#         self.bn = nn.BatchNorm1d(inputdim)
#         self.out_bn = nn.BatchNorm1d(inputdim)
#         self.layernorm = nn.LayerNorm(inputdim)
        
#         # ★ 新增：静态条件 FiLM
#         self.cond_film   = CondFiLMStatic(H=self.channels)
        
#     def forward(self, x, diffusion_step,
#             cond_static: torch.Tensor = None,
#             sc_gamma: torch.Tensor = None, sc_beta: torch.Tensor = None):
#         """
#         x: [B,T,F]  (F = pre_dim + embedding_hour_dim)
#         diffusion_step: [B,1] (long)
#         cond_static: [B,T,H], cond_mask: [B,T,1]
#         sc_gamma/sc_beta: [B,T,H]，H需等于self.channels
#         """

#         B, T, C = x.shape
#         x = x.permute(0, 2, 1)                       # [B,F,T]
#         x = self.input_projection(x)                  # [B,C,T]
#         x = F.relu(x).permute(0, 2, 1)                # [B,T,C]

#         # 静态 FiLM（整段）
#         if cond_static is not None:
#             gamma, beta, gain = self.cond_film(cond_static)  # [B,T,C], [B,T,C], scalar/[B,1,1]
#             x = x + gain * (x * gamma + beta)

#         diffusion_emb = self.diffusion_embedding(diffusion_step)        # [B,1,C]

#         skip = []
#         for layer in self.residual_layers:
#             x, skip_connection = layer(x, diffusion_emb)
#             # 自条件 FiLM（时间通道来自上一轮x0_hat_time）
#             if sc_gamma is not None:
#                 x = x * (1.0 + sc_gamma) + sc_beta
#             skip.append(skip_connection)

#         x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
#         x = self.output_layer(x)                                        # [B,T,F]
#         return x


        
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         super(Conv, self).__init__()
#         self.padding = dilation * (kernel_size - 1) // 2
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
#         self.conv = nn.utils.weight_norm(self.conv)
#         nn.init.kaiming_normal_(self.conv.weight)

#     def forward(self, x):
#         out = self.conv(x)
#         return out

# class Residual_block(nn.Module):
#     def __init__(self, res_channels, skip_channels, dilation, 
#                  diffusion_step_embed_dim_out):
#         super(Residual_block, self).__init__()
#         self.res_channels = res_channels

#         # the layer-specific fc for diffusion step embedding
#         self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

#         # dilated conv layer
#         self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

#         # add mel spectrogram upsampler and conditioner conv1x1 layer
#         self.upsample_conv2d = torch.nn.ModuleList()
#         for s in [16, 16]:
#             conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
#             conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
#             torch.nn.init.kaiming_normal_(conv_trans2d.weight)
#             self.upsample_conv2d.append(conv_trans2d)
#         self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

#         # residual conv1x1 layer, connect to next residual layer
#         self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
#         self.res_conv = nn.utils.weight_norm(self.res_conv)
#         nn.init.kaiming_normal_(self.res_conv.weight)

#         # skip conv1x1 layer, add to all skip outputs through skip connections
#         self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
#         self.skip_conv = nn.utils.weight_norm(self.skip_conv)
#         nn.init.kaiming_normal_(self.skip_conv.weight)

#     def forward(self, x, diffusion_step_embed):
#         # x, mel_spec, diffusion_step_embed = input_data
#         h = x

#         # add in diffusion step embedding
#         part_t = self.fc_t(diffusion_step_embed)
#         h = part_t + h

#         h = h.permute(0, 2, 1)
#         B, C, L = h.shape
#         assert C == self.res_channels
        
#         # dilated conv layer
#         h = self.dilated_conv_layer(h)


#         # gated-tanh nonlinearity
#         out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

#         # residual and skip outputs
#         res = self.res_conv(out).permute(0, 2, 1)
#         assert x.shape == res.shape
#         skip = self.skip_conv(out).permute(0, 2, 1)

#         return (x + res) * math.sqrt(0.5), skip  # normalize for training stability

# models/diff_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from models.cond_film import CondFiLMStatic


def Conv1d_with_init(in_channels, out_channels, kernel_size, padding=0):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


# ===================== 新增：Transformer Encoder =====================
class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x: [B, T, C]
        return self.encoder(x)


class diff_CSDI(nn.Module):
    def __init__(self, channels, inputdim, num_steps, embedding_dim, rdcl, args):
        super().__init__()
        self.channels = channels
        self.pre_dim = args.pre_trained_dim
        self.embedding_hour_dim = args.embedding_hour_dim
        self.module_type = args.module_type  # conv | lstm | transformer

        self.diffusion_embedding = DiffusionEmbedding(num_steps, embedding_dim)

        self.input_projection = Conv1d_with_init(
            self.pre_dim + self.embedding_hour_dim,
            self.channels,
            1
        )

        # ===== 去噪网络（互斥）=====
        if self.module_type == "none":
            # self.denoiser = nn.ModuleList([
            #     Residual_block(
            #         res_channels=self.channels,
            #         skip_channels=self.channels,
            #         dilation=2 ** (i % 4),
            #         diffusion_step_embed_dim_out=embedding_dim,
            #     )
            #     for i in range(rdcl)
            # ])
            self.denoiser = nn.ModuleList([
                Residual_block(
                    res_channels=self.channels,
                    skip_channels=self.channels,
                    dilation=2 ** (i % 4),
                    diffusion_step_embed_dim_out=embedding_dim,
                    use_global_cond=True,
                    use_event_cond=True,
                )
                for i in range(rdcl)
            ])

        elif self.module_type == "lstm":
            self.denoiser = nn.LSTM(
                input_size=self.channels,
                hidden_size=self.channels,
                num_layers=1,
                batch_first=True
            )

        elif self.module_type == "transformer":
            self.denoiser = TemporalTransformer(
                d_model=self.channels,
                nhead=args.tf_nhead,
                num_layers=args.tf_layers,
                dropout=args.tf_dropout
            )

        else:
            raise ValueError(f"Unknown temporal_type {self.temporal_type}")

        self.output_layer = nn.Linear(
            self.channels,
            self.pre_dim + self.embedding_hour_dim
        )

        self.cond_film = CondFiLMStatic(H=self.channels)
    
    def forward(
        self,
        x,
        diffusion_step,
        cond_global=None,
        cond_event=None,
        sc_gamma=None,
        sc_beta=None
    ):
        """
        x: [B, T, F]
        """
        # 输入投影
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        x = F.relu(x).permute(0, 2, 1)  # [B,T,C]
    
        # diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)
    
        # ===== 去噪逻辑（互斥）=====
        if self.module_type == "none":
            # skip = []
            # for layer in self.denoiser:
            #     # ★ 将条件传入每一层 residual block
            #     x, s = layer(x, diffusion_emb, cond_static=cond_static)
    
            #     # 自条件 FiLM
            #     if sc_gamma is not None:
            #         x = x * (1.0 + sc_gamma) + sc_beta
            #     skip.append(s)
            # x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(skip))
            skip = []
            for layer in self.denoiser:
                x, s = layer(
                    x,
                    diffusion_emb,
                    cond_global=cond_global,
                    cond_event=cond_event
                )
                skip.append(s)
            
            x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(skip))

    
        elif self.module_type == "lstm":
            x, _ = self.denoiser(x)
    
            # 如果你希望 LSTM 也支持条件，可以在这里用 cond_static 融合：
            # 例如 x = x + cond_projection(cond_static)
    
        elif self.module_type == "transformer":
            x = self.denoiser(x)
    
            # 同理，Transformer 可以通过 cross-attention 融合 cond_static
    
        return self.output_layer(x)




# ================= Residual block 不变 =================
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class Residual_block(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        dilation,
        diffusion_step_embed_dim_out,
        use_global_cond=False,
        use_event_cond=False,
    ):
        super().__init__()
        self.res_channels = res_channels
        self.use_global_cond = use_global_cond
        self.use_event_cond = use_event_cond

        # diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv
        self.dilated_conv_layer = Conv(
            self.res_channels,
            2 * self.res_channels,
            kernel_size=3,
            dilation=dilation
        )

        # residual & skip conv
        self.res_conv = nn.utils.weight_norm(
            nn.Conv1d(res_channels, res_channels, kernel_size=1)
        )
        self.skip_conv = nn.utils.weight_norm(
            nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        )

        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        # ===== global FiLM =====
        if self.use_global_cond:
            self.global_film = CondFiLMStatic(H=res_channels)

        # ===== event residual projection =====
        if self.use_event_cond:
            self.event_proj = nn.Linear(res_channels, res_channels)

    def forward(self, x, diffusion_step_embed, cond_global=None, cond_event=None):
        """
        x: [B,T,C]
        cond_global: [B,T,C]
        cond_event:  [B,T,C]
        """
        h = x

        # diffusion embedding
        part_t = self.fc_t(diffusion_step_embed)
        h = h + part_t  # [B,T,C]

        h = h.permute(0, 2, 1)  # [B,C,T]
        h = self.dilated_conv_layer(h)

        # gated activation
        out = torch.tanh(h[:, :self.res_channels, :]) * \
              torch.sigmoid(h[:, self.res_channels:, :])

        out = out.permute(0, 2, 1)  # [B,T,C]

        # ===== global FiLM =====
        if self.use_global_cond and cond_global is not None:
            gamma, beta, gain = self.global_film(cond_global)
            out = out + gain * (out * gamma + beta)

        # ===== event residual injection =====
        if self.use_event_cond and cond_event is not None:
            out = out + self.event_proj(cond_event)

        # residual & skip
        res = self.res_conv(out.permute(0, 2, 1)).permute(0, 2, 1)
        skip = self.skip_conv(out.permute(0, 2, 1)).permute(0, 2, 1)

        return (x + res) * math.sqrt(0.5), skip



# class Residual_block(nn.Module):
#     def __init__(self, res_channels, skip_channels, dilation, diffusion_step_embed_dim_out):
#         super().__init__()
#         self.fc_t = nn.Linear(diffusion_step_embed_dim_out, res_channels)
#         self.dilated_conv = Conv(res_channels, 2 * res_channels, kernel_size=3, dilation=dilation)
#         self.res_conv = nn.utils.weight_norm(nn.Conv1d(res_channels, res_channels, 1))
#         self.skip_conv = nn.utils.weight_norm(nn.Conv1d(res_channels, skip_channels, 1))

#     def forward(self, x, diffusion_step_embed):
#         h = x + self.fc_t(diffusion_step_embed)
#         h = self.dilated_conv(h.permute(0, 2, 1))
#         out = torch.tanh(h[:, :h.size(1)//2]) * torch.sigmoid(h[:, h.size(1)//2:])
#         res = self.res_conv(out).permute(0, 2, 1)
#         skip = self.skip_conv(out).permute(0, 2, 1)
#         return (x + res) * math.sqrt(0.5), skip
