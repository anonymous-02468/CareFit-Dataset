import torch
import torch.nn as nn
import sys
sys.path.append("../")
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation='leaky_relu'):
        super(ConvBlock, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, use_knowledge_jumping=True):
        super(ResidualBlock, self).__init__()
        
        self.use_knowledge_jumping = use_knowledge_jumping
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, activation='none')
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        if use_knowledge_jumping:
            self.knowledge_gate = nn.Sequential(
                nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.knowledge_transform = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
            
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        conv1_out = out
        out = self.conv2(out)
        
        if self.use_knowledge_jumping:
            combined = torch.cat([conv1_out, out], dim=1)
            attention = self.knowledge_gate(combined)
            transformed = self.knowledge_transform(combined)
            out = attention * transformed + residual
        else:
            out += residual
            
        out = self.activation(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1)
        avg_out = self.fc(avg_out).unsqueeze(-1)
        
        max_out = self.max_pool(x).squeeze(-1)
        max_out = self.fc(max_out).unsqueeze(-1)

        attention = avg_out + max_out
        return x * attention

class ECGNet(nn.Module):
    def __init__(self, input_channels=1, kernel_size=15, use_knowledge_jumping=False):
        super(ECGNet, self).__init__()

        self.kernal_size = kernel_size

        self.use_knowledge_jumping = use_knowledge_jumping
        
        self.initial_conv = nn.Sequential(
            ConvBlock(input_channels, 16, kernel_size=self.kernal_size, stride=2),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(16, 64, stride=2, use_knowledge_jumping=self.use_knowledge_jumping),
            ResidualBlock(64, 64, use_knowledge_jumping=self.use_knowledge_jumping)
        )
        
        self.attention = ChannelAttention(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,
            activation='relu',
            batch_first=True
        )
    
    def forward(self, x):
        x = self.initial_conv(x)

        x = self.residual_blocks(x)
        
        x = self.attention(x)

        x = self.global_pool(x).squeeze(-1)

        return x
    

class CoAttention(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = 64
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

        self.para1 = nn.Parameter(torch.ones(1))
        self.para2 = nn.Parameter(torch.ones(1))

    def forward(self, aligned_features):

        queries = torch.stack([self.query(x) for x in aligned_features])
        keys = torch.stack([self.key(x) for x in aligned_features])
        values = torch.stack([self.value(x) for x in aligned_features])
        
        attn_scores = torch.einsum('mbd,nbd->bmn', queries, keys) 
        attn_weights = self.softmax(attn_scores / (self.d_model ** 0.5))
        
        fused = torch.einsum('bmn,nbd->mbd', attn_weights, values)
        fused = fused[0] * self.para1 + fused[1] * (1 - self.para1)

        return fused

class MultimodalFusionBlock(nn.Module):
    def __init__(self, d_model=256, ecg_encoder=None, smo2_encoder=None):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.smo2_encoder = smo2_encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU(),
            nn.Linear(16, 64)
        )

        self.co_attention = CoAttention()
        self.regressor = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.regressor64_1 = nn.Sequential(
            nn.Linear(32, 1)
        )

        self.regressor64_2 = nn.Sequential(
            nn.Linear(32, 1)
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, ecg, smo2, personal_feat):
        smo2_feat = self.smo2_encoder(smo2)
        ecg_feat = self.ecg_encoder(ecg)
        text_fused_feat = self.text_encoder(personal_feat)
        modalities = [ecg_feat, smo2_feat]
        fused_feat = self.co_attention(modalities) + self.alpha * ecg_feat + self.beta * text_fused_feat
        signals_feat = self.regressor(fused_feat)
        output1 = self.regressor64_1(signals_feat)
        output2 = self.regressor64_2(signals_feat)
        return output1, output2


class EnhancedTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1):
        super(EnhancedTimeSeriesModel, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.dropout = 0.2
        self.attention_dim = 64

        self.input_norm = nn.LayerNorm(input_size)
        

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.depthwise_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            groups=1,
            padding=1
        )
        self.pointwise_conv = nn.Conv1d(1, 1, kernel_size=1)
        
        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 64)
        )
        
    
    def forward(self, x):
        # (batch, 1, seq_len)
        x = rearrange(x, 'b d t -> b t d')  # (batch, seq_len, 1)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)
        
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_size*2)
        
        output = self.output_net(context_vector)
        
        return output
