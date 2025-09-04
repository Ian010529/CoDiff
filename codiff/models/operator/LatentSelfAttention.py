import torch
import torch.nn as nn

class LatentSelfAttention(nn.Module):
    """
    在一个批次的潜在向量之间应用自注意力。
    输入和输出的形状保持严格一致: [1, Batch, Dimension]
    """
    def __init__(self, latent_dim: int, num_heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # PyTorch默认的MHA输入格式是 (Seq, Batch, Dim)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, 
                                               num_heads=num_heads)
        
        #  FFN 定义 ---
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        输入 z 的形状: [1, Batch_Size, latent_dim]ba
        输出 z 的形状: [1, Batch_Size, latent_dim]
        """
        assert len(z.shape) == 3 and z.shape[0] == 1, \
            f"Expected input shape [1, B, D], but got {z.shape}"

        
        # 使用 transpose 进行维度重排
        z_seq = z.transpose(0, 1)
        
        # --- MHSA Block ---
        # 自注意力计算
        attn_output, _ = self.attention(z_seq, z_seq, z_seq)
        # 第一个残差连接
        z_seq = self.norm1(z_seq + attn_output)
        # ------------------
        
        # --- FFN Block ---
        ffn_output = self.ffn(z_seq)
        # 第二个残差连接
        z_seq = self.norm2(z_seq + ffn_output * self.gamma)
        # z_seq = self.norm2(z_seq + ffn_output)
        # -----------------
        
        # 恢复原始形状
        return z_seq.transpose(0, 1)