import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 核心组件：带残差的 SE 模块 ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(8, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# --- ASPP 模块 (多尺度空洞卷积金字塔) ---
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[2, 4, 6]):
        super(ASPP, self).__init__()

        # ✅ 用 GroupNorm 替代 BatchNorm，避免 batch_size=1 的问题
        def make_conv_block(in_c, out_c, kernel, padding=0, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel, padding=padding, dilation=dilation, bias=False),
                nn.GroupNorm(8, out_c),  # 8 个 group，适配 64 通道
                nn.ReLU(inplace=True)
            )

        # 1x1 卷积分支
        self.conv1 = make_conv_block(in_ch, out_ch, 1)

        # 多个空洞卷积分支
        self.aspp_branches = nn.ModuleList()
        for rate in rates:
            self.aspp_branches.append(
                make_conv_block(in_ch, out_ch, 3, padding=rate, dilation=rate)
            )

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

        # 融合所有分支
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        aspp_outs = [branch(x) for branch in self.aspp_branches]

        x_global = self.global_avg_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1] + aspp_outs + [x_global], dim=1)

        return self.conv_out(x_cat)

# --- 基础卷积层 ---
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


# --- 抽象 RSU 模块 (集成输出端 SE 和 残差) ---
class RSU(nn.Module):
    def __init__(self, name, in_ch, mid_ch, out_ch, use_se=True):
        super(RSU, self).__init__()
        n = int(name[-1]) if name[-1].isdigit() else 4

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # Encoder
        self.enc_layers = nn.ModuleList([REBNCONV(out_ch, mid_ch, dirate=1)])
        for i in range(n - 2):
            self.enc_layers.append(REBNCONV(mid_ch, mid_ch, dirate=1))

        # Bottom
        self.bottom = REBNCONV(mid_ch, mid_ch, dirate=1)

        # Decoder
        self.dec_layers = nn.ModuleList([])
        for i in range(n - 2):
            self.dec_layers.append(REBNCONV(mid_ch * 2, mid_ch, dirate=1))
        self.dec_final = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        hxin = self.rebnconvin(x)

        # Encoder forward
        responses = [self.enc_layers[0](hxin)]
        for i in range(1, len(self.enc_layers)):
            responses.append(self.enc_layers[i](self.pool(responses[-1])))

        # Bottom
        hx = self.bottom(self.pool(responses[-1]))

        # Decoder forward
        for i in range(len(self.dec_layers)):
            hx = self.dec_layers[i](torch.cat((_upsample_like(hx, responses[-(i + 1)]), responses[-(i + 1)]), 1))

        hx1d = self.dec_final(torch.cat((_upsample_like(hx, responses[0]), responses[0]), 1))

        return self.se(hx1d + hxin)



# --- 最终集成的 U2NETP (加入 ASPP) ---
class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.stage1 = RSU('RSU7', in_ch, 16, 64)
        self.stage2 = RSU('RSU6', 64, 16, 64)
        self.stage3 = RSU('RSU5', 64, 16, 64)
        self.stage4 = RSU('RSU4', 64, 16, 64)
        self.stage5 = RSU('RSU4', 64, 16, 64)
        self.stage6 = RSU('RSU4', 64, 16, 64)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # 只保留一个轻量 ASPP
        self.aspp = ASPP(64, 64, rates=[2, 4, 6])

        self.stage5d = RSU('RSU4', 128, 16, 64)
        self.stage4d = RSU('RSU4', 128, 16, 64)
        self.stage3d = RSU('RSU5', 128, 16, 64)
        self.stage2d = RSU('RSU6', 128, 16, 64)
        self.stage1d = RSU('RSU7', 128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        # 新增边缘分支
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx2 = self.stage2(self.pool(hx1))
        hx3 = self.stage3(self.pool(hx2))
        hx4 = self.stage4(self.pool(hx3))
        hx5 = self.stage5(self.pool(hx4))
        hx6 = self.stage6(self.pool(hx5))

        hx6 = self.aspp(hx6)

        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        # 浅层边缘输出
        edge = self.edge_head(hx1)

        return (
            torch.sigmoid(d0),
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
            torch.sigmoid(edge)
        )