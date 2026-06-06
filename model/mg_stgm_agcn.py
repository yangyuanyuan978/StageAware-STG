import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# ==================== 多粒度邻接矩阵构建 ====================
def get_limb_adjacency(num_node=25):
    """
    构建肢体级邻接矩阵
    支持 NTU (25关节点) 和 Kinetics (18关节点) 格式
    """
    A = np.zeros((num_node, num_node))
    
    if num_node == 25:
        # NTU 格式: 25个关节点
        spine = [0, 1, 20, 2, 3]
        left_arm = [20, 4, 5, 6, 7, 21, 22]
        right_arm = [20, 8, 9, 10, 11, 23, 24]
        left_leg = [0, 12, 13, 14, 15]
        right_leg = [0, 16, 17, 18, 19]
    elif num_node == 18:
        # Kinetics 格式: 18个关节点
        spine = [0, 1, 2, 3, 4, 5]
        left_arm = [5, 6, 7, 8]
        right_arm = [2, 9, 10, 11]
        left_leg = [11, 12, 13, 14]
        right_leg = [8, 15, 16, 17]
    else:
        # 对于其他关节点数量，使用简单的最近邻连接
        for i in range(num_node):
            for j in range(max(0, i-2), min(num_node, i+3)):
                A[i, j] = 1
        return A

    limbs = [spine, left_arm, right_arm, left_leg, right_leg]

    for limb in limbs:
        for i in limb:
            for j in limb:
                if i < num_node and j < num_node:
                    A[i, j] = 1

    return A



def get_global_adjacency(num_node=25):
    """构建全局全连接邻接矩阵"""
    return np.ones((num_node, num_node))


def normalize_adjacency(A):
    """归一化邻接矩阵"""
    A = A + np.eye(A.shape[0])
    D = np.sum(A, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0.
    D = np.diag(D)
    return np.dot(np.dot(D, A), D)


def normalize_digraph_np(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


# ==================== 基础模块 ====================

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.PA

        x_flat = x.view(N, C * T, V)
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, -1)
            A2 = self.conv_b[i](x).view(N, -1, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1)) + A[i]
            z = self.conv_d[i](torch.matmul(x_flat, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        del x_flat

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


# ==================== 多粒度GCN模块（加 mg_mode 补丁） ====================

class MultiGranularityGCN(nn.Module):
    """
    高效多粒度GCN - 通过邻接矩阵融合实现多粒度
    计算量与原始GCN相同，但能捕获多粒度信息

    mg_mode:
      - 'learned'：默认，fusion_weights 可学习（保持你原逻辑）
      - 'joint'  ：仅关节级
      - 'limb'   ：仅肢体级
      - 'global' ：仅全局级
    """
    def __init__(self, in_channels, out_channels, A_joint, num_point=25, coff_embedding=4, num_subset=3,
                 mg_mode='learned', mg_fusion='static', gate_reduction=16, gate_dropout=0.0, pa_lambda=1.0,
                 align_norm=False):
        super(MultiGranularityGCN, self).__init__()

        align_norm = bool(align_norm)
        if align_norm:
            A_limb = normalize_digraph_np(get_limb_adjacency(num_point))
            A_global = normalize_digraph_np(get_global_adjacency(num_point))
        else:
            A_limb = normalize_adjacency(get_limb_adjacency(num_point))
            A_global = normalize_adjacency(get_global_adjacency(num_point))

        self.register_buffer('A_joint', torch.from_numpy(A_joint.astype(np.float32)))
        self.register_buffer('A_limb', torch.from_numpy(np.stack([A_limb] * num_subset).astype(np.float32)))
        self.register_buffer('A_global', torch.from_numpy(np.stack([A_global] * num_subset).astype(np.float32)))

        mg_mode = str(mg_mode).lower() if mg_mode is not None else 'learned'
        self.mg_mode = mg_mode
        self.mg_fusion = str(mg_fusion).lower() if mg_fusion is not None else 'static'
        self.pa_lambda = float(pa_lambda)

        if mg_mode == 'joint':
            self.fusion_weights = nn.Parameter(torch.tensor([20.0, -20.0, -20.0], dtype=torch.float32),
                                               requires_grad=False)
        elif mg_mode == 'limb':
            self.fusion_weights = nn.Parameter(torch.tensor([-20.0, 20.0, -20.0], dtype=torch.float32),
                                               requires_grad=False)
        elif mg_mode == 'global':
            self.fusion_weights = nn.Parameter(torch.tensor([-20.0, -20.0, 20.0], dtype=torch.float32),
                                               requires_grad=False)
        else:
            self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32))

        if self.mg_fusion == 'dynamic_subset':
            hidden = max(1, int(in_channels) // int(gate_reduction))
            self.gate = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(),
                nn.Dropout(p=float(gate_dropout)),
                nn.Linear(hidden, num_subset * 3)
            )

        self.PA = nn.Parameter(torch.from_numpy(A_joint.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(num_subset):
            conv_branch_init(self.conv_d[i], num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        if self.mg_fusion == 'dynamic_subset':
            if self.mg_mode in ('joint', 'limb', 'global'):
                w = torch.softmax(self.fusion_weights, dim=0)
                W = w.view(1, 1, 3).expand(N, self.num_subset, 3)
            else:
                u = x.mean(dim=(2, 3))
                W = self.gate(u).view(N, self.num_subset, 3)
                W = torch.softmax(W, dim=-1)

            A_stack = torch.stack([self.A_joint, self.A_limb, self.A_global], dim=1)
            A = torch.einsum('nsg,sgvw->nsvw', W, A_stack)
            A = A + (self.pa_lambda * self.PA).unsqueeze(0)
        else:
            w = torch.softmax(self.fusion_weights, dim=0)
            A = w[0] * self.A_joint + w[1] * self.A_limb + w[2] * self.A_global + (self.pa_lambda * self.PA)

        x_flat = x.view(N, C * T, V)
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, -1)
            A2 = self.conv_b[i](x).view(N, -1, V)
            if self.mg_fusion == 'dynamic_subset':
                A_base = A[:, i]
            else:
                A_base = A[i]
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1)) + A_base
            z = self.conv_d[i](torch.matmul(x_flat, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        del x_flat

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


# ====================STGM模块 ====================

class SelectiveTemporalGatingBlock(nn.Module):
    """
    STGM-inspired selective temporal gating block.
+    This block keeps input-dependent temporal/channel gating,
+    but does not implement recurrent state-space selective scan.
    """
    def __init__(self, d_model, d_state=8, d_conv=4, expand=1.5):
        super(SelectiveTemporalGatingBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.act = nn.SiLU()

    def forward(self, x):
        B, C, T, V = x.shape

        x = rearrange(x, 'b c t v -> (b v) t c')

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        del xz

        x = rearrange(x, 'b t d -> b d t')
        x = self.conv1d(x)[:, :, :T]
        x = rearrange(x, 'b d t -> b t d')

        x = self.act(x)
        x_proj = self.x_proj(x)
        dt = self.dt_proj(x_proj)
        del x_proj

        x = x * torch.sigmoid(dt)
        del dt

        x = x * self.act(z)
        del z

        x = self.out_proj(x)

        x = rearrange(x, '(b v) t c -> b c t v', b=B, v=V)
        return x


# ==================== 组合单元 ====================

class TCN_GCN_unit(nn.Module):
    """标准TCN-GCN单元"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class MG_TCN_GCN_unit(nn.Module):
    """多粒度TCN-GCN单元（加 mg_mode 参数并向下传递）"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_point=25,
                 mg_mode='learned', mg_fusion='static', gate_reduction=16, gate_dropout=0.0, pa_lambda=1.0,
                 align_norm=False):
        super(MG_TCN_GCN_unit, self).__init__()
        self.gcn1 = MultiGranularityGCN(
            in_channels, out_channels, A, num_point,
            mg_mode=mg_mode,
            mg_fusion=mg_fusion,
            gate_reduction=gate_reduction,
            gate_dropout=gate_dropout,
            pa_lambda=pa_lambda,
            align_norm=align_norm,
        )
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class STGM_GCN_unit(nn.Module):
    """
    GCN unit with the proposed Selective Temporal Gating Module (STGM).
+    The attribute name `stgm` is kept only for backward compatibility
+    with existing checkpoints/configurations.
    """
    def __init__(self, in_channels, out_channels, A, d_state=8, stride=1, residual=True):
        super(STGM_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.stgm = SelectiveTemporalGatingBlock(out_channels, d_state=d_state, expand=1.5)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.stride = stride

        if stride > 1:
            self.temporal_pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
        else:
            self.temporal_pool = None

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        bn_init(self.bn, 1)

    def forward(self, x):
        gcn_out = self.gcn1(x)
        stgm_out = self.stgm(gcn_out)

        if self.temporal_pool is not None:
            stgm_out = self.temporal_pool(stgm_out)

        out = self.bn(stgm_out) + self.residual(x)
        return self.relu(out)


# ==================== 主模型（加 mg_mode 参数） ====================

class Model(nn.Module):
    """
    Multi-Granularity Selective Temporal Gating AGCN (MG-STGM-AGCN)

    新增 mg_mode：
      - learned/joint/limb/global
    """
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=dict(), in_channels=3,
                 use_mg=True, use_stgm=True, stgm_layers=[8, 9, 10],
                 use_stgm=None, stgm_layers=None,
                 use_checkpoint=False, mg_mode='learned', mg_fusion='static', gate_reduction=16,
                 gate_dropout=0.0, pa_lambda=1.0, align_norm=False):
        super(Model, self).__init__()
        self._use_checkpoint = use_checkpoint
        # Backward-compatible aliases: old configs may still use use_stgm/stgm_layers.
        if use_stgm is None:
            use_stgm = use_stgm
        if stgm_layers is None:
            stgm_layers = stgm_layers

        if graph is None:
            raise ValueError("Graph cannot be None")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # Layer 1-3: 多粒度GCN或标准GCN
        if use_mg:
            self.l1 = MG_TCN_GCN_unit(
                3, 64, A, residual=False, num_point=num_point,
                mg_mode=mg_mode, mg_fusion=mg_fusion, gate_reduction=gate_reduction,
                gate_dropout=gate_dropout, pa_lambda=pa_lambda,
                align_norm=align_norm,
            )
            self.l2 = MG_TCN_GCN_unit(
                64, 64, A, num_point=num_point,
                mg_mode=mg_mode, mg_fusion=mg_fusion, gate_reduction=gate_reduction,
                gate_dropout=gate_dropout, pa_lambda=pa_lambda,
                align_norm=align_norm,
            )
            self.l3 = MG_TCN_GCN_unit(
                64, 64, A, num_point=num_point,
                mg_mode=mg_mode, mg_fusion=mg_fusion, gate_reduction=gate_reduction,
                gate_dropout=gate_dropout, pa_lambda=pa_lambda,
                align_norm=align_norm,
            )
        else:
            self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
            self.l2 = TCN_GCN_unit(64, 64, A)
            self.l3 = TCN_GCN_unit(64, 64, A)

        # Layer 4-7: 标准TCN-GCN
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)

       # Layer 8-10: STGM-GCN or TCN-GCN
        if use_stgm and 8 in stgm_layers:
+            self.l8 = STGM_GCN_unit(128, 256, A, stride=2)
        else:
            self.l8 = TCN_GCN_unit(128, 256, A, stride=2)

        if use_stgm and 9 in stgm_layers:
+            self.l9 = STGM_GCN_unit(256, 256, A)
        else:
            self.l9 = TCN_GCN_unit(256, 256, A)

        if use_stgm and 10 in stgm_layers:
+            self.l10 = STGM_GCN_unit(256, 256, A)
        else:
            self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        if self._use_checkpoint and self.training:
            x = checkpoint(self.l8, x, use_reentrant=False)
            x = checkpoint(self.l9, x, use_reentrant=False)
            x = checkpoint(self.l10, x, use_reentrant=False)
        else:
            x = self.l8(x)
            x = self.l9(x)
            x = self.l10(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)