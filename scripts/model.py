import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, MessagePassing, GATConv
from torch_scatter import scatter_softmax


class AttentiveCrossMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=4, dropout_rate=0.3):
        super().__init__(aggr='add')
        self.heads = heads
        self.attn_dim = out_channels // heads
        self.out_channels = out_channels
        
        # 边特征投影
        self.edge_proj = Linear(edge_dim, in_channels)
        input_dim = in_channels * 3  # x_i + x_j + proj_edge

        # Q/K投影层
        self.attn_query = Linear(in_channels, self.attn_dim * heads)
        self.attn_key = Linear(in_channels, self.attn_dim * heads)

        # V投影层
        self.attn_value = Sequential(
            Linear(input_dim, self.attn_dim * heads),
            nn.ReLU(),
            Dropout(dropout_rate),
            Linear(self.attn_dim * heads, self.attn_dim * heads)
        )

        self.attn_scale = nn.Parameter(torch.ones(heads))
        self.out_proj = Linear(out_channels, out_channels)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index):
        # 投影边特征
        edge_attr_proj = self.edge_proj(edge_attr)

        # 生成Q/K/V
        query = self.attn_query(x_j).view(-1, self.heads, self.attn_dim)
        key = self.attn_key(x_i).view(-1, self.heads, self.attn_dim)
        v_input = torch.cat([x_i, x_j, edge_attr_proj], dim=-1)
        value = self.attn_value(v_input).view(-1, self.heads, self.attn_dim)

        # 注意力计算
        attn_logits = (query * key).sum(dim=-1) * self.attn_scale
        row = edge_index[1]
        attn_weights = scatter_softmax(attn_logits, row, dim=0)
        attn_weights = self.dropout(attn_weights)

        # 加权消息
        weighted_msg = value * attn_weights.unsqueeze(-1)
        weighted_msg = weighted_msg.view(-1, self.out_channels)
        return self.out_proj(weighted_msg)

    def update(self, aggr_out):
        return aggr_out


class MPNN_HeteroGNN(nn.Module):
    def __init__(self, num_residue_types, num_lig_atom_types, 
                 num_interac_PP_type, num_interac_PL_type,
                 num_heads=4, hidden_dim=128, dropout_rate=0.3, num_blocks=3):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # 节点特征投影
        self.residue_proj = Sequential(
            Linear(num_residue_types, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.ligand_proj = Sequential(
            Linear(num_lig_atom_types, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 消息传递模块
        self.residue_blocks = nn.ModuleList()
        self.ligand_blocks = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()

        # 边更新模块
        self.edge_updaters = nn.ModuleList()

        for _ in range(num_blocks):
            # Residue内部模块
            residue_block = nn.ModuleDict({
                'peptide': GATConv(hidden_dim, hidden_dim, dropout=dropout_rate),
                'interaction': AttentiveCrossMessagePassing(
                    hidden_dim, hidden_dim, num_interac_PP_type, 
                    heads=num_heads, dropout_rate=dropout_rate
                ),
                'geometric': AttentiveCrossMessagePassing(
                    hidden_dim, hidden_dim, 16, 
                    heads=num_heads, dropout_rate=dropout_rate
                ),
                'norm': nn.LayerNorm(hidden_dim)
            })
            self.residue_blocks.append(residue_block)

            # Ligand内部模块
            ligand_block = nn.ModuleDict({
                'ligand': GATConv(hidden_dim, hidden_dim, dropout=dropout_rate),
                'norm': nn.LayerNorm(hidden_dim)
            })
            self.ligand_blocks.append(ligand_block)

            # 跨类型模块
            cross_block = nn.ModuleDict({
                'cross': AttentiveCrossMessagePassing(
                    hidden_dim, hidden_dim, num_interac_PL_type,
                    heads=num_heads, dropout_rate=dropout_rate
                ),
                'cross_geo': AttentiveCrossMessagePassing(
                    hidden_dim, hidden_dim, 5,
                    heads=1, dropout_rate=dropout_rate
                ),
                'norm': nn.LayerNorm(hidden_dim)
            })
            self.cross_blocks.append(cross_block)

            # 边特征更新器
            edge_updater = nn.ModuleDict({
                'PL_edge': Sequential(
                    Linear(2*hidden_dim + num_interac_PL_type, hidden_dim),
                    nn.ReLU(),
                    Dropout(dropout_rate),
                    Linear(hidden_dim, num_interac_PL_type)
                ),
                'PP_edge': Sequential(
                    Linear(2*hidden_dim + num_interac_PP_type, hidden_dim),
                    nn.ReLU(),
                    Dropout(dropout_rate),
                    Linear(hidden_dim, num_interac_PP_type)
                )
            })
            self.edge_updaters.append(edge_updater)

        # 分类层
        self.classifier_before = Sequential(
            Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            Dropout(p=dropout_rate),
            Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            Dropout(p=dropout_rate)
        )
        
        self.classifier_final = Linear(128, 20)

        # 边预测模块
        self.edge_predictors = nn.ModuleDict({
            'PL': nn.ModuleDict({
                'count': self._build_edge_predictor(num_interac_PL_type),
                'zero_logits': self._build_edge_predictor(num_interac_PL_type, output_act=None)
            }),
            'PP': nn.ModuleDict({
                'count': self._build_edge_predictor(num_interac_PP_type),
                'zero_logits': self._build_edge_predictor(num_interac_PP_type, output_act=None)
            })
        })
        
    
    def _build_edge_predictor(self, edge_dim, output_act=nn.Softplus()):
        layers = [
            Linear(2*self.hidden_dim + edge_dim, 256),
            nn.ReLU(),
            Dropout(self.dropout_rate),
            Linear(256, edge_dim)
        ]
        if output_act is not None:
            layers.append(output_act)
        return Sequential(*layers)
    

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch):
        # 初始化特征
        x_res = self.residue_proj(x_dict['residue'])
        x_lig = self.ligand_proj(x_dict['ligand'])
        current_edge_attrs = {
            ('residue', 'interacts_between', 'residue'): edge_attr_dict[('residue', 'interacts_between', 'residue')].clone(),
            ('ligand', 'interacts_with', 'residue'): edge_attr_dict[('ligand', 'interacts_with', 'residue')].clone(),
            ('residue', 'geometrical', 'residue'): edge_attr_dict[('residue', 'geometrical', 'residue')].clone(),
            ('ligand', 'geometrical', 'residue'): edge_attr_dict[('ligand', 'geometrical', 'residue')].clone()
        }

        for i in range(self.num_blocks):
            # Residue内部更新
            msg_peptide = self.residue_blocks[i]['peptide'](
                x_res, edge_index_dict[('residue', 'peptide_bonded', 'residue')]
            )
            pp_geo_edge_index = edge_index_dict[('residue', 'geometrical', 'residue')]
            pp_geo_edge_attr = current_edge_attrs[('residue', 'geometrical', 'residue')]
            msg_geometric = self.residue_blocks[i]['geometric'](
                (x_res, x_res), pp_geo_edge_index, pp_geo_edge_attr
            )
            pp_edge_index = edge_index_dict[('residue', 'interacts_between', 'residue')]
            pp_edge_attr = current_edge_attrs[('residue', 'interacts_between', 'residue')]
            msg_interact = self.residue_blocks[i]['interaction'](
                (x_res, x_res), pp_edge_index, pp_edge_attr
            )
            x_res = self.residue_blocks[i]['norm'](x_res + msg_peptide + msg_interact + msg_geometric)

            # Ligand内部更新
            msg_lig = self.ligand_blocks[i]['ligand'](x_lig, edge_index_dict[('ligand', 'ligand_bonded', 'ligand')])
            x_lig = self.ligand_blocks[i]['norm'](x_lig + msg_lig)

            # 跨类型更新
            pl_geo_edge_index = edge_index_dict[('ligand', 'geometrical', 'residue')]
            pl_geo_edge_attr = current_edge_attrs[('ligand', 'geometrical', 'residue')]
            msg_cross_geo = self.cross_blocks[i]['cross_geo'](
                (x_lig, x_res), pl_geo_edge_index, pl_geo_edge_attr
            )
            pl_edge_index = edge_index_dict[('ligand', 'interacts_with', 'residue')]
            pl_edge_attr = current_edge_attrs[('ligand', 'interacts_with', 'residue')]
            msg_cross = self.cross_blocks[i]['cross'](
                (x_lig, x_res), pl_edge_index, pl_edge_attr
            )
            x_res = self.cross_blocks[i]['norm'](x_res + msg_cross + msg_cross_geo)

            # 更新边特征
            # 更新PP边
            src_pp, dst_pp = pp_edge_index
            pp_edge_feat = torch.cat([x_res[src_pp], x_res[dst_pp], pp_edge_attr], dim=-1)
            current_edge_attrs[('residue', 'interacts_between', 'residue')] = \
                self.edge_updaters[i]['PP_edge'](pp_edge_feat)
            
            # 更新PL边
            src_pl, dst_pl = pl_edge_index[0], pl_edge_index[1]
            pl_edge_feat = torch.cat([x_lig[src_pl], x_res[dst_pl], pl_edge_attr], dim=-1)
            current_edge_attrs[('ligand', 'interacts_with', 'residue')] = \
                self.edge_updaters[i]['PL_edge'](pl_edge_feat)

        # 分类输出
        features_128 = self.classifier_before(x_res)
        residue_logits = self.classifier_final(features_128)

        # 更新边预测部分
        edge_pred_dict = {}
        # 预测PP边
        pp_src, pp_dst = edge_index_dict[('residue', 'interacts_between', 'residue')]
        pp_edge_feat = torch.cat([x_res[pp_src], x_res[pp_dst], current_edge_attrs[('residue', 'interacts_between', 'residue')]], dim=-1)
        edge_pred_dict[('residue', 'interacts_between', 'residue')] = {
            'count': self.edge_predictors['PP']['count'](pp_edge_feat),
            'zero_logits': self.edge_predictors['PP']['zero_logits'](pp_edge_feat)
        }

        # 预测PL边
        pl_src, pl_dst = edge_index_dict[('ligand', 'interacts_with', 'residue')]
        pl_edge_feat = torch.cat([x_lig[pl_src], x_res[pl_dst], current_edge_attrs[('ligand', 'interacts_with', 'residue')]], dim=-1)
        edge_pred_dict[('ligand', 'interacts_with', 'residue')] = {
            'count': self.edge_predictors['PL']['count'](pl_edge_feat),
            'zero_logits': self.edge_predictors['PL']['zero_logits'](pl_edge_feat)
        }

        return {
            'residue': residue_logits,
            'features_128': features_128,  # 新增128维特征输出
            'edges': edge_pred_dict
        }