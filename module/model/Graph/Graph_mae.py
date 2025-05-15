import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GAT, GAE
from itertools import chain

from functools import partial
from .loss_func import sce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_module(m_type, in_dim, num_hidden, out_dim, num_layers, dropout, activation, norm, **kwargs) -> nn.Module:
    if m_type in "gat":
        model = GAT(in_channels=in_dim,
                    hidden_channels=num_hidden,
                    num_layers=num_layers,
                    out_channels=out_dim,
                    dropout=dropout,
                    act=activation,
                    norm=norm,
                    **kwargs)
    elif m_type in "gcn":
        model = GCN(in_channels=in_dim,
                    hidden_channels=num_hidden,
                    num_layers=num_layers,
                    out_channels=out_dim,
                    dropout=dropout,
                    act=activation,
                    norm=norm,
                    **kwargs)
    elif m_type == "mlp":
        # * just for decoder
        model = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        model = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError(f"{m_type} not implemented")

    return model

class MAE2(nn.Module):
    def __init__(
            self,
            cfg,
            remask_method: str = "fixed",
            mask_method: str = "random",
            delayed_ema_epoch: int = 0,
            momentum: float = 0.995,
            replace_rate: float = 0.0,
            zero_init: bool = False,
        ):
        super(MAE2, self).__init__()
        in_dim = cfg.in_dim
        num_hidden = cfg.num_hidden
        num_layers = cfg.num_layers
        num_dec_layers = cfg.num_dec_layers
        num_remasking = cfg.num_remasking
        nhead = cfg.nhead
        nhead_out=cfg.nhead_out
        activation = cfg.activation
        feat_drop = cfg.feat_drop
        negative_slope = cfg.negative_slope
        residual = cfg.residual
        norm = cfg.norm
        alpha_l = cfg.alpha_l
        lam = cfg.lam


        mask_ratio = cfg.mask_ratio
        remask_ratio = cfg.remask_ratio
        encoder_type= cfg.encoder_type
        decoder_type= cfg.decoder_type
        loss_fn = cfg.loss_fn

        self._mask_rate = mask_ratio
        self._remask_rate = remask_ratio
        self._remask_method = remask_method
        self._mask_method = mask_method
        self._alpha_l = alpha_l

        self.delayed_ema_epoch = delayed_ema_epoch

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._output_hidden_size = num_hidden
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam

        self.masked_node_num = 0

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        model_args ={}
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            model_args["heads"] = nhead
            model_args["negative_slope"] = negative_slope
            model_args["residual"] = residual

        else:
            enc_num_hidden = num_hidden

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden

        self.encoder = setup_module(
            m_type=encoder_type,
            in_dim= in_dim,
            num_hidden=num_hidden,
            out_dim=num_hidden,
            num_layers=num_layers,
            dropout=feat_drop,
            activation=activation,
            norm=norm,
            **model_args)


        self.decoder = setup_module(
            m_type=decoder_type,
            in_dim=num_hidden,
            num_hidden=num_hidden,
            out_dim=in_dim,
            num_layers=num_dec_layers,
            dropout=feat_drop,
            activation=activation,
            norm=norm,
            **model_args)

        # self.dec_fc = nn.Sequential(nn.Linear(in_dim, in_dim),
        #                         nn.LayerNorm(in_dim),
        #                         nn.Sigmoid())
        self.dec_fc = nn.Identity()

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim)).to(device)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden)).to(device)

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if not zero_init:
            self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 16),
            nn.PReLU(),
            nn.Linear(16, num_hidden),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 16),
            nn.PReLU(),
            nn.Linear(16, num_hidden),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )

        self.encoder_ema = setup_module(
            m_type=encoder_type,
            in_dim= in_dim,
            num_hidden=num_hidden,
            out_dim=num_hidden,
            num_layers=num_layers,
            dropout=feat_drop,
            activation=activation,
            norm=norm,
            **model_args)

        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()

        self.print_num_parameters()

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(
            f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l=None):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    def mask_predict(self,x, edge_index):

        x_tilda, mask = self.encoding_mask_noise(x, self._mask_rate)
        h = self.encoder(x_tilda, edge_index)
        h_orig = self.encoder_to_decoder(h)

        h_tilda = self.fixed_remask(h_orig, mask)
        z = self.dec_fc(self.decoder(h_tilda, edge_index))
        return z


    def forward(self, g,targets=None, drop_g1=None, drop_g2=None):
        loss = self.mask_attr_prediction(g, targets, drop_g1, drop_g2)

        return loss

    def mask_attr_prediction(self, g, targets=None, drop_g1=None, drop_g2=None):
        x, edge_index = g.x, g.edge_index

        x_tilda, mask = self.encoding_mask_noise(x, self._mask_rate)

        h = self.encoder(x_tilda,edge_index)
        with torch.no_grad():
            x_bar = self.projector_ema(self.encoder_ema(x,edge_index))

        z_bar = self.predictor(self.projector(h))
        loss_latent = sce_loss(z_bar, x_bar, alpha=1)

        # ---- attribute reconstruction ----
        h_orig = self.encoder_to_decoder(h)

        loss_rec_all = 0
        if self._remask_method == "random":
            for j in range(self._num_remasking):
                h = h_orig.clone()
                h_tilda, remask = self.random_remask(h,self._mask_rate)
                z_j = self.dec_fc(self.decoder(h_tilda, edge_index))

                loss_rec = self.criterion(x[mask], z_j[mask])
                loss_rec_all += loss_rec

            loss_rec = loss_rec_all

        elif self._remask_method == "fixed":
            h_tilda = self.fixed_remask(h,mask)
            z = self.dec_fc(self.decoder(h_tilda, edge_index))
            loss_rec = self.criterion(x[mask],z[mask])
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self,x, edge_index):
        h = self.encoder(x, edge_index)
        return h

    def reconstruct_random_mask(self, g, mask_ratio=None):
        """
        Randomly masks a subset of nodes, then reconstructs their features.
        Returns:
          rec_feats: Tensor of shape [num_masked, in_dim] — reconstructed node features
          masked_idx: 1D LongTensor of masked node indices
        """
        x, edge_index = g.x, g.edge_index
        device = x.device

        # 1) Determine mask ratio
        mask_rate = mask_ratio if mask_ratio is not None else self._mask_rate

        # 2) Randomly mask input features (using encoding_mask_noise)
        x_masked, mask = self.encoding_mask_noise(x, mask_rate)
        masked_idx = torch.nonzero(mask, as_tuple=False).view(-1)

        # 3) Encode & prepare for decoder
        h = self.encoder(x_masked, edge_index)  # [N, H]
        h2d = self.encoder_to_decoder(h)  # [N, H]

        # 4) Insert decoder mask tokens at masked positions
        h2d_masked = self.dec_fc(self.fixed_remask(h2d, mask))

        # 5) Decode to reconstruct original feature dimension
        rec = self.decoder(h2d_masked, edge_index)  # [N, in_dim]

        # 6) Extract reconstructed features for masked nodes
        rec_feats = rec[masked_idx]

        return rec, rec_feats

    def get_encoder(self):
        return self.encoder

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters(), self.dec_fc.parameters()])

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict


    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.size(0)
        mask_vector = torch.ones(num_nodes, device=x.device)
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        self.masked_node_num = num_mask_nodes

        mask_nodes = perm[: num_mask_nodes]
        mask_vector[mask_nodes] = 0
        mask = (mask_vector.squeeze()==0)
        out_x = x.clone()

        out_x[mask] = 0
        out_x[mask] += self.enc_mask_token

        return out_x, mask

    def random_remask(self, h, remask_rate=0.5):
        num_nodes = h.size(0)
        remask_vector = torch.ones(num_nodes, device=h.device)
        perm = torch.randperm(num_nodes, device=h.device)

        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        remask_vector[remask_nodes] = 0
        remask = (remask_vector.squeeze() == 0)

        out_h = h.clone()

        out_h[remask] = 0
        out_h[remask] += self.dec_mask_token

        return out_h, remask

    def fixed_remask(self, h, mask):
        h = h.clone()
        h[mask] = 0
        h[mask] += self.dec_mask_token
        return h


