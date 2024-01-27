import torch
import torch.nn as nn

global mu
global var


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sizes = [config["input"]] + config["hiddens"]
        out_sizes = config["hiddens"] + [config["output"]]
        layers = []
        for in_size, out_size, use_bias, dropout, use_bn, use_ln, activation in zip(
                in_sizes,
                out_sizes,
                config["use_biases"],
                config["dropouts"],
                config["use_batch_norms"],
                config["use_layer_norms"],
                config["activations"],
        ):
            layers.append(nn.Linear(in_size, out_size, use_bias))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_size))
            if use_ln:
                layers.append(nn.LayerNorm(out_size))
            if activation is not None:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "softmax":
                    layers.append(nn.Softmax())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation.startswith("leaky_relu"):
                    neg_slope = float(activation.split(":")[1])
                    layers.append(nn.LeakyReLU(neg_slope))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinBnDrop(nn.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


class Encoder(nn.Module):
    def __init__(self, args, nfeatures_modality1=1, nfeatures_modality2=1, hidden_modality1=1, hidden_modality2=1,
                 z_dim=64):
        super(Encoder, self).__init__()
        args.encoder['input'] = hidden_modality1 + hidden_modality2
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.encoder_modality1 = LinBnDrop(self.nfeatures_modality1, hidden_modality1, p=0.2, act=nn.ReLU())
        self.encoder_modality2 = LinBnDrop(self.nfeatures_modality2, hidden_modality2, p=0.2, act=nn.ReLU())
        self.encoder = MLP(args.encoder)
        self.weights_modality1 = nn.Parameter(torch.rand((1, self.nfeatures_modality1)) * 0.001, requires_grad=True)
        self.weights_modality2 = nn.Parameter(torch.rand((1, self.nfeatures_modality2)) * 0.001, requires_grad=True)
        self.fc_mu = LinBnDrop(z_dim, z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim, z_dim, p=0.2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        global mu
        global var
        x_modality1 = self.encoder_modality1(x[:, :self.nfeatures_modality1] * self.weights_modality1)
        x_modality2 = self.encoder_modality2(x[:, self.nfeatures_modality1:] * self.weights_modality2)
        x = torch.cat([x_modality1, x_modality2], 1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x


class Decoder(nn.Module):
    def __init__(self, nfeatures_modality1=1, nfeatures_modality2=1, z_dim=64):
        super(Decoder, self).__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.decoder1 = LinBnDrop(z_dim, nfeatures_modality1, act=nn.ReLU())
        self.decoder2 = LinBnDrop(z_dim, nfeatures_modality2, act=nn.ReLU())

    def forward(self, x):
        x_1 = self.decoder1(x)
        x_2 = self.decoder2(x)
        x = torch.cat((x_1, x_2), 1)
        return x


class scMoMtF(nn.Module):
    def __init__(self, args, nfeatures_modality1=1, nfeatures_modality2=1, hidden_modality1=1, hidden_modality2=1, z_dim=64,
                 classify_dim=1):
        super(scMoMtF, self).__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.encoder = Encoder(args, self.nfeatures_modality1, self.nfeatures_modality2, hidden_modality1, hidden_modality2, z_dim)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder = Decoder(self.nfeatures_modality1, self.nfeatures_modality2, z_dim)
        args.discriminator['input'] = self.nfeatures_modality1 + self.nfeatures_modality2
        self.discriminator = MLP(args.discriminator)
        self.prob_layer = torch.nn.Softmax(dim=1)

    def forward(self, x):
        global mu
        global var
        h = self.encoder(x)
        x_res = self.decoder(h)
        x_r = self.discriminator(x)
        x_d = self.discriminator(x_res.detach())
        x_g = self.discriminator(x_res)
        x_cluster = self.classify(h)
        return x_res, h, x_r, x_d, x_g, x_cluster, mu, var
