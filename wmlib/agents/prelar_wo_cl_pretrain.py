import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .. import core, nets


class PreLARwoCLPretrain(nn.Module):

    def __init__(self, config, obs_space, act_space, step):
        super(PreLARwoCLPretrain, self).__init__()

        self.config = config
        self.precision = config.precision
        self.enable_fp16 = self.precision == 16
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = WorldModel(config, obs_space, self.step,self.enable_fp16)

    def train(self, data, state=None):
        metrics = {}
        self.wm.train()
        state, outputs, wm_metrics = self.wm.train_iter(data, state)
        self.wm.eval()
        metrics.update({f'wm/{k}':v for k, v in wm_metrics.items()})
        return core.dict_detach(state), metrics

    def eval(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.eval_iter(data, state)
        metrics.update(mets)
        return core.dict_detach(state), metrics

    def report(self, data, recon=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                report = {}
                data = self.wm.preprocess(data)
                for key in self.wm.heads['decoder'].cnn_keys:
                    name = key.replace('/', '_')
                    video_pred, report_metrics = self.wm.video_pred(data, key)
                    report[f'openl_{name}'] = video_pred.detach().cpu().numpy()
                    for k, v in report_metrics.items():
                        report[k] = v
                    if recon:
                        report[f'recon_{name}'] = self.wm.video_recon(data, key).detach().cpu().numpy()
                return report

    def init_optimizers(self):
        wm_modules = [self.wm.encoder.parameters(), self.wm.rssm.parameters(),self.wm.va_encoder.parameters(),
                      *[head.parameters() for head in self.wm.heads.values()]]
        self.wm.model_opt = core.Optimizer('model', wm_modules, enable_fp16=self.enable_fp16,**self.config.model_opt)

    def save_model(self, logdir, suffix=''):
        torch.save(self.state_dict(), logdir / f'variables{suffix}.pt')
        torch.save(self.wm.rssm.state_dict(), logdir / f'rssm_variables{suffix}.pt')
        torch.save(self.wm.va_encoder.state_dict(), logdir / f'va_encoder_variables{suffix}.pt')
        torch.save(self.wm.encoder.state_dict(), logdir / f'encoder_variables{suffix}.pt')
        torch.save(self.wm.heads['decoder'].state_dict(), logdir / f'decoder_variables{suffix}.pt')


class WorldModel(nn.Module):

    def __init__(self, config, obs_space, step, enable_fp16=False):
        super(WorldModel, self).__init__()

        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.enable_fp16 = enable_fp16
        self.step = step
        
        action_dim = config.vanet.stoch * config.vanet.discrete if config.vanet.discrete else config.vanet.stoch
        action_dim = config.vanet.deter if (config.vanet.type_ == 'deter') else action_dim
        self.rssm = nets.EnsembleRSSM(action_dim=action_dim, **config.rssm)

        self.va_encoder = nets.VANet(shapes,**config.vanet)
        

        if self.config.encoder_type == 'plaincnn':
            self.encoder = nets.PlainCNNEncoder(shapes, **config.encoder)
        elif self.config.encoder_type == 'resnet':
            self.encoder = nets.ResNetEncoder(shapes, **config.encoder)
        elif self.config.encoder_type == 'deco_resnet':
            self.encoder = nets.DecoupledResNetEncoder(shapes, **config.encoder, **config.encoder_deco)
        else:
            raise NotImplementedError

        self.heads = torch.nn.ModuleDict()

        if self.config.decoder_type == 'plaincnn':
            self.heads['decoder'] = nets.PlainCNNDecoder(shapes, **config.decoder)
        elif self.config.decoder_type == 'resnet':
            self.heads['decoder'] = nets.ResNetDecoder(shapes, **config.decoder)
        elif self.config.decoder_type == 'deco_resnet':
            self.heads['decoder'] = nets.DecoupledResNetDecoder(shapes, **config.decoder, **config.decoder_deco)
        else:
            raise NotImplementedError
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = core.EmptyOptimizer()

        self.contextualized = self.config.encoder_type == 'deco_resnet' or self.config.decoder_type == 'deco_resnet'

    def train_iter(self, data, state=None):
        with torch.cuda.amp.autocast(enabled=self.enable_fp16):
            self.zero_grad(set_to_none=True)  # delete grads
            model_loss, state, outputs, metrics = self.loss(data, state)

        # Backward passes under autocast are not recommended.
        self.model_opt.backward(model_loss)
        metrics.update(self.model_opt.step(model_loss))
        metrics['model_loss'] = model_loss.item()

        return state, outputs, metrics

    def eval_iter(self, data, state=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                model_loss, state, outputs, metrics = self.loss(data, state)
        metrics['model_loss'] = model_loss.item()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
        else:
            embed = self.encoder(data)
        va_action_state, va_action = self.va_encode(data)
        post, prior = self.rssm.observe(embed, va_action, data['is_first'], state)

        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        va_kl_loss, va_kl_value = self.va_encoder.kl_loss(va_action_state)
        
        likes = {}
        losses = {'kl': kl_loss, 'va_kl': va_kl_loss}

        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else feat.detach()
            if name == 'decoder' and self.contextualized:
                out = head(inp, shortcut)
            else:
                out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                # NOTE: for bernoulli log_prob with float values (data["discount"]) means binary_cross_entropy_with_logits
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()

        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f'{name}_loss': value.detach().cpu() for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean().item()
        metrics['action_kl'] = va_kl_value.mean().item()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean().item()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean().item()
        metrics['embed_norm'] = embed.pow(2).mean().item()
        metrics['embed_mean'] = embed.mean().item()
        metrics['embed_std'] = embed.std().item()
        metrics['deter_norm'] = post['deter'].pow(2).mean().item()

        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_'):
                continue
            if value.dtype == torch.int32:
                value = value.float()
            if value.dtype == torch.uint8:
                # value = value.float() / 255.0 - 0.5
                value = value.float()
            obs[key] = value
        obs['image'] = obs['image'] / 255.0 - 0.5
        if obs['image'].shape[-3] == 6 :
            obs['image'], obs['flow'] =  torch.chunk(obs['image'],2,dim=-3)
        return obs
    
    def va_encode(self, data):
        if self.va_encoder._va_method == 'flow':
            va_action_state = self.va_encoder(None,data['flow'][:,1:])
        else:
            va_action_state = self.va_encoder(data['image'][:,:-1],data['image'][:,1:])
        if self.va_encoder.type == 'mix':
            va_action =  torch.cat([va_action_state['deter'],va_action_state['stoch']],-1)
        else:
            va_action = va_action_state[self.va_encoder.type]
        return va_action_state, self.va_encoder.append_action(va_action, ahead=True)
    
    def video_pred(self, data, key):
        decoder = self.heads['decoder']
        truth = data[key][:6] + 0.5
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
            shortcut = {k: v[:6] for k, v in shortcut.items()}
        else:
            embed = self.encoder(data)
            shortcut = None
        va_action_state, va_action = self.va_encode(data)
        obs_len = 5
        states, _ = self.rssm.observe(
            embed[:6, :obs_len], va_action[:6, :obs_len], data['is_first'][:6, :obs_len]
        )
        if self.contextualized:
            recon = decoder(self.rssm.get_feat(states), shortcut)[key].mode[:6]
        else:
            recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(va_action[:6, obs_len:], init)
        if self.contextualized:
            openl = decoder(self.rssm.get_feat(prior), shortcut)[key].mode
        else:
            openl = decoder(self.rssm.get_feat(prior))[key].mode
        model = torch.cat([recon[:, :obs_len] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)

        report = {}
        report['post_logit_diff'] = (F.softmax(states['logit'][:, 1:], -1) -
                                     F.softmax(states['logit'][:, :-1], -1)).pow(2).mean().item()
        report['prior_logit_diff'] = (F.softmax(prior['logit'][:, 1:], -1) -
                                      F.softmax(prior['logit'][:, :-1], -1)).pow(2).mean().item()
        report['post_deter_diff'] = (states['deter'][:, 1:] - states['deter'][:, :-1]).pow(2).mean().item()
        report['prior_deter_diff'] = (prior['deter'][:, 1:] - prior['deter'][:, :-1]).pow(2).mean().item()

        return rearrange(video, 'b t c h w -> t c h (b w) '), report

    def video_pred_noise(self, data, key):
        decoder = self.heads['decoder']
        truth = data[key][:6] + 0.5
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
            shortcut = {k: v[:6] for k, v in shortcut.items()}
        else:
            embed = self.encoder(data)
            shortcut = None
        va_action_state, va_action = self.va_encode(data)
        obs_len = 5
        states, _ = self.rssm.observe(
            embed[:6, :obs_len], va_action[:6, :obs_len], data['is_first'][:6, :obs_len]
        )
        if self.contextualized:
            recon = decoder(self.rssm.get_feat(states), shortcut)[key].mode[:6]
        else:
            recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(va_action[:6, obs_len:], init)
        if self.contextualized:
            openl = decoder(self.rssm.get_feat(prior), shortcut)[key].mode
        else:
            openl = decoder(self.rssm.get_feat(prior))[key].mode
        model = torch.cat([recon[:, :obs_len] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)

        report = {}
        report['post_logit_diff'] = (F.softmax(states['logit'][:, 1:], -1) -
                                     F.softmax(states['logit'][:, :-1], -1)).pow(2).mean().item()
        report['prior_logit_diff'] = (F.softmax(prior['logit'][:, 1:], -1) -
                                      F.softmax(prior['logit'][:, :-1], -1)).pow(2).mean().item()
        report['post_deter_diff'] = (states['deter'][:, 1:] - states['deter'][:, :-1]).pow(2).mean().item()
        report['prior_deter_diff'] = (prior['deter'][:, 1:] - prior['deter'][:, :-1]).pow(2).mean().item()

        return rearrange(video, 'b t c h w -> t c h (b w) '), report

    def video_recon(self, data, key):
        decoder = self.heads['decoder']
        truth = data[key][:6] + 0.5
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
            shortcut = {k: v[:6] for k, v in shortcut.items()}
        else:
            embed = self.encoder(data)
            shortcut = None
        va_action_state, va_action = self.va_encode(data)
        states, _ = self.rssm.observe(embed[:6], va_action[:6], data['is_first'][:6])
        if self.contextualized:
            recon = decoder(self.rssm.get_feat(states), shortcut)[key].mode[:6]
        else:
            recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        model = recon + 0.5
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)
        return rearrange(video, 'b t c h w -> t c h (b w) ')