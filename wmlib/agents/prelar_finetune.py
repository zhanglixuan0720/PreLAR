import torch
from einops import rearrange
from . import expl
from .. import core, nets
from .actor_critic import ActorCritic
from .base import BaseAgent, BaseWorldModel


class PreLARFinetune(BaseAgent):

    def __init__(self, config, obs_space, act_space, step):
        super(PreLARFinetune, self).__init__(config, obs_space, act_space, step)

        self.wm = WorldModel(config, obs_space, act_space, self.step, self.enable_fp16)
        self._task_behavior = ActorCritic(config, self.act_space, self.step,self.enable_fp16)

        self.init_expl_behavior()

    def policy(self, obs, state=None, mode="train"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                if state is None:
                    latent = self.wm.rssm.initial(len(obs["reward"]), obs["reward"].device)
                    action = torch.zeros((len(obs["reward"]),) + self.act_space.shape).to(obs["reward"].device)
                    state =  latent, action
                    if self.wm.action_encoder.twl > 1:
                        self.wm.action_encoder.action_buffer.clear()
                        self.wm.action_encoder.action_buffer.append(action)

                latent, action = state

                if self.config.encoder_type in ['deco_resnet']:
                    embed = self.wm.encoder(self.wm.preprocess(obs), is_eval=True)
                else:
                    embed = self.wm.encoder(self.wm.preprocess(obs))
            
                sample = (mode == "train") or not self.config.eval_state_mean
                if self.wm.action_encoder.twl > 1:
                    action = self.wm.action_encoder.action_buffer.get_actions()
                va_action = self.wm.action_encoder(action,sample)
                va_action = va_action['stoch'] if self.wm.action_encoder.type =='stoch' else va_action['deter']
                if self.wm.action_encoder.twl > 1:
                    va_action = va_action[:,-1]
                latent, _ = self.wm.rssm.obs_step(
                    latent, va_action, embed, obs["is_first"], sample
                )
                feat = self.wm.rssm.get_feat(latent)
                action = self.get_action(feat, mode)
                if self.wm.action_encoder.twl > 1:
                    self.wm.action_encoder.action_buffer.append(action)
                outputs = {"action": action.cpu()}
                state = ( latent, action)

        return outputs, state

    def report(self, data):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                report = {}
                data = self.wm.preprocess(data)
                for key in self.wm.heads["decoder"].cnn_keys:
                    name = key.replace("/", "_")
                    report[f"openl_{name}"] = self.wm.video_pred(data, key).detach().cpu().numpy()
                if self.wm.contextualized:
                    # show augmented context observation
                    report[f"cond_aug_{self.config.encoder_deco.ctx_aug}"] = (self.wm.encoder._deco_net['cond_aug'](
                        data["image"][:1, 0]) + 0.5)[0].clamp(0.0, 1.0).detach().cpu().numpy()
                return report

    def init_optimizers(self):
        if self.config.finetune_rssm:
            wm_modules = [self.wm.rssm.parameters(), self.wm.action_encoder.parameters(), *[head.parameters() for head in self.wm.heads.values()]]
            wm_enc_modules = [self.wm.encoder.parameters()]
        else:
            wm_modules = [self.wm.action_encoder.parameters(), *[head.parameters() for head in self.wm.heads.values()]]
            wm_enc_modules = [self.wm.rssm.parameters(), self.wm.encoder.parameters()]
        self.wm.enc_model_opt = core.Optimizer("enc_model", wm_enc_modules,enable_fp16=self.enable_fp16, **self.config.enc_model_opt)
        self.wm.model_opt = core.Optimizer("model", wm_modules,enable_fp16=self.enable_fp16, **self.config.model_opt)

        if self.config.enc_lr_type == "no_pretrain":
            self.wm.enc_model_scheduler = core.ConstantLR(
                self.wm.enc_model_opt.opt, factor=0., total_iters=self.config.pretrain)
        else:
            self.wm.enc_model_scheduler = None

        self._task_behavior.actor_opt = core.Optimizer("actor", self._task_behavior.actor.parameters(),
                                                       enable_fp16=self.enable_fp16,**self.config.actor_opt)
        self._task_behavior.critic_opt = core.Optimizer("critic", self._task_behavior.critic.parameters(),
                                                        enable_fp16=self.enable_fp16,**self.config.critic_opt)


class WorldModel(BaseWorldModel):

    def __init__(self, config, obs_space, act_space, step, enable_fp16 = False):
        super(WorldModel, self).__init__()

        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.step = step
        self.enable_fp16 = enable_fp16
        self.use_feat = config.use_feat
        
        self.action_encoder = nets.ActionEncoder(action_dim=act_space['action'].shape[0], **config.action_encoder)
        action_embed_dim = config.action_encoder.stoch * config.action_encoder.discrete if config.action_encoder.discrete else config.action_encoder.stoch 
        action_embed_dim = config.action_encoder.deter if (config.action_encoder.type_ == 'deter') else action_embed_dim
        self.rssm = nets.EnsembleRSSM(action_dim=action_embed_dim,**config.rssm)
        self.af_rssm = self.rssm # alias

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
            self.heads["decoder"] = nets.PlainCNNDecoder(shapes, **config.decoder)
        elif self.config.decoder_type == 'resnet':
            self.heads["decoder"] = nets.ResNetDecoder(shapes, **config.decoder)
        elif self.config.decoder_type == 'deco_resnet':
            self.heads['decoder'] = nets.DecoupledResNetDecoder(shapes, **config.decoder, **config.decoder_deco)
        else:
            raise NotImplementedError
        self.heads["reward"] = nets.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads["discount"] = nets.MLP([], **config.discount_head)
        if config.loss_scales.get("aux_reward", 0.0) != 0:
            self.heads["aux_reward"] = nets.MLP([], **config.reward_head)
        for name in config.grad_heads:
            assert name in self.heads, name

        if self.config.beta != 0:
            intr_embed_dim = config.rssm.deter + config.rssm.stoch * config.rssm.discrete if self.use_feat else config.rssm.embed_dim
            self.intr_bonus = expl.VideoIntrBonus(
                config.beta, config.k, config.intr_seq_length,
                intr_embed_dim,
                config.queue_dim,
                config.queue_size,
                config.intr_reward_norm,
                config.beta_type,
            )

        self.model_opt = core.EmptyOptimizer()
        self.enc_model_opt = core.EmptyOptimizer()
        self.enc_model_scheduler = None

        self.contextualized = self.config.encoder_type in ['deco_resnet']


    def train_iter(self, data, state=None):
        with torch.cuda.amp.autocast(enabled=self.enable_fp16):
            self.zero_grad(set_to_none=True)  # delete grads
            model_loss, state, outputs, metrics = self.loss(data, state)

        # Backward passes under autocast are not recommended.
        self.model_opt.backward(model_loss)
        metrics.update(self.enc_model_opt.step(model_loss, external_scaler=self.model_opt.scaler))
        metrics.update(self.model_opt.step(model_loss))
        metrics["model_loss"] = model_loss.item()
        if self.enc_model_scheduler is not None:
            self.enc_model_scheduler.step()

        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
        else:
            embed = self.encoder(data)
            shortcut = None
        va_action = self.action_encoder(data["action"])
        va_action = va_action['stoch'] if self.action_encoder.type =='stoch' else va_action['deter']
        post, prior = self.rssm.observe(embed, va_action, data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        likes = {}
        losses = {"kl": kl_loss}

        feat = self.rssm.get_feat(post)

        plain_reward = data["reward"]
        if self.config.beta != 0:
            intr_embed = feat if self.use_feat else embed
            data, intr_rew_len, int_rew_mets = self.intr_bonus.compute_bonus(data, intr_embed)

        for name, head in self.heads.items():
            if name == "aux_reward":
                continue
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else feat.detach()
            if name == "reward" and self.config.beta != 0:
                inp = inp[:, :intr_rew_len]
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
                if key == 'reward':
                    reward_predict = dist.mode

        if self.config.loss_scales.get("aux_reward", 0.0) != 0:
            head = self.heads["aux_reward"]
            dist = head(feat)
            like = dist.log_prob(plain_reward)
            losses["aux_reward"] = -like.mean()

        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value.detach().cpu() for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean().item()
        metrics["embed_mean"] = embed.mean().item()
        metrics["embed_abs_mean"] = embed.abs().mean().item()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean().item()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean().item()
        metrics["reward_target_mean"] = data['reward'].mean().item()
        metrics["reward_predict_mean"] = reward_predict.mean().item()
        metrics["reward_target_std"] = data['reward'].std().item()
        metrics["reward_predict_std"] = reward_predict.std().item()
        if self.config.decoder_type == 'deco_resnet' and self.heads['decoder']._current_attmask is not None:
            metrics["attmask"] = self.heads['decoder']._current_attmask
        if self.config.beta != 0:
            metrics.update(**int_rew_mets)
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        if self.contextualized:
            output = self.encoder(data)
            embed, shortcut = output['embed'], output['shortcut']
            shortcut_recon = {k: v[:6] for k, v in shortcut.items()}
            shortcut_openl = shortcut_recon
        else:
            embed = self.encoder(data)
            shortcut_recon, shortcut_openl = None, None
        va_action = self.action_encoder(data["action"])
        va_action = va_action['stoch'] if self.action_encoder.type =='stoch' else va_action['deter']
        obs_len = 5
        states, _ = self.rssm.observe(
            embed[:6, :obs_len], va_action[:6, :obs_len], data["is_first"][:6, :obs_len]
        )
        if self.contextualized:
            recon = decoder(self.rssm.get_feat(states), shortcut_recon)[key].mode[:6]
        else:
            recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(va_action[:6, obs_len:], init)
        if self.contextualized:
            openl = decoder(self.rssm.get_feat(prior), shortcut_openl)[key].mode
        else:
            openl = decoder(self.rssm.get_feat(prior))[key].mode
        model = torch.cat([recon[:, :obs_len] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)
        return rearrange(video, 'b t c h w -> t c h (b w) ')
    
    def imagine(self, policy, start, is_terminal, horizon):
        b,t = start['deter'].shape[0], start['deter'].shape[1]
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        recover = lambda x: x.reshape([b, t] + list(x.shape[1:]))
        start = {k: flatten(v) for k, v in start.items()}
        if self.config.imag_batch != -1:
            index = torch.randperm(len(start["deter"]), device=start["deter"].device)[:self.config.imag_batch]
            select = lambda x: torch.index_select(x, dim=0, index=index)
            start = {k: select(v) for k, v in start.items()}
        start["feat"] = self.rssm.get_feat(start)
        start["action"] = torch.zeros_like(policy(start["feat"]).mode)
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq["feat"][-1].detach()).sample()
            recover_action = recover(action)
            va_action = self.action_encoder(recover_action)
            va_action = va_action['stoch'] if self.action_encoder.type =='stoch' else va_action['deter']
            va_action = flatten(va_action)
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, va_action) #TODO
            feat = self.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean
            if is_terminal is not None:
                true_first = 1.0 - flatten(is_terminal).to(disc.dtype)
                true_first *= self.config.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * torch.ones(seq["feat"].shape[:-1]).to(seq["feat"].device)
        seq["discount"] = disc
        seq["weight"] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq

