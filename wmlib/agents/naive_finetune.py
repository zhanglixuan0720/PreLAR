import torch
from einops import rearrange
from . import expl
from .. import core, nets
from .actor_critic import ActorCritic
from .base import BaseAgent, BaseWorldModel


class Naive_Finetune(BaseAgent):

    def __init__(self, config, obs_space, act_space, step):
        super(Naive_Finetune, self).__init__(config, obs_space, act_space, step)

        self.wm = WorldModel(config, obs_space, act_space, self.step, self.enable_fp16)
        self._task_behavior = ActorCritic(config, self.act_space, self.step,self.enable_fp16)

        self.init_expl_behavior()
        # self.init_modules()

    def policy(self, obs, state=None, mode="train"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                if state is None:
                    # latent = self.wm.rssm.initial(len(obs["reward"]), obs["reward"].device)
                    af_latent = self.wm.af_rssm.initial(len(obs["reward"]), obs["reward"].device)
                    action = torch.zeros((len(obs["reward"]),) + self.act_space.shape).to(obs["reward"].device)
                    state = af_latent, action
                af_latent, action = state

                af_sample = (mode == "train") or not self.config.eval_state_mean
                if self.config.encoder_type == 'deco_resnet':
                    embed = self.wm.encoder(self.wm.preprocess(obs), eval=True)
                else:
                    embed = self.wm.encoder(self.wm.preprocess(obs))
                af_latent, _ = self.wm.af_rssm.obs_step(
                    af_latent, action, embed, obs["is_first"], af_sample
                )
                # af_embed = self.wm.af_rssm.get_feat(af_latent)
                # sample = (mode == "train") or not self.config.eval_state_mean
                # if self.config.concat_embed:
                #     af_embed = torch.cat([embed, af_embed], -1)
                # latent, _ = self.wm.rssm.obs_step(
                #     latent, action, af_embed, obs["is_first"], sample
                # )
                feat = self.wm.af_rssm.get_feat(af_latent)
                action = self.get_action(feat, mode)
                outputs = {"action": action.cpu()}
                state = (af_latent, action)

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
                    report[f"cond_aug_{self.config.encoder_deco.ctx_aug}"] = (self.wm.encoder.cond_aug(
                        data["image"][:1, 0]) + 0.5)[0].clamp(0.0, 1.0).detach().cpu().numpy()
                return report

    def init_optimizers(self):
        # wm_modules = [self.wm.rssm.parameters(), *[head.parameters() for head in self.wm.heads.values()]]
        # wm_enc_modules = [self.wm.encoder.parameters(), self.wm.af_rssm.parameters()]
        if self.config.finetune_rssm:
            wm_modules = [self.wm.af_rssm.parameters(), *[head.parameters() for head in self.wm.heads.values()]]
            wm_enc_modules = [self.wm.encoder.parameters()]
        else:
            wm_modules = [*[head.parameters() for head in self.wm.heads.values()]]
            wm_enc_modules = [self.wm.af_rssm.parameters(), self.wm.encoder.parameters()]

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

        self.af_rssm = nets.EnsembleRSSM(action_dim=act_space['action'].shape[0],**config.af_rssm)
        self.rssm = self.af_rssm
        # self.rssm = nets.EnsembleRSSM(action_dim=act_space['action'].shape[0],**config.rssm)

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
            self.heads["decoder"] = nets.DecoupledResNetDecoder()(shapes, **config.decoder, **config.decoder_deco)
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
            self.intr_bonus = expl.VideoIntrBonus(
                config.beta, config.k, config.intr_seq_length,
                config.af_rssm.deter + config.af_rssm.stoch * config.af_rssm.discrete,
                config.queue_dim,
                config.queue_size,
                config.intr_reward_norm,
                config.beta_type,
            )

        self.model_opt = core.EmptyOptimizer()
        self.enc_model_opt = core.EmptyOptimizer()
        self.enc_model_scheduler = None

        self.contextualized = self.config.encoder_type == 'deco_resnet'

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
        af_post, af_prior = self.af_rssm.observe(embed, data["action"], data["is_first"], state)
        # af_embed = self.af_rssm.get_feat(af_post)
        # af_feat = af_embed
        # if self.config.concat_embed:
        #     af_embed = torch.cat([embed, af_embed], -1)
        # post, prior = self.rssm.observe(af_embed, data["action"], data["is_first"], state)
        # kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        # assert len(kl_loss.shape) == 0
        af_kl_loss, af_kl_value = self.af_rssm.kl_loss(af_post, af_prior, **self.config.kl)
        # assert len(af_kl_loss.shape) == 0
        likes = {}
        losses = { "kl": af_kl_loss}

        feat = self.af_rssm.get_feat(af_post)

        plain_reward = data["reward"]
        if self.config.beta != 0:
            data, intr_rew_len, int_rew_mets = self.intr_bonus.compute_bonus(data, feat)

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
            embed=embed, feat=feat, post=af_post, prior=af_prior, likes=likes, kl=af_kl_value
        )
        metrics = {f"{name}_loss": value.detach().cpu() for name, value in losses.items()}
        # metrics["model_kl"] = kl_value.mean().item()
        metrics["embed_mean"] = embed.mean().item()
        metrics["embed_abs_mean"] = embed.abs().mean().item()
        metrics["af_model_kl"] = af_kl_value.mean().item()
        # metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean().item()
        # metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean().item()
        metrics["af_prior_ent"] = self.af_rssm.get_dist(af_prior).entropy().mean().item()
        metrics["af_post_ent"] = self.af_rssm.get_dist(af_post).entropy().mean().item()
        metrics["reward_target_mean"] = data['reward'].mean().item()
        metrics["reward_predict_mean"] = reward_predict.mean().item()
        metrics["reward_target_std"] = data['reward'].std().item()
        metrics["reward_predict_std"] = reward_predict.std().item()
        if self.config.decoder_type == 'deco_resnet' and self.heads['decoder']._current_attmask is not None:
            metrics["attmask"] = self.heads['decoder']._current_attmask
        if self.config.beta != 0:
            metrics.update(**int_rew_mets)
        last_state = {k: v[:, -1] for k, v in af_post.items()}
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
        # NOTE af_rssm observe all but then be cut in rssm.observe
        obs_len = 5
        # af_post, _ = self.af_rssm.observe(embed, data["action"], data["is_first"])
        states, _ = self.af_rssm.observe(embed[:6, :obs_len], data["action"][:6, :obs_len], data["is_first"][:6, :obs_len])
        # af_embed = self.af_rssm.get_feat(af_post)
        # if self.config.concat_embed:
        #     af_embed = torch.cat([embed, af_embed], -1)
        # states, _ = self.rssm.observe(
        #     af_embed[:6, :obs_len], data["action"][:6, :obs_len], data["is_first"][:6, :obs_len]
        # )
        if self.contextualized:
            recon = decoder(self.af_rssm.get_feat(states), shortcut_recon)[key].mode[:6]
        else:
            recon = decoder(self.af_rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.af_rssm.imagine(data["action"][:6, obs_len:], init)
        if self.contextualized:
            openl = decoder(self.af_rssm.get_feat(prior), shortcut_openl)[key].mode
        else:
            openl = decoder(self.af_rssm.get_feat(prior))[key].mode
        model = torch.cat([recon[:, :obs_len] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)
        B, T, C, H, W = video.shape
        return rearrange(video, 'b t c h w -> t c h (b w) ')
        # return video.permute((1, 3, 0, 4, 2)).reshape((T, H, B * W, C))
