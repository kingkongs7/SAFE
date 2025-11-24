import os, multiprocessing as mp
from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from peft import (
    LoraConfig, AdaLoraConfig, LoHaConfig, IA3Config, LNTuningConfig,
    get_peft_model)

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC


def _make_adapter(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Wrap `model` with the requested PEFT adapter."""
    t = cfg.get("peft_type", "lora").lower()
    tgt = ["q_proj", "v_proj"]
    ffn = ["fc1", "fc2"]
    task_str = "IMAGE_CLASSIFICATION"

    if t == "lora":
        pcfg = LoraConfig(r=cfg.get("r", 8), lora_alpha=cfg.get("alpha", 16),
                          target_modules=tgt,
                          modules_to_save=["visual_projection"],
                          task_type=task_str)
    elif t == "adalora":
        pcfg = AdaLoraConfig(init_r=cfg.get("r", 8), target_r=cfg.get("target_r", 2),
                             lora_alpha=16, tinit=200, beta1=0.85, beta2=0.85,
                             target_modules=tgt, task_type=task_str)
    elif t == "loha":
        pcfg = LoHaConfig(r=cfg.get("r", 4), lora_alpha=cfg.get("alpha", 8),
                          target_modules=tgt, task_type=task_str)
    elif t == "ia3":
        pcfg = IA3Config(target_modules=tgt + ffn, feedforward_modules=ffn)
    elif t == "ln":
        pcfg = LNTuningConfig(task_type=task_str,
                              target_modules=[r"layer_norm\d*", r"post_layernorm"],
                              modules_to_save=["visual_projection"])
    else:
        raise ValueError(f"Unsupported peft_type: {t}")

    wrapped = get_peft_model(model, pcfg)
    wrapped.print_trainable_parameters()
    return wrapped

@torch.no_grad()
def _embed_text(texts: List[str], tokenizer: CLIPTokenizer, clip: CLIPModel) -> torch.Tensor:
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(clip.device)
    return clip.get_text_features(**toks)

@DETECTOR.register_module(module_name="clip_aaai_f2")
class CLIPDeepFakeDetector(AbstractDetector):
    STATIC_PROMPTS = ["a photo of a real face", "a photo of a fake face"]
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg  = cfg
        self.mode = cfg.get("train_mode", "cls").lower()
        assert self.mode in ("prompt", "cls", "dual")

        # backbone & adapter
        self.clip = CLIPModel.from_pretrained(cfg["model_weight"])
        self.clip.vision_model = self.build_backbone(cfg)
        self.d = self.clip.config.projection_dim

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg["model_weight"])

        # classifier head
        txt_emb = _embed_text(self.STATIC_PROMPTS, self.tokenizer, self.clip)
        self.register_buffer("static_txt", txt_emb)

        if self.mode in ("cls", "dual"):
            self.cls_head = nn.Linear(self.d, 2)
        else:
            self.register_parameter("cls_head", None)

        self.lambda_clip  = float(cfg.get("lambda_clip", 0.6))
        self.loss_fn = self.build_loss(cfg)

    def build_backbone(self, cfg):
        return _make_adapter(self.clip.vision_model, cfg)

    def build_loss(self, cfg):
        return LOSSFUNC[cfg["loss_func"]]()

    def features(self, data_dict):
        return self.clip.get_image_features(data_dict["image"])

    def classifier(self, features):
        if self.mode == "prompt":
            img_n = features / features.norm(dim=-1, keepdim=True)
            txt_n = self.static_txt / self.static_txt.norm(dim=-1, keepdim=True)
            return img_n @ txt_n.T * self.clip.logit_scale.exp()
        return self.cls_head(features)

    def _txt_emb(self, data_dict):
        if "text_label" in data_dict.keys() and data_dict["text_label"] is not None:
            txts = data_dict["text_label"]
            if isinstance(txts, torch.Tensor):
                return self.clip.get_text_features(input_ids=txts.to(self.clip.device))
            return _embed_text(list(txts), self.tokenizer, self.clip)
        return self.static_txt.to(self.clip.device)


    def _clip_logits(self, img_emb, txt_emb):
        try:
            img_n = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_n = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            return img_n @ txt_n.T * self.clip.logit_scale.exp()
        except:
            print(img_emb.shape)
            print(img_emb.shape)

    def forward(self, data_dict: Dict[str, torch.Tensor], inference=False):
        img_feat = self.features(data_dict)
        out: Dict[str, torch.Tensor] = {"feat": img_feat}

        if self.mode == "cls":
            logits = self.classifier(img_feat)
            out["cls"] = logits
        else:
            txt_emb = self._txt_emb(data_dict)
            logits_clip = self._clip_logits(img_feat, txt_emb)
            out["clip"] = logits_clip
            out["clip_t2i"] = logits_clip.T
            if self.mode == "dual":
                logits_cls = self.classifier(img_feat)
                out["cls"] = logits_cls
            else:
                out["cls"] = logits_clip

        out["prob"] = torch.softmax(out["cls"], dim=-1)[:, -1]
        return out

    def get_losses(self, data_dict, pred):
        y = data_dict["label"]
        B = y.size(0)
        ce = self.loss_fn

        if self.mode == "cls":
            loss = self.loss_fn(pred["cls"], y)
        elif self.mode == "prompt":
            tgt = torch.arange(B, device=y.device)
            loss_img = ce(pred["clip"],      tgt)   # image → text
            loss_txt = ce(pred["clip_t2i"],  tgt)   # text  → image
            loss = (loss_img + loss_txt) / 2        # 
        else:
            tgt = torch.arange(B, device=y.device)
            loss_img = ce(pred["clip"],     tgt)
            loss_txt = ce(pred["clip_t2i"], tgt)
            loss_clip = (loss_img + loss_txt) / 2
            loss_cls  = ce(pred["cls"], y)
            # print("loss_clip: {} loss_cls: {}".format(loss_clip.item(), loss_cls.item()))
            loss = self.lambda_clip * loss_clip + (1 - self.lambda_clip) * loss_cls
        return {"overall": loss}

    def get_train_metrics(self, data_dict, pred):
        y = data_dict["label"].detach()
        logits = pred["cls"].detach()
        auc, eer, acc, ap = calculate_metrics_for_train(y, logits)
        return {"auc": auc, "eer": eer, "acc": acc, "ap": ap}
