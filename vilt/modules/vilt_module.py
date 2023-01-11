import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)


#another ViLT transformer class with bottleneck tokens 


class ViLTTransformerBN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() #specific to pytorch lightning 
        #print(self.hparams)

        self.config=config
        #bert config for the text part 
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #new addition 
        self.bottleneck_embeddings=nn.Embedding(config['num_bottleneck_tokens'],config['hidden_size']).to(config['device']) #not initiliazlied from original weights 
        

        #number of classes
        self.num_classes = config["num_classes"]

        # if self.hparams.config["load_path"] == "":
        #print(self.hparams.config["vit"])
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)
        #print(self.transformer)

        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)

        #not adding initializer here since iniitialization will be done using dictionary key wise 

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch["image"]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
       

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        #bottleneck embeds 
        bottleneck_masks=torch.ones(image_embeds.shape[0],self.hparams.config['num_bottleneck_tokens']).to(self.config['device'])
        bottleneck_embeds=self.bottleneck_embeddings(torch.arange(self.hparams.config['num_bottleneck_tokens']).to(self.config['device']).unsqueeze(0).repeat(image_embeds.shape[0],1)) #expand to (batch_size,num_bottleneck_tokens,hidden_size)

        #second operation : attention between image and bottleneck tokens
        co_image_bn_embeds = torch.cat([bottleneck_embeds, image_embeds], dim=1)
        co_image_bn_masks = torch.cat([bottleneck_masks, image_masks], dim=1)

        for i, blk in enumerate(self.transformer.blocks):
            co_image_bn_embeds, _attn = blk(co_image_bn_embeds, mask=co_image_bn_masks)

        co_image_bn_embeds = self.transformer.norm(co_image_bn_embeds) #layernorm of transformer on the image and bn embeddings
        bn_image_feats, image_feats = (
            co_image_bn_embeds[:, :bottleneck_embeds.shape[1]],
            co_image_bn_embeds[:, bottleneck_embeds.shape[1]:],
        )

        #third operation : attention between text and bottleneck tokens
        # print(text_embeds.shape,bottleneck_embeds.shape, image_embeds.shape, image_masks.shape, text_masks.shape, bottleneck_masks.shape)
        # print(image_masks)
        co_text_bn_embeds = torch.cat([text_embeds, bottleneck_embeds], dim=1)
        co_text_bn_masks = torch.cat([text_masks, bottleneck_masks], dim=1)

        for i, blk in enumerate(self.transformer.blocks):
            co_text_bn_embeds, _attn = blk(co_text_bn_embeds, mask=co_text_bn_masks)

        ### not using any token type embeddings for the bottleneck tokens 

        co_text_bn_embeds = self.transformer.norm(co_text_bn_embeds) #layernorm of transformer on the text and bn embeddings 
        text_feats, bn_text_feats = (
            co_text_bn_embeds[:, :text_embeds.shape[1]],
            co_text_bn_embeds[:, text_embeds.shape[1]:],
        )
        #cls_feats = self.pooler(co_text_bn_embeds)

        #pooler takes the first token of co_text_bn_embeds and passes it through set of linear layers
        #take average of the bn_feats 
        bn_text_feats_avg=bn_text_feats.mean(dim=1)
        bn_image_feats_avg=bn_image_feats.mean(dim=1)

        #average the text and image features #can be concatenated layer as well
        bn_feats=(bn_text_feats_avg+bn_image_feats_avg)/2

        #pass it to classifier 
        cls_logits=self.classifier(bn_feats)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits


#main difference from previous version is that we are using a single sequence for the text, image and bottleneck tokens
#we are using a single transformer for the entire sequence
# [text_embeds, image_embeds, bottleneck_embeds]
# [text_masks, image_masks, bottleneck_masks]

#first forward pass : attention between text and bottleneck tokens
#second forward pass : attention between image and bottleneck tokens

#first forward pass mask design: [mixture of 0s and 1s, all zeros, all ones]
#second forward pass mask design: [all zeros, mixture of 0s and 1s, all ones]

#take bottleneck embeddings for both cases 
#take average of bottleneck embeddings for both cases 
#possible combination: 1) concatenation 2) averaging

#add 3 types of token type embeddings or use pretrained for language and vision and add no token type embeddings for the bottleneck tokens

class ViLTTransformerBNv2(pl.LightningModule): 
    #in this version of implementation, there is no dynamic masking during dataloading 
    #hence there are two forward posses involved 
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() #specific to pytorch lightning 
        self.config=config

        #bert config for the text part 
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"], 
            max_position_embeddings=config["max_text_len"], #keep this max_text_len=40 to initialize the pretrained vilt weights
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #new addition 
        self.bottleneck_embeddings=nn.Embedding(config['num_bottleneck_tokens'],config['hidden_size']).to(config['device']) #not initiliazlied from original weights 
        
        #number of classes
        self.num_classes = config["num_classes"]

        # if self.hparams.config["load_path"] == "":
        #print(self.hparams.config["vit"])
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)
        #print(self.transformer)

        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(2*hs, hs), #hs -> 2*hs
                nn.LayerNorm(hs), 
                nn.GELU(),
                nn.Linear(hs, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)

        #not adding initializer here since iniitialization will be done using dictionary key wise 

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch["image"]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
       

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        #bottleneck embeds 
        bottleneck_masks=torch.ones(image_embeds.shape[0],self.hparams.config['num_bottleneck_tokens']).to(self.config['device']) #always ones
        bottleneck_embeds=self.bottleneck_embeddings(torch.arange(self.hparams.config['num_bottleneck_tokens']).to(self.config['device']).unsqueeze(0).repeat(image_embeds.shape[0],1)) #expand to (batch_size,num_bottleneck_tokens,hidden_size)

        #combined embeddings
        combined_embeds=torch.cat([text_embeds,image_embeds,bottleneck_embeds],dim=1)

        ####################################################### FIRST FORWARD PASS SECTION #########################################################
        #first operation (forward pass) consists of attention between image and bottleneck tokens 
        #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
        text_mask_fwd_pass_1=torch.zeros(image_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
        fwd_pass_1_mask=torch.cat([text_mask_fwd_pass_1,image_masks,bottleneck_masks],dim=1)
        for i, blk in enumerate(self.transformer.blocks):
            co_image_bn_embeds, _attn = blk(combined_embeds, mask=fwd_pass_1_mask)
        
        co_image_bn_embeds = self.transformer.norm(co_image_bn_embeds) #layernorm of transformer on the image and bn embeddings
        image_feats, bn_image_feats = (
            co_image_bn_embeds[:, text_embeds.shape[1]:text_embeds.shape[1]+image_embeds.shape[1]],
            co_image_bn_embeds[:, text_embeds.shape[1]+image_embeds.shape[1]:],
        )
        #print(image_feats.shape,bn_image_feats.shape)


        ####################################################### SECOND FORWARD PASS SECTION #########################################################
        #second operation (forward pass) consists of attention between text and bottleneck tokens 
        #so text masks would be mixture of 0s and 1s, bottleneck masks would be all ones and image masks would be all zeros 
        image_mask_max_len=image_masks.shape[1]
        image_mask_fwd_pass_2=torch.zeros(image_embeds.shape[0],image_mask_max_len).to(self.config['device'])
        fwd_pass_2_mask=torch.cat([text_masks,image_mask_fwd_pass_2,bottleneck_masks],dim=1)

        for i, blk in enumerate(self.transformer.blocks):
            co_text_bn_embeds, _attn = blk(combined_embeds, mask=fwd_pass_2_mask)
        
        co_text_bn_embeds = self.transformer.norm(co_text_bn_embeds) #layernorm of transformer on the image and bn embeddings
        text_feats, bn_text_feats = (
            co_text_bn_embeds[:, :text_embeds.shape[1]],
            co_text_bn_embeds[:, text_embeds.shape[1]+image_embeds.shape[1]:],
        )

        #pooler takes the first token of co_text_bn_embeds and passes it through set of linear layers
        #take average of the bn_feats 
        bn_text_feats_avg=bn_text_feats.mean(dim=1)
        bn_image_feats_avg=bn_image_feats.mean(dim=1)

        #for now pooling is used but later concatenation of the two can be done on the feature dimensions

        #average the text and image features #can be concatenated layer as well
        #bn_feats=(bn_text_feats_avg+bn_image_feats_avg)/2
        bn_feats=torch.cat([bn_text_feats_avg,bn_image_feats_avg],dim=1)

        #pass it to classifier 
        cls_logits=self.classifier(bn_feats)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits

















    
