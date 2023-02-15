import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from operator import mul
import math

########################################## original vilt implementation with SSL objective##########################################
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


#another ViLT transformer class with bottleneck tokens ---- need to revisit this ------
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

class ViLTTransformerBNv2(pl.LightningModule):  ####### uses bottleneck token representations for downstream tasks #############
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
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #new addition 
        self.bottleneck_embeddings=nn.Embedding(config['num_bottleneck_tokens'],config['hidden_size']).to(config['device']) #not initiliazlied from original weights 
        
        #number of classes
        self.num_classes = config["num_classes"]
        self.patch_size = self.hparams.config["patch_size"]
        # if self.hparams.config["load_path"] == "":
        #print(self.hparams.config["vit"])
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)
        #print(self.transformer)

        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(hs, hs), #hs -> 2*hs
                nn.LayerNorm(hs), 
                nn.GELU(),
                nn.Linear(hs, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)

        #not adding initializer here since iniitialization will be done using dictionary key wise 

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag

        text_embeds = self.text_embeddings(text_ids)

        #if image_embeds is None and image_masks is None:
        img = batch["image"]
        #print(img)
        
        #bottleneck embeds 
        bottleneck_masks=torch.ones(text_embeds.shape[0],self.hparams.config['num_bottleneck_tokens']).to(self.config['device']) #always ones
        bottleneck_embeds=self.bottleneck_embeddings(torch.arange(self.hparams.config['num_bottleneck_tokens']).to(self.config['device']).unsqueeze(0).repeat(text_embeds.shape[0],1)) #expand to (batch_size,num_bottleneck_tokens,hidden_size)
        if(self.mod_dropout_flag==False): #no modality dropout .... no need to use modality dropout mask
            
            #print(img)
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
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )
            #combined embeddings
            combined_embeds=torch.cat([text_embeds,image_embeds,bottleneck_embeds],dim=1)
            #bottleneck embeds 
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

            bn_feats=(bn_image_feats+bn_text_feats)/2 #average of the bn feats from both the forward passes

        else: #modality dropout here 

            #two possible pathways in the forward pass 
            # (1) where modality dropout choice is image and complete modality is text: 
                # run the first forward pass with filtered image samples and then run the second forward pass with complete text samples
            # (2) where modality dropout choice is text and complete modality is image:
                # run the first forward pass with complete image samples and then run the second forward pass with filtered text samples
            
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped

            #filtered segment containing the missing modality samples 
            mmod_images=img[index_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[index_modality_dropped,:] 
            mmod_bottleneck_embeds=bottleneck_embeds[index_modality_dropped,:]
            mmod_text_masks=text_masks[index_modality_dropped,:]
            mmod_bottleneck_masks=bottleneck_masks[index_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[index_complete_modality,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[index_complete_modality,:]
            compl_bottleneck_embeds=bottleneck_embeds[index_complete_modality,:]
            compl_text_masks=text_masks[index_complete_modality,:]
            compl_bottleneck_masks=bottleneck_masks[index_complete_modality,:]

            #create a dummy tensor for the bottleneck tokens embeddings
            bn_feats=torch.randn(text_embeds.shape[0],
                        self.hparams.config['num_bottleneck_tokens'],
                        self.hparams.config['hidden_size']).to(self.config['device'])


            #print(compl_images.shape,mmod_images.shape)

            #idea is to inject the mmod_bottleneck_embeds (passed through vilt) into the indices i.e. index_modality_dropped  
            #compl_bottleneck_embeds (passed through vilt) into the indices i.e. index_complete_modality
            #print("bn feats before:",bn_feats)

            if(self.mod_choice=='image'): #modality dropout choice is image and complete modality is text
                
                ####################################### ATTENTION BETWEEN TEXT AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                ## perform attention between text and bottleneck tokens for the missing modality samples
                #mask for image tokens would be zeros and text would be normal masks and bottleneck tokens would be all ones

                ####################################### creating the dummy tensort for the image tokens ########################################
                #maximum height of the images from the missing modality batches
                max_height= max([img.shape[1] for img in mmod_images])
                #maximum width of the images from the missing modality batches
                max_width= max([img.shape[2] for img in mmod_images])

                #num width patches
                num_width_patches=(max_width//self.hparams.config['patch_size'])
                #num height patches
                num_height_patches=(max_height//self.hparams.config['patch_size'])

                #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
                image_mmod_mask_len=num_height_patches*num_width_patches
                mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hparams.config['hidden_size']).to(self.config['device'])

                ####################################### creating the dummy tensort for the image masks ########################################
                #make zero tensors of (batch size, num_height_patches x num_width_patches)
                image_mmod_mask_fwd_pass=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
                #print(image_mmod_mask_fwd_pass)

                #include to0ken type embeddings for the image tokens
                mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                )

                
                combined_img_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)


                image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len).to(self.config['device'])
                total_mmod_mask_fwd_pass_image=torch.cat([mmod_text_masks,image_mmod_mask_fwd_pass,mmod_bottleneck_masks],dim=1)

                #forward pass for text and bottleneck for the missing modality samples
                for i, blk in enumerate(self.transformer.blocks):
                    co_img_mmod_bn_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
                
                co_img_mmod_bn_embeds = self.transformer.norm(co_img_mmod_bn_embeds)
                #extract the bottleneck embeddings from co_img_mmod_bn_embeds
                mmod_bn_feats = co_img_mmod_bn_embeds[:, mmod_text_embeds.shape[1]+mmod_image_embeds.shape[1]:]
                #(this has a shape (B',N,H) where B' is the number of missing modality samples and N is the number of bottleneck tokens and H is the hidden size)


            elif(self.mod_choice=='text'):
                ####################################### ATTENTION BETWEEN IMAGE AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                ## perform attention between image and bottleneck tokens for the missing modality samples
                #mask for text tokens would be zeros and image would be normal masks and bottleneck tokens would be all ones

                #pass the mmod images through visual embed encoder 
                (
                    mmod_image_embeds,
                    mmod_image_masks,
                    patch_index,
                    image_labels,
                ) = self.transformer.visual_embed(
                    mmod_images,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image
                )

                #include to0ken type embeddings for the text and image tokens
                mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
                )

                combined_text_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)
                text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
                total_mmod_mask_fwd_pass_text=torch.cat([text_mmod_mask_fwd_pass,mmod_image_masks,mmod_bottleneck_masks],dim=1)

                #forward pass for image and bottleneck for the missing modality samples
                for i, blk in enumerate(self.transformer.blocks):
                    co_text_mmod_bn_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

                co_text_mmod_bn_embeds = self.transformer.norm(co_text_mmod_bn_embeds)

                #extract the bottleneck embeddings from co_text_mmod_bn_embeds
                mmod_bn_feats = co_text_mmod_bn_embeds[:, mmod_text_embeds.shape[1]+mmod_image_embeds.shape[1]:]
                #(this has a shape (B',N,H) where B' is the number of missing modality samples and N is the number of bottleneck tokens and H is the hidden size)

            ####################################### ATTENTION BETWEEN (1) IMAGE (2) TEXT AND BOTTLENECK TOKENS FOR COMPLETE MODALITY SAMPLES ########################################
            #pass the completed images through visual embed encoder
            (
                    compl_image_embeds,
                    compl_image_masks,
                    patch_index,
                    image_labels,
            ) = self.transformer.visual_embed(
                    compl_images,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image
            )

            #include token type embeddings for the text and image tokens
            compl_text_embeds, compl_image_embeds = (

                compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
            )

            ## perform two stage attention operation between 
            # (1) image and bottleneck tokens (2) text and bottleneck tokens for complete modality samples
            combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds,compl_bottleneck_embeds],dim=1)

            ####################################################### FIRST FORWARD PASS SECTION #########################################################
            #first operation (forward pass) consists of attention between image and bottleneck tokens 
            #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
            text_mask_fwd_pass_1_compl=torch.zeros(compl_image_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            fwd_pass_1_mask_compl=torch.cat([text_mask_fwd_pass_1_compl,compl_image_masks,compl_bottleneck_masks],dim=1)
            for i, blk in enumerate(self.transformer.blocks):
                co_image_bn_embeds_compl, _attn = blk(combined_embeds_compl, mask=fwd_pass_1_mask_compl)
            
            co_image_bn_embeds_compl = self.transformer.norm(co_image_bn_embeds_compl) #layernorm of transformer on the image and bn embeddings
            bn_image_feats_compl=co_image_bn_embeds_compl[:, compl_text_embeds.shape[1]+compl_image_embeds.shape[1]:] #first part of the bottleneck tokens 
            #(this has a shape (B",N,H) where B" is the number of complete modality samples and N is the number of bottleneck tokens and H is the hidden size)


            ####################################################### SECOND FORWARD PASS SECTION #########################################################
            #second operation (forward pass) consists of attention between text and bottleneck tokens 
            #so text masks would be mixture of 0s and 1s, bottleneck masks would be all ones and image masks would be all zeros 
            image_mask_max_len=compl_image_masks.shape[1]
            image_mask_fwd_pass_2=torch.zeros(compl_image_embeds.shape[0],image_mask_max_len).to(self.config['device'])
            fwd_pass_2_mask_compl=torch.cat([compl_text_masks,image_mask_fwd_pass_2,compl_bottleneck_masks],dim=1)
            for i, blk in enumerate(self.transformer.blocks):
                co_text_bn_embeds_compl, _attn = blk(combined_embeds_compl, mask=fwd_pass_2_mask_compl)
            
            co_text_bn_embeds_compl = self.transformer.norm(co_text_bn_embeds_compl) #layernorm of transformer on the image and bn embeddings
            bn_text_feats_compl = co_text_bn_embeds_compl[:, compl_text_embeds.shape[1]+compl_image_embeds.shape[1]:]
            #(this has a shape (B",N,H) where B" is the number of complete modality samples and N is the number of bottleneck tokens and H is the hidden size)

            #average the text and image features for the complete modality samples
            bn_feats_compl=(bn_text_feats_compl+bn_image_feats_compl)/2
            #insert the missing modality bottleneck features and complete modality features into respective locations inside the feature matrix
            #this is done to maintain the order of the samples in the feature matrix

            #first insert the missing modality features
            id_modality_dropped=index_modality_dropped.tolist()
            id_complete_modality=index_complete_modality.tolist()
            bn_feats[id_modality_dropped,:]=mmod_bn_feats
            bn_feats[id_complete_modality,:]=bn_feats_compl


            #print("bn_feats after:",bn_feats)



        # #pooler takes the first token of co_text_bn_embeds and passes it through set of linear layers
        # #take average of the bn_feats 
        # bn_text_feats_avg=bn_text_feats.mean(dim=1)
        # bn_image_feats_avg=bn_image_feats.mean(dim=1)

        # #for now pooling is used but later concatenation of the two can be done on the feature dimensions
        bn_avg_feats=bn_feats.mean(dim=1)
        
        #average the text and image features #can be concatenated layer as well
        #bn_feats=(bn_text_feats_avg+bn_image_feats_avg)/2
        #bn_feats=torch.cat([bn_text_feats_avg,bn_image_feats_avg],dim=1)

        #pass it to classifier 
        cls_logits=self.classifier(bn_avg_feats)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)
        return cls_logits


########################################################### VERSION UTILIZES VILT WITH NORMAL MASKING WITHOUT BOTTLENECK TOKENS TO GENERATE THE FINAL PREDICTIONS ###########################################################
class ViLTTransformer_token_aggregate(pl.LightningModule): #uilize the token information from visual and text modalities instead of bottleneck tokens for final predictions
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
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #number of classes
        self.num_classes = config["num_classes"]
        self.patch_size = self.hparams.config["patch_size"]
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)
       
        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(hs, hs), #hs -> 2*hs
                nn.LayerNorm(hs), 
                nn.GELU(),
                nn.Linear(hs, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)

        #not adding initializer here since iniitialization will be done using dictionary key wise 

    def handle_missing_image_modality(self, mmod_images, 
                        mmod_text_embeds, 
                        mmod_text_masks, 
                        image_token_type_idx):

            max_height= max([img.shape[1] for img in mmod_images])
            #maximum width of the images from the missing modality batches
            max_width= max([img.shape[2] for img in mmod_images])

            #num width patches
            num_width_patches=(max_width//self.hparams.config['patch_size'])
            #num height patches
            num_height_patches=(max_height//self.hparams.config['patch_size'])

            #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
            image_mmod_mask_len=num_height_patches*num_width_patches
            mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hparams.config['hidden_size']).to(self.config['device'])

            ####################################### creating the dummy tensort for the image masks ########################################
            #make zero tensors of (batch size, num_height_patches x num_width_patches)
            #include to0ken type embeddings for the image tokens
            image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
            mmod_text_embeds, mmod_image_embeds = (
                        mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                        mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                    )
 
            combined_img_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds],dim=1)
                    
            total_mmod_mask_fwd_pass_image=torch.cat([mmod_text_masks,image_mmod_mask_fwd_pass],dim=1)

            #forward pass for text and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_img_mmod_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
                    
            combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)

            mmod_image_embeds=combined_img_mmod_embeds[:,mmod_text_embeds.shape[1]:,:]

            return (mmod_image_embeds,combined_img_mmod_embeds)


    def handle_missing_text_modality(self,mmod_images,
                                mask_image,
                                mmod_text_embeds,
                                mmod_text_masks,image_token_type_idx):


            (mmod_image_embeds,
                        mmod_image_masks,
                        patch_index,
                        image_labels,
            ) = self.transformer.visual_embed(
                        mmod_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
            )

            #include to0ken type embeddings for the text and image tokens
            mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
            )

            combined_text_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds],dim=1)
            text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            total_mmod_mask_fwd_pass_text=torch.cat([text_mmod_mask_fwd_pass,mmod_image_masks],dim=1)

            #forward pass for image and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_text_mmod_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

            combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)
            mmod_image_embeds=combined_text_mmod_embeds[:,mmod_text_embeds.shape[1]:,:]

            return (mmod_image_embeds,combined_text_mmod_embeds)


    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag
        text_embeds = self.text_embeddings(text_ids)
        img = batch["image"]

        #no modality dropout .... no need to use modality dropout mask
            
        
        if((self.mod_dropout_flag==False) or (not any (mod_drop_flag)) ): #no modality dropout .... no need to use modality dropout mask
            
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
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )
            #combined embeddings
            combined_embeds=torch.cat([text_embeds,image_embeds],dim=1)
            
            ####################################################### FIRST FORWARD PASS SECTION #########################################################
            #first operation (forward pass) consists of attention between image and bottleneck tokens 
            #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
            
            fwd_pass_mask=torch.cat([text_masks,image_masks],dim=1)
            for i, blk in enumerate(self.transformer.blocks):
                combined_embeds, _attn = blk(combined_embeds, mask=fwd_pass_mask)
            
            combined_embeds = self.transformer.norm(combined_embeds) #layernorm of transformer on the image and bn embeddings
            text_feats, image_feats = (
                combined_embeds[:, 0:text_embeds.shape[1]],
                combined_embeds[:, text_embeds.shape[1]+image_embeds.shape[1]:],
            )
            first_token=combined_embeds[:,0,:] #directly using the first joint token for classification
            

        else: #modality dropout here 
            #two possible pathways in the forward pass 
            # (1) where modality dropout choice is image and complete modality is text: 
                # run the first forward pass with filtered image samples and then run the second forward pass with complete text samples
            # (2) where modality dropout choice is text and complete modality is image:
                # run the first forward pass with complete image samples and then run the second forward pass with filtered text samples
            
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped
            id_modality_dropped=index_modality_dropped.tolist()
            id_modality_complete=index_complete_modality.tolist()

            #filtered segment containing the missing modality samples 
            mmod_images=img[id_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[id_modality_dropped,:] 
            mmod_text_masks=text_masks[id_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[id_modality_complete,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[id_modality_complete,:]
            compl_text_masks=text_masks[id_modality_complete,:]


            if(len(id_modality_complete)==0):
                
                if(self.mod_choice=="image"):
                    
                    mmod_image_embeds,combined_img_mmod_embeds=self.handle_missing_image_modality(mmod_images, 
                                                mmod_text_embeds, 
                                                mmod_text_masks, 
                                                image_token_type_idx)
                    
                    text_feats=combined_img_mmod_embeds[:,:mmod_text_embeds.shape[1],:]
                    first_token=text_feats[:,0,:] #directly using the first joint token from text features for classification

                elif(self.mod_choice=="text"):

                    mmod_image_embeds,combined_text_mmod_embeds=self.handle_missing_text_modality(mmod_images,
                                                                    mask_image,
                                                                    mmod_text_embeds,
                                                                    mmod_text_masks,
                                                                    image_token_type_idx)

                    image_feats=combined_text_mmod_embeds[:,mmod_text_embeds.shape[1]:,:]
                    first_token=image_feats[:,0,:] #directly using the first joint token from image features for classification

            else:
                ####################################### ATTENTION BETWEEN (1) IMAGE (2) TEXT AND BOTTLENECK TOKENS FOR COMPLETE MODALITY SAMPLES ########################################
                #pass the completed images through visual embed encoder
                (
                    compl_image_embeds,
                    compl_image_masks,
                        patch_index,
                        image_labels,
                ) = self.transformer.visual_embed(
                        compl_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
                )

                #include token type embeddings for the text and image tokens
                compl_text_embeds, compl_image_embeds = (

                    compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                    compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
                )

                ## perform two stage attention operation between 
                # (1) image and bottleneck tokens (2) text and bottleneck tokens for complete modality samples
                combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds],dim=1)

                ####################################################### FIRST FORWARD PASS SECTION #########################################################
                #first operation (forward pass) consists of attention between image and bottleneck tokens 
                #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
                fwd_pass_mask_compl=torch.cat([compl_text_masks,compl_image_masks],dim=1)
                for i, blk in enumerate(self.transformer.blocks):
                    combined_embeds_compl, _attn = blk(combined_embeds_compl, mask=fwd_pass_mask_compl)
                
                combined_embeds_compl = self.transformer.norm(combined_embeds_compl) #layernorm of transformer on the image and bn embeddings

                compl_text_embeds, compl_image_embeds = ( 
                    combined_embeds_compl[:, 0:compl_text_embeds.shape[1]],
                    combined_embeds_compl[:, compl_text_embeds.shape[1]+compl_image_embeds.shape[1]:],
                )


                
                if(self.mod_choice=='image'): #modality dropout choice is image and complete modality is text
                    #print('Here')
                    ####################################### ATTENTION BETWEEN TEXT AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between text and bottleneck tokens for the missing modality samples
                    #mask for image tokens would be zeros and text would be normal masks and bottleneck tokens would be all ones
                    ####################################### creating the dummy tensort for the image tokens ########################################
                    #maximum height of the images from the missing modality batches
                    
                    mmod_image_embeds,combined_img_mmod_embeds=self.handle_missing_image_modality(mmod_images, 
                                                mmod_text_embeds, 
                                                mmod_text_masks, 
                                                image_token_type_idx)

                    #find minimum image sizes between the complete and missing modality samples
                    image_embeds_shape=min(mmod_image_embeds.shape[1],compl_image_embeds.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        compl_image_embeds=compl_image_embeds[:,:image_embeds_shape,:]

                    combined_img_mmod_embeds=torch.cat([combined_img_mmod_embeds[:,:mmod_text_embeds.shape[1],:],mmod_image_embeds],dim=1)
                    combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds],dim=1)

                    #print(combined_img_mmod_embeds.shape,combined_embeds_compl.shape)
                    #print(combined_embeds_compl.shape[1],combined_img_mmod_embeds.shape[1])
                    #aggregate with combined embeds empl for the complete modality samples in the specific indices
                    combined_embeds_repr=torch.zeros(text_embeds.shape[0],
                                    image_embeds_shape+text_embeds.shape[1],
                                    combined_embeds_compl.shape[2]).to(self.config['device'])

                    combined_embeds_repr[id_modality_complete,:]=combined_embeds_compl
                    combined_embeds_repr[id_modality_dropped,:]=combined_img_mmod_embeds
                    first_token=combined_embeds_repr[:,0,:]

                elif(self.mod_choice=='text'):
                    ####################################### ATTENTION BETWEEN IMAGE AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between image and bottleneck tokens for the missing modality samples
                    #mask for text tokens would be zeros and image would be normal masks and bottleneck tokens would be all ones
                    #pass the mmod images through visual embed encoder 
                    
                    mmod_image_embeds,combined_text_mmod_embeds=self.handle_missing_text_modality(mmod_images,
                                                                    mask_image,mmod_text_embeds,
                                                                    mmod_text_masks,
                                                                    image_token_type_idx)


                    #combined_embeds_repr=torch.zeros(combined_embeds_compl.shape[0],combined_embeds_compl.shape[1]+combined_text_mmod_embeds.shape[1],combined_embeds_compl.shape[2]).to(self.config['device'])
                    image_embeds_shape=min(mmod_image_embeds.shape[1],compl_image_embeds.shape[1])
                    
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        compl_image_embeds=compl_image_embeds[:,:image_embeds_shape,:]

                    combined_text_mmod_embeds=torch.cat([combined_text_mmod_embeds[:,:mmod_text_embeds.shape[1],:],mmod_image_embeds],dim=1)
                    combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds],dim=1)


                    combined_embeds_repr=torch.zeros(text_embeds.shape[0],
                                    image_embeds_shape+text_embeds.shape[1],
                                    combined_embeds_compl.shape[2]).to(self.config['device'])

                    #print(combined_embeds_repr.shape,combined_embeds_compl.shape,combined_text_mmod_embeds.shape)

                    combined_embeds_repr[id_modality_complete,:]=combined_embeds_compl
                    combined_embeds_repr[id_modality_dropped,:]=combined_text_mmod_embeds
                    first_token=combined_embeds_repr[:,0,:]
                    
            #(this has a shape (B",N,H) where B" is the number of complete modality samples and N is the number of bottleneck tokens and H is the hidden size)

        #pass it to classifier
        cls_logits=self.classifier(first_token)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits

####################################################### version of token averaging that takes into account no complete samples in batch and no missing modality samples in batch ########################################################
################################## updated implementation of ViLTTransformerBN_token_average that takes into account no complete samples in batch and no missing modality samples in batch ########################################

class ViLTTransformerBN_modality_token_average(pl.LightningModule): #uilize the token information from visual and text modalities instead of bottleneck tokens for final predictions
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

        #self.text_embeddings declared here
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #new addition 
        self.bottleneck_embeddings=nn.Embedding(config['num_bottleneck_tokens'],config['hidden_size']).to(config['device']) #not initiliazlied from original weights 
        
        #number of classes
        self.num_classes = config["num_classes"]
        self.patch_size = self.hparams.config["patch_size"]
        
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)

        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(hs, hs), #hs -> 2*hs
                nn.LayerNorm(hs), 
                nn.GELU(),
                nn.Linear(hs, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)


    def handle_missing_image_modality(self, mmod_images, 
                        mmod_text_embeds, 
                        mmod_text_masks, 
                        mmod_bottleneck_embeds, 
                        mmod_bottleneck_masks,
                        image_token_type_idx):

            ####################################### creating the dummy tensort for the image tokens ########################################
            #maximum height of the images from the missing modality batches
            max_height= max([img.shape[1] for img in mmod_images])
            #maximum width of the images from the missing modality batches
            max_width= max([img.shape[2] for img in mmod_images])

            #num width patches
            num_width_patches=(max_width//self.hparams.config['patch_size'])
            #num height patches
            num_height_patches=(max_height//self.hparams.config['patch_size'])

            #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
            image_mmod_mask_len=num_height_patches*num_width_patches
            mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hparams.config['hidden_size']).to(self.config['device'])

            ####################################### creating the dummy tensort for the image masks ########################################
            #make zero tensors of (batch size, num_height_patches x num_width_patches)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
                    
            #include to0ken type embeddings for the image tokens
            mmod_text_embeds, mmod_image_embeds = (
                        mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                        mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                    )

            combined_img_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len).to(self.config['device'])
            total_mmod_mask_fwd_pass_image=torch.cat([mmod_text_masks,image_mmod_mask_fwd_pass,mmod_bottleneck_masks],dim=1)

            #forward pass for text and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_img_mmod_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
                    
            combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)
            mmod_text_feats = combined_img_mmod_embeds[:, 0:mmod_text_embeds.shape[1]] #text features for the missing modality samples

            return mmod_text_feats, mmod_image_embeds, image_mmod_mask_len

    
    def handle_missing_text_modality(self,mmod_images,
                                mask_image,
                                mmod_text_embeds,
                                mmod_text_masks,image_token_type_idx,
                                mmod_bottleneck_embeds, 
                                mmod_bottleneck_masks
                                ):

            (mmod_image_embeds,mmod_image_masks,patch_index,image_labels) = self.transformer.visual_embed(
                        mmod_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
            )

            #include token type embeddings for the text and image tokens
            mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
                    )


            combined_text_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)
            text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            total_mmod_mask_fwd_pass_text=torch.cat([text_mmod_mask_fwd_pass,mmod_image_masks,mmod_bottleneck_masks],dim=1)

            #forward pass for image and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_text_mmod_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

            combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)
                    
            #extract the image embeddings from co_text_mmod_bn_embeds
            mmod_image_feats = combined_text_mmod_embeds[:, mmod_text_embeds.shape[1]:mmod_text_embeds.shape[1]+mmod_image_embeds.shape[1]]

            return mmod_text_embeds, mmod_image_feats

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag

        text_embeds = self.text_embeddings(text_ids)

        #if image_embeds is None and image_masks is None:
        img = batch["image"]
        #print(img)
        
        #bottleneck embeds 
        bottleneck_masks=torch.ones(text_embeds.shape[0],self.hparams.config['num_bottleneck_tokens']).to(self.config['device']) #always ones
        bottleneck_embeds=self.bottleneck_embeddings(torch.arange(self.hparams.config['num_bottleneck_tokens']).to(self.config['device']).unsqueeze(0).repeat(text_embeds.shape[0],1)) #expand to (batch_size,num_bottleneck_tokens,hidden_size)
        if(self.mod_dropout_flag==False or (not any (mod_drop_flag)) ): #no modality dropout .... no need to use modality dropout mask
            
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
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )
            #combined embeddings
            combined_embeds=torch.cat([text_embeds,image_embeds,bottleneck_embeds],dim=1)
            combined_embeds_pass_2=combined_embeds.clone()

            ####################################################### FIRST FORWARD PASS SECTION #########################################################
            #first operation (forward pass) consists of attention between image and bottleneck tokens 
            #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
            text_mask_fwd_pass_1=torch.zeros(image_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            fwd_pass_1_mask=torch.cat([text_mask_fwd_pass_1,image_masks,bottleneck_masks],dim=1)
            for i, blk in enumerate(self.transformer.blocks):
                combined_embeds, _attn = blk(combined_embeds, mask=fwd_pass_1_mask)  #error here need to fix .. same embedding space not used at all
            
            combined_embeds = self.transformer.norm(combined_embeds) #layernorm of transformer on the image and bn embeddings
            image_feats, bn_image_feats = (
                combined_embeds[:, text_embeds.shape[1]:text_embeds.shape[1]+image_embeds.shape[1]],
                combined_embeds[:, text_embeds.shape[1]+image_embeds.shape[1]:],
            )
            #print(image_feats.shape,bn_image_feats.shape)


            ####################################################### SECOND FORWARD PASS SECTION #########################################################
            #second operation (forward pass) consists of attention between text and bottleneck tokens 
            #so text masks would be mixture of 0s and 1s, bottleneck masks would be all ones and image masks would be all zeros 
            image_mask_max_len=image_masks.shape[1]
            image_mask_fwd_pass_2=torch.zeros(image_embeds.shape[0],image_mask_max_len).to(self.config['device'])
            fwd_pass_2_mask=torch.cat([text_masks,image_mask_fwd_pass_2,bottleneck_masks],dim=1)

            for i, blk in enumerate(self.transformer.blocks):
                combined_embeds_pass_2, _attn = blk(combined_embeds_pass_2, mask=fwd_pass_2_mask)
            
            combined_embeds_pass_2 = self.transformer.norm(combined_embeds_pass_2) #layernorm of transformer on the image and bn embeddings
            text_feats, bn_text_feats = (
                combined_embeds_pass_2[:, :text_embeds.shape[1]],
                combined_embeds_pass_2[:, text_embeds.shape[1]+image_embeds.shape[1]:],
            )
            

            image_feats_avg=torch.mean(image_feats,dim=1) #average pooling of the image features
            text_feats_avg=torch.mean(text_feats,dim=1) #average pooling of the text features
            avg_token_feats=(image_feats_avg+text_feats_avg)/2 #average of the bn feats from both the forward passes

        else: #modality dropout here 

            #two possible pathways in the forward pass 
            # (1) where modality dropout choice is image and complete modality is text: 
                # run the first forward pass with filtered image samples and then run the second forward pass with complete text samples
            # (2) where modality dropout choice is text and complete modality is image:
                # run the first forward pass with complete image samples and then run the second forward pass with filtered text samples
            
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped
            id_modality_dropped=index_modality_dropped.tolist()
            id_modality_complete=index_complete_modality.tolist()

            #filtered segment containing the missing modality samples 
            mmod_images=img[id_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[id_modality_dropped,:] 
            mmod_bottleneck_embeds=bottleneck_embeds[id_modality_dropped,:]
            mmod_text_masks=text_masks[id_modality_dropped,:]
            mmod_bottleneck_masks=bottleneck_masks[id_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[id_modality_complete,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[id_modality_complete,:]
            compl_bottleneck_embeds=bottleneck_embeds[id_modality_complete,:]
            compl_text_masks=text_masks[id_modality_complete,:]
            compl_bottleneck_masks=bottleneck_masks[id_modality_complete,:]



            if(len(id_modality_complete)==0): #there are no samples with complete modality in the current batch

                #use the missing modality samples to run the first forward pass
                if(self.mod_choice=="image"):
                    
                    mmod_text_feats,mmod_image_embeds,image_mmod_mask_len = self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       mmod_bottleneck_embeds,
                                                       mmod_bottleneck_masks,
                                                       image_token_type_idx
                                                       )      
                    avg_token_feats=torch.mean(mmod_text_feats,dim=1) #average pooling of the text features

                elif(self.mod_choice=="text"):

                    mmod_text_embeds, mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                                mask_image,
                                                                mmod_text_embeds,
                                                                mmod_text_masks,
                                                                image_token_type_idx,
                                                                mmod_bottleneck_embeds, 
                                                                mmod_bottleneck_masks)
                    avg_token_feats=torch.mean(mmod_image_feats,dim=1) 

            else:
                ####################################### ATTENTION BETWEEN (1) IMAGE (2) TEXT AND BOTTLENECK TOKENS FOR COMPLETE MODALITY SAMPLES ########################################
                #pass the completed images through visual embed encoder
                (
                    compl_image_embeds,
                    compl_image_masks,
                        patch_index,
                        image_labels,
                ) = self.transformer.visual_embed(
                        compl_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
                )

                #include token type embeddings for the text and image tokens
                compl_text_embeds, compl_image_embeds = (

                    compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                    compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
                )

                ## perform two stage attention operation between 
                # (1) image and bottleneck tokens (2) text and bottleneck tokens for complete modality samples
                combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds,compl_bottleneck_embeds],dim=1)
                combined_embeds_compl_pass_2=torch.cat([compl_text_embeds,compl_image_embeds,compl_bottleneck_embeds],dim=1)
                
                ####################################################### FIRST FORWARD PASS SECTION #########################################################
                #first operation (forward pass) consists of attention between image and bottleneck tokens 
                #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
                text_mask_fwd_pass_1_compl=torch.zeros(compl_image_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
                fwd_pass_1_mask_compl=torch.cat([text_mask_fwd_pass_1_compl,compl_image_masks,compl_bottleneck_masks],dim=1)
                for i, blk in enumerate(self.transformer.blocks):
                    combined_embeds_compl, _attn = blk(combined_embeds_compl, mask=fwd_pass_1_mask_compl)
                
                combined_embeds_compl = self.transformer.norm(combined_embeds_compl) #layernorm of transformer on the image and bn embeddings
                bn_image_feats_compl=combined_embeds_compl[:, compl_text_embeds.shape[1]:compl_text_embeds.shape[1]+compl_image_embeds.shape[1]] #first part of the bottleneck tokens 


                ####################################################### SECOND FORWARD PASS SECTION #########################################################
                #second operation (forward pass) consists of attention between text and bottleneck tokens 
                #so text masks would be mixture of 0s and 1s, bottleneck masks would be all ones and image masks would be all zeros 
                image_mask_max_len=compl_image_masks.shape[1]
                image_mask_fwd_pass_2=torch.zeros(compl_image_embeds.shape[0],image_mask_max_len).to(self.config['device'])
                fwd_pass_2_mask_compl=torch.cat([compl_text_masks,image_mask_fwd_pass_2,compl_bottleneck_masks],dim=1)
                for i, blk in enumerate(self.transformer.blocks):
                    combined_embeds_compl_pass_2, _attn = blk(combined_embeds_compl_pass_2, mask=fwd_pass_2_mask_compl)
                
                combined_embeds_compl_pass_2 = self.transformer.norm(combined_embeds_compl_pass_2) #layernorm of transformer on the image and bn embeddings
                bn_text_feats_compl = combined_embeds_compl_pass_2[:, 0:compl_text_embeds.shape[1]]

                #idea is to inject the mmod_bottleneck_embeds (passed through vilt) into the indices i.e. index_modality_dropped  
                #compl_bottleneck_embeds (passed through vilt) into the indices i.e. index_complete_modality
                #print("bn feats before:",bn_feats)

                if(self.mod_choice=='image'): #modality dropout choice is image and complete modality is text
                    #print('Here')
                    ####################################### ATTENTION BETWEEN TEXT AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between text and bottleneck tokens for the missing modality samples
                    #mask for image tokens would be zeros and text would be normal masks and bottleneck tokens would be all ones
                    mmod_text_feats,mmod_image_embeds, image_mmod_mask_len= self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       mmod_bottleneck_embeds,
                                                       mmod_bottleneck_masks,
                                                       image_token_type_idx
                                                       )           

                    #dummy tensor where all the text features are stored
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hparams.config['hidden_size']).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_feats

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(image_mmod_mask_len,bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]
                    
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hparams.config['hidden_size']).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    avg_token_feats=torch.zeros(text_feats.shape[0],self.hparams.config['hidden_size']).to(self.config['device'])
                    avg_token_feats[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    avg_token_feats[id_modality_dropped,:]=text_feats_avg[id_modality_dropped,:]
                    #print(avg_token_feats)
                    
                elif(self.mod_choice=='text'):
                    ####################################### ATTENTION BETWEEN IMAGE AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between image and bottleneck tokens for the missing modality samples
                    #mask for text tokens would be zeros and image would be normal masks and bottleneck tokens would be all ones

                    
                    mmod_text_embeds, mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                                mask_image,mmod_text_embeds,
                                                                mmod_text_masks,image_token_type_idx,
                                                                mmod_bottleneck_embeds, mmod_bottleneck_masks)
                    
                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(mmod_image_feats.shape[1],bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_feats=mmod_image_feats[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]
                    
                    #image features 
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hparams.config['hidden_size']).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_feats

                    #create tensor for text features for both missing and complete modality samples
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hparams.config['hidden_size']).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    avg_token_feats=torch.zeros(text_feats.shape[0],self.hparams.config['hidden_size']).to(self.config['device'])
                    avg_token_feats[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    avg_token_feats[id_modality_dropped,:]=image_feats_avg[id_modality_dropped,:]
                    
                #(this has a shape (B",N,H) where B" is the number of complete modality samples and N is the number of bottleneck tokens and H is the hidden size)


        #pass it to classifier
        cls_logits=self.classifier(avg_token_feats)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits



#################################### this version is different from previous version since for complete samples it allows attention between all tokens ########################################                          
class ViLTTransformerBN_free_attention_modality_token_average(pl.LightningModule): #uilize the token information from visual and text modalities instead of bottleneck tokens for final predictions
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

        #self.text_embeddings declared here
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        #new addition 
        self.bottleneck_embeddings=nn.Embedding(config['num_bottleneck_tokens'],config['hidden_size']).to(config['device']) #not initiliazlied from original weights 
        
        #number of classes
        self.num_classes = config["num_classes"]
        self.patch_size = self.hparams.config["patch_size"]
        
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)

        hs=self.hparams.config["hidden_size"]
        self.classifier = nn.Sequential(
                nn.Linear(hs, hs), #hs -> 2*hs
                nn.LayerNorm(hs), 
                nn.GELU(),
                nn.Linear(hs, self.num_classes)
            )
        self.classifier.apply(objectives.init_weights)


    def handle_missing_image_modality(self, mmod_images, 
                        mmod_text_embeds, 
                        mmod_text_masks, 
                        mmod_bottleneck_embeds, 
                        mmod_bottleneck_masks,
                        image_token_type_idx):

            ####################################### creating the dummy tensort for the image tokens ########################################
            #maximum height of the images from the missing modality batches
            max_height= max([img.shape[1] for img in mmod_images])
            #maximum width of the images from the missing modality batches
            max_width= max([img.shape[2] for img in mmod_images])

            #num width patches
            num_width_patches=(max_width//self.hparams.config['patch_size'])
            #num height patches
            num_height_patches=(max_height//self.hparams.config['patch_size'])

            #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
            image_mmod_mask_len=num_height_patches*num_width_patches
            mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hparams.config['hidden_size']).to(self.config['device'])

            ####################################### creating the dummy tensort for the image masks ########################################
            #make zero tensors of (batch size, num_height_patches x num_width_patches)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
                    
            #include to0ken type embeddings for the image tokens
            mmod_text_embeds, mmod_image_embeds = (
                        mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                        mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                    )

            combined_img_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len).to(self.config['device'])
            total_mmod_mask_fwd_pass_image=torch.cat([mmod_text_masks,image_mmod_mask_fwd_pass,mmod_bottleneck_masks],dim=1)

            #forward pass for text and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_img_mmod_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
                    
            combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)
            mmod_text_feats = combined_img_mmod_embeds[:, 0:mmod_text_embeds.shape[1]] #text features for the missing modality samples

            return mmod_text_feats, mmod_image_embeds, image_mmod_mask_len

    
    def handle_missing_text_modality(self,mmod_images,
                                mask_image,
                                mmod_text_embeds,
                                mmod_text_masks,image_token_type_idx,
                                mmod_bottleneck_embeds, 
                                mmod_bottleneck_masks
                                ):

            (mmod_image_embeds,mmod_image_masks,patch_index,image_labels) = self.transformer.visual_embed(
                        mmod_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
            )

            #include token type embeddings for the text and image tokens
            mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
                    )


            combined_text_mmod_embeds=torch.cat([mmod_text_embeds,mmod_image_embeds,mmod_bottleneck_embeds],dim=1)
            text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            total_mmod_mask_fwd_pass_text=torch.cat([text_mmod_mask_fwd_pass,mmod_image_masks,mmod_bottleneck_masks],dim=1)

            #forward pass for image and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_text_mmod_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

            combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)
                    
            #extract the image embeddings from co_text_mmod_bn_embeds
            mmod_image_feats = combined_text_mmod_embeds[:, mmod_text_embeds.shape[1]:mmod_text_embeds.shape[1]+mmod_image_embeds.shape[1]]

            return mmod_text_embeds, mmod_image_feats

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #text embedding portion 
        text_ids = batch["text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag

        text_embeds = self.text_embeddings(text_ids)

        #if image_embeds is None and image_masks is None:
        img = batch["image"]
        #print(img)
        
        #bottleneck embeds 
        bottleneck_masks=torch.ones(text_embeds.shape[0],self.hparams.config['num_bottleneck_tokens']).to(self.config['device']) #always ones
        bottleneck_embeds=self.bottleneck_embeddings(torch.arange(self.hparams.config['num_bottleneck_tokens']).to(self.config['device']).unsqueeze(0).repeat(text_embeds.shape[0],1)) #expand to (batch_size,num_bottleneck_tokens,hidden_size)
        if(self.mod_dropout_flag==False or (not any (mod_drop_flag)) ): #no modality dropout .... no need to use modality dropout mask
            
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
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )
            #combined embeddings
            combined_embeds=torch.cat([text_embeds,image_embeds,bottleneck_embeds],dim=1)
            
            ####################################################### SINGLE FORWARD PASS SECTION #########################################################
            #first operation (forward pass) consists of attention between image and bottleneck tokens 
            #so text masks would be completely zeros, bottleneck masks would be all ones and image masks would be combination of 1s and zeros 
            fwd_pass_mask=torch.cat([text_masks,image_masks,bottleneck_masks],dim=1)
            for i, blk in enumerate(self.transformer.blocks):
                combined_embeds, _attn = blk(combined_embeds, mask=fwd_pass_mask)  #error here need to fix .. same embedding space not used at all
            
            combined_embeds = self.transformer.norm(combined_embeds) #layernorm of transformer on the image and bn embeddings
            text_feats, image_feats = (
                combined_embeds[:, 0:text_embeds.shape[1]],
                combined_embeds[:, text_embeds.shape[1]:text_embeds.shape[1]+image_embeds.shape[1]],
            )

            image_feats_avg=torch.mean(image_feats,dim=1) #average pooling of the image features
            text_feats_avg=torch.mean(text_feats,dim=1) #average pooling of the text features
            avg_token_feats=(image_feats_avg+text_feats_avg)/2 #average of the bn feats from both the forward passes

        else: #modality dropout here 

            #two possible pathways in the forward pass 
            # (1) where modality dropout choice is image and complete modality is text: 
                # run the first forward pass with filtered image samples and then run the second forward pass with complete text samples
            # (2) where modality dropout choice is text and complete modality is image:
                # run the first forward pass with complete image samples and then run the second forward pass with filtered text samples
            
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped
            id_modality_dropped=index_modality_dropped.tolist()
            id_modality_complete=index_complete_modality.tolist()

            #filtered segment containing the missing modality samples 
            mmod_images=img[id_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[id_modality_dropped,:] 
            mmod_bottleneck_embeds=bottleneck_embeds[id_modality_dropped,:]
            mmod_text_masks=text_masks[id_modality_dropped,:]
            mmod_bottleneck_masks=bottleneck_masks[id_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[id_modality_complete,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[id_modality_complete,:]
            compl_bottleneck_embeds=bottleneck_embeds[id_modality_complete,:]
            compl_text_masks=text_masks[id_modality_complete,:]
            compl_bottleneck_masks=bottleneck_masks[id_modality_complete,:]



            if(len(id_modality_complete)==0): #there are no samples with complete modality in the current batch

                #use the missing modality samples to run the first forward pass
                if(self.mod_choice=="image"):
                    
                    mmod_text_feats,mmod_image_embeds,image_mmod_mask_len = self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       mmod_bottleneck_embeds,
                                                       mmod_bottleneck_masks,
                                                       image_token_type_idx
                                                       )      
                    avg_token_feats=torch.mean(mmod_text_feats,dim=1) #average pooling of the text features

                elif(self.mod_choice=="text"):

                    mmod_text_embeds, mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                                mask_image,
                                                                mmod_text_embeds,
                                                                mmod_text_masks,
                                                                image_token_type_idx,
                                                                mmod_bottleneck_embeds, 
                                                                mmod_bottleneck_masks)
                    avg_token_feats=torch.mean(mmod_image_feats,dim=1) 

            else:
                ####################################### ATTENTION BETWEEN (1) IMAGE TEXT AND BOTTLENECK TOKENS FOR COMPLETE MODALITY SAMPLES WITHOUT RESTRICTIONS ########################################
                #pass the completed images through visual embed encoder
                (
                    compl_image_embeds,
                    compl_image_masks,
                        patch_index,
                        image_labels,
                ) = self.transformer.visual_embed(
                        compl_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
                )

                #include token type embeddings for the text and image tokens
                compl_text_embeds, compl_image_embeds = (

                    compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                    compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
                )

                ## perform two stage attention operation between 
                # (1) image and bottleneck tokens (2) text and bottleneck tokens for complete modality samples
                combined_embeds_compl=torch.cat([compl_text_embeds,compl_image_embeds,compl_bottleneck_embeds],dim=1)
                
                
                ####################################################### FIRST FORWARD PASS SECTION #########################################################
                #forward pass consists of attention between image, text and bottleneck tokens 
                fwd_pass_mask_compl=torch.cat([compl_text_masks,compl_image_masks,compl_bottleneck_masks],dim=1)
                for i, blk in enumerate(self.transformer.blocks):
                    combined_embeds_compl, _attn = blk(combined_embeds_compl, mask=fwd_pass_mask_compl)
                
                combined_embeds_compl = self.transformer.norm(combined_embeds_compl) #layernorm of transformer on the image and bn embeddings
                bn_image_feats_compl=combined_embeds_compl[:, compl_text_embeds.shape[1]:compl_text_embeds.shape[1]+compl_image_embeds.shape[1]] #first part of the bottleneck tokens 
                bn_text_feats_compl = combined_embeds_compl[:, 0:compl_text_embeds.shape[1]]

                #idea is to inject the mmod_bottleneck_embeds (passed through vilt) into the indices i.e. index_modality_dropped  
                #compl_bottleneck_embeds (passed through vilt) into the indices i.e. index_complete_modality
                #print("bn feats before:",bn_feats)

                if(self.mod_choice=='image'): #modality dropout choice is image and complete modality is text
                    #print('Here')
                    ####################################### ATTENTION BETWEEN TEXT AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between text and bottleneck tokens for the missing modality samples
                    #mask for image tokens would be zeros and text would be normal masks and bottleneck tokens would be all ones
                    mmod_text_feats,mmod_image_embeds, image_mmod_mask_len= self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       mmod_bottleneck_embeds,
                                                       mmod_bottleneck_masks,
                                                       image_token_type_idx
                                                       )           

                    #dummy tensor where all the text features are stored
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hparams.config['hidden_size']).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_feats

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(image_mmod_mask_len,bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]
                    
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hparams.config['hidden_size']).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    avg_token_feats=torch.zeros(text_feats.shape[0],self.hparams.config['hidden_size']).to(self.config['device'])
                    avg_token_feats[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    avg_token_feats[id_modality_dropped,:]=text_feats_avg[id_modality_dropped,:]
                    #print(avg_token_feats)
                    
                elif(self.mod_choice=='text'):
                    ####################################### ATTENTION BETWEEN IMAGE AND BOTTLENECK TOKENS FOR MISSING MODALITY SAMPLES ########################################
                    ## perform attention between image and bottleneck tokens for the missing modality samples
                    #mask for text tokens would be zeros and image would be normal masks and bottleneck tokens would be all ones

                    
                    mmod_text_embeds, mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                                mask_image,mmod_text_embeds,
                                                                mmod_text_masks,image_token_type_idx,
                                                                mmod_bottleneck_embeds, mmod_bottleneck_masks)
                    
                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(mmod_image_feats.shape[1],bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_feats=mmod_image_feats[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]
                    
                    #image features 
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hparams.config['hidden_size']).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_feats

                    #create tensor for text features for both missing and complete modality samples
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hparams.config['hidden_size']).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    avg_token_feats=torch.zeros(text_feats.shape[0],self.hparams.config['hidden_size']).to(self.config['device'])
                    avg_token_feats[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    avg_token_feats[id_modality_dropped,:]=image_feats_avg[id_modality_dropped,:]
                    
                #(this has a shape (B",N,H) where B" is the number of complete modality samples and N is the number of bottleneck tokens and H is the hidden size)


        #pass it to classifier
        cls_logits=self.classifier(avg_token_feats)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits

######## prompts based design of the model ########
### design based on the codebase here : https://github.com/KMnP/vpt/blob/main/src/models/vit_prompt/vit.py ###
#### complete_prompts: set of nn.Parameter values for complete samples####
#### dropped_prompts: set of nn.Parameter values for dropped samples####
#### 1. text_prompt: text prompt for the text modality (image missing) ####
#### 2. image_prompt: image prompt for the image modality (text missing) ####

class ViLT_multi_prompt_missing_mod_token_average(pl.LightningModule):

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
        
        #text_embeddings declared here
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']
        self.initialization=config['initialization']
        self.hs=self.hparams.config["hidden_size"]
        self.num_prompts=self.hparams.config["num_prompts"]
        self.prompt_concat_option=self.hparams.config["prompt_concat_option"]
        self.patch_size=self.hparams.config["patch_size"]

        #token type embeddings for the text part
        self.token_type_embeddings = nn.Embedding(2, self.hs)
        self.token_type_embeddings.apply(objectives.init_weights)

        ########################## keeping number of prompts same for now ############################  
        #prompt embeddings for the text part (with image missing)
        self.text_prompt_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts, self.hs),requires_grad=True)
        self.text_prompt_dropout=nn.Dropout(config['prompt_drop_rate'])

        #prompt embeddings for the image part (with text missing)
        self.image_prompt_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts, self.hs),requires_grad=True)
        self.image_prompt_dropout=nn.Dropout(config['prompt_drop_rate'])

        #prompt embeddings for complete samples
        self.complete_prompt_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts, self.hs),requires_grad=True)
        self.complete_prompt_dropout=nn.Dropout(config['prompt_drop_rate'])

        ### initialize the weights using xavier initialization ###
        if(self.initialization=='xavier'):
            self.initialize_weights(option=self.initialization, 
                            patch_size=self.patch_size, 
                            prompt_dim=self.hs)

        elif(self.initialization=='kaiming'):
            self.initialize_weights(option=self.initialization, 
                        patch_size=self.patch_size, 
                        prompt_dim=self.hs)
        # #initialize the text_prompt_embeddings

        #number of classes
        self.num_classes = config["num_classes"]

        #patch size
        self.patch_size = self.hparams.config["patch_size"]
        
        # transformer 
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)


        #classifier declaration
        self.classifier = nn.Sequential(
                nn.Linear(self.hs, self.hs), #hs -> 2*hs
                nn.LayerNorm(self.hs), 
                nn.GELU(),
                nn.Linear(self.hs, self.num_classes)
            )

        #classifier initialization
        self.classifier.apply(objectives.init_weights)


    def initialize_weights(self,option='xavier',patch_size=32,prompt_dim=768):
        if(option=='xavier'):
            
            #initialize the text_prompt_embeddings
            nn.init.xavier_uniform_(self.text_prompt_embeddings.data)
            #initialize the image_prompt_embeddings
            nn.init.xavier_uniform_(self.image_prompt_embeddings.data)
            #initialize the complete_prompt_embeddings
            nn.init.xavier_uniform_(self.complete_prompt_embeddings.data)

        elif(option=='kaiming'):
            
            #initialize the text_prompt_embeddings
            nn.init.kaiming_uniform_(self.text_prompt_embeddings.data)
            #initialize the image_prompt_embeddings
            nn.init.kaiming_uniform_(self.image_prompt_embeddings.data)
            #initialize the complete_prompt_embeddings
            nn.init.kaiming_uniform_(self.complete_prompt_embeddings.data)

        elif (option=='normal'):
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
            #initialize the text_prompt_embeddings
            nn.init.normal_(self.text_prompt_embeddings.data,(-val,val))
            #initialize the image_prompt_embeddings
            nn.init.normal_(self.image_prompt_embeddings.data,(-val,val))
            #initialize the complete_prompt_embeddings
            nn.init.normal_(self.complete_prompt_embeddings.data,(-val,val))

    def handle_missing_image_modality(self,mmod_images,
                                        mmod_text_embeds,
                                        mmod_text_masks,
                                        text_prompt_embeddings,
                                        image_token_type_idx
                                        ):
            #handle missing image modalities using the text prompt embeddings
            ####################################### creating the dummy tensort for the image tokens ########################################
            #maximum height of the images from the missing modality batches
            max_height= max([img.shape[1] for img in mmod_images])
            #maximum width of the images from the missing modality batches
            max_width= max([img.shape[2] for img in mmod_images])

            #num width patches
            num_width_patches=(max_width//self.hparams.config['patch_size'])
            #num height patches
            num_height_patches=(max_height//self.hparams.config['patch_size'])

            #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
            image_mmod_mask_len=num_height_patches*num_width_patches
            mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hs).to(self.config['device'])

            ####################################### creating the dummy tensort for the image masks ########################################
            #make zero tensors of (batch size, num_height_patches x num_width_patches)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
                    
            #include token type embeddings for the image tokens
            mmod_text_embeds, mmod_image_embeds = (
                        mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                        mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                    )

            #concatenate the text and image embeddings with prompt embeddings for the text (used with image modality missing)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len).to(self.config['device'])
            text_prompt_mask_fwd_pass=torch.ones(mmod_text_embeds.shape[0],self.num_prompts).to(self.config['device'])

            if(self.prompt_concat_option=="middle"):
                combined_img_mmod_embeds=torch.cat((mmod_text_embeds,
                                self.text_prompt_dropout(text_prompt_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_image_embeds
                                ),dim=1)
                total_mmod_mask_fwd_pass_image=torch.cat([mmod_text_masks,
                                                text_prompt_mask_fwd_pass,
                                                image_mmod_mask_fwd_pass],dim=1)

            elif(self.prompt_concat_option=="prepend"):

                combined_img_mmod_embeds=torch.cat((self.text_prompt_dropout(text_prompt_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_text_embeds,
                                mmod_image_embeds
                                ),dim=1)
                total_mmod_mask_fwd_pass_image=torch.cat([text_prompt_mask_fwd_pass,
                                            mmod_text_masks,
                                            image_mmod_mask_fwd_pass],dim=1)

            #forward pass for text and bottleneck for the missing modality samples
            for i, blk in enumerate(self.transformer.blocks):
                combined_img_mmod_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
            
            combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)

            if(self.prompt_concat_option=="middle"):
                mmod_text_feats = combined_img_mmod_embeds[:, 0:mmod_text_embeds.shape[1]]
            elif(self.prompt_concat_option=="prepend"):
                mmod_text_feats = combined_img_mmod_embeds[:, self.num_prompts:self.num_prompts+mmod_text_embeds.shape[1]]

            return mmod_text_feats, mmod_image_embeds, image_mmod_mask_len

    def handle_missing_text_modality(self,mmod_images,
                                mmod_text_embeds,
                                mmod_text_masks,
                                image_prompt_embeddings,
                                image_token_type_idx,
                                mask_image
                                ):

            (mmod_image_embeds, mmod_image_masks,
            patch_index,image_labels) = self.transformer.visual_embed(
                    mmod_images,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image
            )

            #include token type embeddings for the text and image tokens
            mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
            )

            text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            image_prompt_mask_fwd_pass=torch.ones(mmod_image_embeds.shape[0],self.num_prompts).to(self.config['device'])

            if(self.prompt_concat_option=="middle"):
                #conatenate the text and image embeddings with prompt embeddings for the image (used with text modality missing)
                combined_text_mmod_embeds=torch.cat((mmod_text_embeds,
                                self.image_prompt_dropout(image_prompt_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_image_embeds
                                ),dim=1)
            
                total_mmod_mask_fwd_pass_text=torch.cat([text_mmod_mask_fwd_pass,image_prompt_mask_fwd_pass,mmod_image_masks],dim=1)

            elif(self.prompt_concat_option=="prepend"):

                combined_text_mmod_embeds=torch.cat((self.image_prompt_dropout(image_prompt_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_text_embeds,
                                mmod_image_embeds
                                ),dim=1)
                total_mmod_mask_fwd_pass_text=torch.cat([image_prompt_mask_fwd_pass,text_mmod_mask_fwd_pass,mmod_image_masks],dim=1)


            for i, blk in enumerate(self.transformer.blocks):
                combined_text_mmod_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

            combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)

            if(self.prompt_concat_option=="middle"):
                mmod_image_feats = combined_text_mmod_embeds[:, mmod_text_embeds.shape[1]+self.num_prompts:]
            elif(self.prompt_concat_option=="prepend"):
                mmod_image_feats = combined_text_mmod_embeds[:, self.num_prompts+mmod_text_embeds.shape[1]:]

            return mmod_text_embeds, mmod_image_feats
    
    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #print(self.complete_prompt_embeddings )
        #text embedding portion 
        text_ids = batch["text_ids"]

        #text masks and modality dropout flag
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag

        #read the text embeddings
        text_embeds = self.text_embeddings(text_ids)

        #read the images 
        img = batch["image"]

        ## masks for complete prompt embeddings ##
        complete_prompt_masks=torch.ones(text_embeds.shape[0],
                        self.num_prompts).to(self.config['device']) #always ones

        if(self.mod_dropout_flag==False or (not any (mod_drop_flag)) ): #no modality dropout .... no need to use modality dropout mask
            
            #image embeddings, image masks
            image_embeds,image_masks,patch_index,image_labels = self.transformer.visual_embed(img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                )

            #text embeddings + token type embeddings, image embeddings + token type embeddings
            text_embeds, image_embeds = (
                    text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )


            if(self.prompt_concat_option=="middle"):
                #concatenate the text and image embeddings with the complete prompt embeddings
                complete_embeds=torch.cat((text_embeds,
                                self.complete_prompt_dropout(self.complete_prompt_embeddings.expand(text_embeds.shape[0],-1,-1)),
                                image_embeds
                                ),dim=1)
                ### concatenate the masks #####
                complete_masks=torch.cat((text_masks,complete_prompt_masks,image_masks),dim=1)
            
            elif(self.prompt_concat_option=="prepend"):

                complete_embeds=torch.cat((self.complete_prompt_dropout(self.complete_prompt_embeddings.expand(text_embeds.shape[0],-1,-1)),
                                text_embeds,
                                image_embeds
                                ),dim=1)
                complete_masks=torch.cat((complete_prompt_masks,text_masks,image_masks),dim=1)

            for i, blk in enumerate(self.transformer.blocks):
                complete_embeds, _attn = blk(complete_embeds, mask=complete_masks)

            #take the first token here ###
            pool_token=complete_embeds[:,0,:] #taking CLS of the combined sequence 
            #print(pool_token.size())

        else:

            #missing modality scenario ####
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped
            id_modality_dropped=index_modality_dropped.tolist()
            id_modality_complete=index_complete_modality.tolist()

            #filtered segment containing the missing modality samples 
            mmod_images=img[id_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[id_modality_dropped,:] 
            mmod_text_masks=text_masks[id_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[id_modality_complete,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[id_modality_complete,:]
            compl_text_masks=text_masks[id_modality_complete,:]
            
            if(len(id_modality_complete)==0):

                    if(self.mod_choice=="image"):

                        # text_prompt_embeddings=self.text_prompt_dropout(self.text_prompt_embeddings.expand(mmod_images.shape[0],-1,-1))

                        mmod_text_feats,mmod_image_embeds,image_mmod_mask_len = self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       self.text_prompt_embeddings,
                                                       image_token_type_idx
                                                       )      
                        pool_token=torch.mean(mmod_text_feats,dim=1) #average pooling of the text features
                        #we can take the first token here as well

                    elif(self.mod_choice=="text"):

                        # image_prompt_embeddings=self.image_prompt_dropout(self.image_prompt_embeddings.expand(mmod_images.shape[0],-1,-1))
                        mmod_text_embeds,mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       self.image_prompt_embeddings,
                                                       image_token_type_idx,
                                                       mask_image
                                                       )      
                        pool_token=torch.mean(mmod_image_feats,dim=1)

            else:

                #attention between image, text and complete prompt embeddings
                (
                    compl_image_embeds,
                    compl_image_masks,
                        patch_index,
                        image_labels,
                ) = self.transformer.visual_embed(
                        compl_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
                )

                #include token type embeddings for the text and image tokens
                compl_text_embeds, compl_image_embeds = (

                    compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                    compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
                )
                
                #complete masks for the prompts###
                compl_prompt_masks=torch.ones(compl_text_embeds.shape[0],self.num_prompts).to(self.config['device']) #always ones
                #concatenate the text and image embeddings with the complete prompt embeddings

                if(self.prompt_concat_option=="middle"):
                    combined_embeds_compl=torch.cat((compl_text_embeds,
                                    self.complete_prompt_dropout(self.complete_prompt_embeddings.expand(compl_text_embeds.shape[0],-1,-1)),
                                    compl_image_embeds
                                    ),dim=1)

                    # complete embeddings + complete masks #
                    combined_embeds_compl_mask=torch.cat((compl_text_masks,compl_prompt_masks,compl_image_masks),dim=1)

                    #### first forward pass for the complete modality samples ####
                    for i, blk in enumerate(self.transformer.blocks):
                        combined_embeds_compl, _attn = blk(combined_embeds_compl, mask=combined_embeds_compl_mask)

                    combined_embeds_compl = self.transformer.norm(combined_embeds_compl)
                    bn_image_feats_compl=combined_embeds_compl[:, compl_text_embeds.shape[1]+self.num_prompts:] #first part of the bottleneck tokens 
                    bn_text_feats_compl = combined_embeds_compl[:, 0:compl_text_embeds.shape[1]]

                elif(self.prompt_concat_option=="prepend"):
                    
                    combined_embeds_compl=torch.cat((self.complete_prompt_dropout(self.complete_prompt_embeddings.expand(compl_text_embeds.shape[0],-1,-1)),
                                    compl_text_embeds,
                                    compl_image_embeds
                                    ),dim=1)
                    
                    # complete embeddings + complete masks #
                    combined_embeds_compl_mask=torch.cat((compl_prompt_masks,compl_text_masks,compl_image_masks),dim=1)

                    #### first forward pass for the complete modality samples ####
                    #### first forward pass for the complete modality samples ####
                    for i, blk in enumerate(self.transformer.blocks):
                        combined_embeds_compl, _attn = blk(combined_embeds_compl, mask=combined_embeds_compl_mask)

                    combined_embeds_compl = self.transformer.norm(combined_embeds_compl)
                    bn_image_feats_compl=combined_embeds_compl[:, compl_text_embeds.shape[1]+self.num_prompts:] #first part of the bottleneck tokens
                    bn_text_feats_compl = combined_embeds_compl[:, self.num_prompts:self.num_prompts+compl_text_embeds.shape[1]]


                if(self.mod_choice=='image'):

                    mmod_text_feats,mmod_image_embeds, image_mmod_mask_len= self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       self.text_prompt_embeddings,
                                                       image_token_type_idx
                                                       )

                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hs).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_feats

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(image_mmod_mask_len,bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]

                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hs).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    pool_token=torch.zeros(text_feats.shape[0],self.hs).to(self.config['device'])
                    pool_token[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    pool_token[id_modality_dropped,:]=text_feats_avg[id_modality_dropped,:]


                elif(self.mod_choice=='text'):

                    mmod_text_embeds,mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       self.image_prompt_embeddings,
                                                       image_token_type_idx,
                                                        mask_image
                                                       )      

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(mmod_image_feats.shape[1],bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_feats=mmod_image_feats[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]

                    #image features 
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hs).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_feats

                    #create tensor for text features for both missing and complete modality samples
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hs).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    pool_token=torch.zeros(text_feats.shape[0],self.hs).to(self.config['device'])
                    pool_token[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    pool_token[id_modality_dropped,:]=image_feats_avg[id_modality_dropped,:]

            #pass it to classifier
        cls_logits=self.classifier(pool_token)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits

###### deep version of prompting with multiple prompts per layer and specific to different types of modalities #####        
class ViLT_multi_deep_prompt_missing_mod_token_average(pl.LightningModule):

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
        
        #text_embeddings declared here
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.mod_dropout_flag=config['mod_dropout_flag']
        self.mod_choice=config['mod_choice']
        self.initialization=config['initialization']
        self.hs=self.hparams.config["hidden_size"]
        self.num_prompts=self.hparams.config["num_prompts"]
        self.patch_size=self.hparams.config["patch_size"]
        self.layers=self.hparams.config["num_layers"]
        self.prompt_drop_rate=self.hparams.config["prompt_drop_rate"]
        self.total_num_layers=self.layers-1

        #token type embeddings for the text part
        self.token_type_embeddings = nn.Embedding(2, self.hs)
        self.token_type_embeddings.apply(objectives.init_weights)

        ########################## keeping number of prompts same for now ############################  
        ############################prompt embeddings for the text part (with image missing)#########################################
        self.text_prompt_deep_embeddings = nn.Parameter(torch.zeros(self.total_num_layers,self.num_prompts,self.hs),requires_grad=True)
        self.text_prompt_dropout=nn.Dropout(self.prompt_drop_rate)
        #prompt embeddings for the text part (with image missing) -- initial prompt
        self.text_prompt_init_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts,self.hs),requires_grad=True)

        #####################################prompt embeddings for the image part (with text missing) #######################################
        self.image_prompt_deep_embeddings = nn.Parameter(torch.zeros(self.total_num_layers,self.num_prompts,self.hs),requires_grad=True)
        self.image_prompt_dropout=nn.Dropout(self.prompt_drop_rate)
        #prompt embeddings for the image part (with text missing) -- initial prompt
        self.image_prompt_init_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts,self.hs),requires_grad=True)

        ################################################prompt embeddings for complete samples ##########################################
        self.complete_prompt_deep_embeddings = nn.Parameter(torch.zeros(self.total_num_layers,self.num_prompts,self.hs),requires_grad=True)
        self.complete_prompt_dropout=nn.Dropout(self.prompt_drop_rate)
        #prompt embeddings for the image part with text part
        self.complete_prompt_init_embeddings = nn.Parameter(torch.zeros(1,self.num_prompts,self.hs),requires_grad=True)

        ### initialize the weights using xavier,kaiming,normal initialization ###
        self.initialize_weights(option=self.initialization, 
                            patch_size=self.patch_size, 
                            prompt_dim=self.hs)

        #number of classes
        self.num_classes = config["num_classes"]

        #patch size
        self.patch_size = self.hparams.config["patch_size"]
        
        # transformer 
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False)

        #classifier declaration
        self.classifier = nn.Sequential(
                nn.Linear(self.hs, self.hs), #hs -> 2*hs
                nn.LayerNorm(self.hs), 
                nn.GELU(),
                nn.Linear(self.hs, self.num_classes)
            )

        #classifier initialization
        self.classifier.apply(objectives.init_weights)

    #initialize the weights of the Parameters
    def initialize_weights(self,option='xavier',patch_size=32,prompt_dim=768):
            if(option=='xavier'):
                
                #initialize the text_prompt_embeddings
                nn.init.xavier_uniform_(self.text_prompt_deep_embeddings.data)
                nn.init.xavier_uniform_(self.text_prompt_init_embeddings.data)

                #initialize the image_prompt_embeddings
                nn.init.xavier_uniform_(self.image_prompt_deep_embeddings.data)
                nn.init.xavier_uniform_(self.image_prompt_init_embeddings.data)

                #initialize the complete_prompt_embeddings
                nn.init.xavier_uniform_(self.complete_prompt_deep_embeddings.data)
                nn.init.xavier_uniform_(self.complete_prompt_init_embeddings.data)


            elif(option=='kaiming'):
                
                #initialize the text_prompt_embeddings
                nn.init.kaiming_uniform_(self.text_prompt_deep_embeddings.data)
                nn.init.kaiming_uniform_(self.text_prompt_init_embeddings.data)

                #initialize the image_prompt_embeddings
                nn.init.kaiming_uniform_(self.image_prompt_deep_embeddings.data)
                nn.init.kaiming_uniform_(self.image_prompt_init_embeddings.data)

                #initialize the complete_prompt_embeddings
                nn.init.kaiming_uniform_(self.complete_prompt_deep_embeddings.data)
                nn.init.kaiming_uniform_(self.complete_prompt_init_embeddings.data)

            elif (option=='normal'):
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
                #initialize the text_prompt_embeddings
                nn.init.normal_(self.text_prompt_deep_embeddings.data,(-val,val))
                nn.init.normal_(self.text_prompt_init_embeddings.data,(-val,val))

                #initialize the image_prompt_embeddings
                nn.init.normal_(self.image_prompt_deep_embeddings.data,(-val,val))
                nn.init.normal_(self.image_prompt_init_embeddings.data,(-val,val))

                #initialize the complete_prompt_embeddings
                nn.init.normal_(self.complete_prompt_deep_embeddings.data,(-val,val))
                nn.init.normal_(self.complete_prompt_init_embeddings.data,(-val,val))

    def handle_missing_image_modality(self,mmod_images,
                                        mmod_text_embeds,
                                        mmod_text_masks,
                                        image_token_type_idx
                                        ):
            #handle missing image modalities using the text prompt embeddings
            ####################################### creating the dummy tensort for the image tokens ########################################
            #maximum height of the images from the missing modality batches
            max_height= max([img.shape[1] for img in mmod_images])
            #maximum width of the images from the missing modality batches
            max_width= max([img.shape[2] for img in mmod_images])

            #num width patches
            num_width_patches=(max_width//self.hparams.config['patch_size'])
            #num height patches
            num_height_patches=(max_height//self.hparams.config['patch_size'])

            #make zero tensors of (batch size, num_height_patches x num_width_patches, hidden_size)
            image_mmod_mask_len=num_height_patches*num_width_patches
            mmod_image_embeds=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,self.hs).to(self.config['device'])

            ####################################### creating the dummy tensort for the image masks ########################################
            #make zero tensors of (batch size, num_height_patches x num_width_patches)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_images.shape[0],image_mmod_mask_len,dtype=torch.long).to(self.config['device'])
                    
            #include token type embeddings for the image tokens
            mmod_text_embeds, mmod_image_embeds = (
                        mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                        mmod_image_embeds+ self.token_type_embeddings(torch.full_like(image_mmod_mask_fwd_pass, image_token_type_idx))
                    )

            #concatenate the text and image embeddings with prompt embeddings for the text (used with image modality missing)
            image_mmod_mask_fwd_pass=torch.zeros(mmod_image_embeds.shape[0],image_mmod_mask_len).to(self.config['device'])
            text_prompt_mask_fwd_pass=torch.ones(mmod_text_embeds.shape[0],self.num_prompts).to(self.config['device'])

            #elif(self.prompt_concat_option=="prepend"):

            ##### prepend the inital prompt embeddings to the text and image embeddings #####
            combined_img_mmod_embeds_init=torch.cat((self.text_prompt_dropout(self.text_prompt_init_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_text_embeds,
                                mmod_image_embeds
                                ),dim=1)
            total_mmod_mask_fwd_pass_image=torch.cat([text_prompt_mask_fwd_pass,
                                            mmod_text_masks,
                                            image_mmod_mask_fwd_pass],dim=1)
                                            
            #forward pass for text and bottleneck for the missing modality samples
            for i in range(self.num_layers):

                if(i==0):
                    combined_img_mmod_embeds, _attn = self.transformer.blocks[i](combined_img_mmod_embeds_init, mask=total_mmod_mask_fwd_pass_image)
                else:
                    #add the prompts for specific layers
                    if(i<=self.text_prompt_deep_embeddings.shape[0]):

                        deep_text_prompt_embed= self.text_prompt_dropout(self.text_prompt_deep_embeddings[i-1].expand(mmod_text_embeds.shape[0],-1,-1))
                        combined_img_mmod_embeds=torch.cat((deep_text_prompt_embed,
                                                            combined_img_mmod_embeds[:,self.num_prompts:self.num_prompts+mmod_text_embeds.shape[0],:],
                                                            combined_img_mmod_embeds[:,self.num_prompts+mmod_text_embeds.shape[0]:,:]
                                                            ),dim=1)

                        #forward pass for text and bottleneck for the missing modality samples
                    combined_img_mmod_embeds, _attn = self.transformer.blocks[i](combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
            
            combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)
            mmod_text_feats = combined_img_mmod_embeds[:, self.num_prompts:self.num_prompts+mmod_text_embeds.shape[1]]

            return mmod_text_feats, mmod_image_embeds, image_mmod_mask_len

    def handle_missing_text_modality(self,mmod_images,
                                mmod_text_embeds,
                                mmod_text_masks,
                                image_token_type_idx,
                                mask_image
                                ):

            (mmod_image_embeds, mmod_image_masks,
            patch_index,image_labels) = self.transformer.visual_embed(
                    mmod_images,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image
            )

            #include token type embeddings for the text and image tokens
            mmod_text_embeds, mmod_image_embeds = (
                    mmod_text_embeds + self.token_type_embeddings(torch.zeros_like(mmod_text_masks)),
                    mmod_image_embeds + self.token_type_embeddings(torch.full_like(mmod_image_masks, image_token_type_idx))
            )

            text_mmod_mask_fwd_pass=torch.zeros(mmod_text_embeds.shape[0],self.hparams.config['max_len']).to(self.config['device'])
            image_prompt_mask_fwd_pass=torch.ones(mmod_image_embeds.shape[0],self.num_prompts).to(self.config['device'])

            #initial embeddings 
            combined_text_mmod_embeds_init=torch.cat((self.image_prompt_dropout(self.image_prompt_init_embeddings.expand(mmod_text_embeds.shape[0],-1,-1)),
                                mmod_text_embeds,
                                mmod_image_embeds
                                ),dim=1)
            total_mmod_mask_fwd_pass_text=torch.cat([image_prompt_mask_fwd_pass,text_mmod_mask_fwd_pass,mmod_image_masks],dim=1)


            for i in range(self.layers):
                if(i==0):
                    combined_text_mmod_embeds, _attn = self.transformer.blocks[i](combined_text_mmod_embeds_init, mask=total_mmod_mask_fwd_pass_text)
                else:
                    if(i<=self.image_prompt_deep_embeddings.shape[0]):
                        deep_image_prompt_embed= self.image_prompt_dropout(self.image_prompt_deep_embeddings[i-1].expand(mmod_text_embeds.shape[0],-1,-1))
                        combined_text_mmod_embeds=torch.cat((deep_image_prompt_embed,
                                                            combined_text_mmod_embeds[:,self.num_prompts:self.num_prompts+mmod_text_embeds.shape[0],:],
                                                            combined_text_mmod_embeds[:,self.num_prompts+mmod_text_embeds.shape[0]:,:]
                                                            ),dim=1)
                    
                    combined_text_mmod_embeds, _attn = self.transformer.blocks[i](combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

        
            combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)
            mmod_image_feats = combined_text_mmod_embeds[:, self.num_prompts+mmod_text_embeds.shape[1]:]

            return mmod_text_embeds, mmod_image_feats

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        #print(self.complete_prompt_embeddings )
        #text embedding portion 
        text_ids = batch["text_ids"]

        #text masks and modality dropout flag
        text_masks = batch[f"text_masks"]
        mod_drop_flag=batch[f"mod_drop_flag"] #modality dropout flag

        #read the text embeddings
        text_embeds = self.text_embeddings(text_ids)

        #read the images 
        img = batch["image"]

        ## masks for complete prompt embeddings ##
        complete_prompt_masks=torch.ones(text_embeds.shape[0],
                        self.num_prompts).to(self.config['device']) #always ones

        if(self.mod_dropout_flag==False or (not any (mod_drop_flag)) ): #no modality dropout .... no need to use modality dropout mask
            
            #image embeddings, image masks
            image_embeds,image_masks,patch_index,image_labels = self.transformer.visual_embed(img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                )

            #text embeddings + token type embeddings, image embeddings + token type embeddings
            text_embeds, image_embeds = (
                    text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                    image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            )

            complete_embeds_init=torch.cat((self.complete_prompt_dropout(self.complete_prompt_deep_embeddings.expand(text_embeds.shape[0],-1,-1)),
                                text_embeds,
                                image_embeds
                                ),dim=1)
            complete_masks=torch.cat((complete_prompt_masks,text_masks,image_masks),dim=1)

            # for i, blk in enumerate(self.transformer.blocks):
            #     complete_embeds, _attn = blk(complete_embeds, mask=complete_masks)

            for i in range(self.layers):
                if(i==0):
                    combined_embeds, _attn = self.transformer.blocks[i](combined_embeds_init, mask=complete_masks)
                else:
                    if(i<=self.complete_prompt_deep_embeddings.shape[0]):
                        
                        deep_complete_prompt_embed= self.complete_prompt_dropout(self.complete_prompt_deep_embeddings[i-1].expand(text_embeds.shape[0],-1,-1))
                        combined_embeds=torch.cat((deep_complete_prompt_embed,
                                                            combined_embeds[:,self.num_prompts:self.num_prompts+text_embeds.shape[0],:],
                                                            combined_embeds[:,self.num_prompts+text_embeds.shape[0]:,:]
                                                            ),dim=1)

                    combined_embeds, _attn = self.transformer.blocks[i](combined_embeds, mask=complete_masks)

            #take the first token here ###
            pool_token=combined_embeds[:,0,:] #taking CLS of the combined sequence 
            #print(pool_token.size())

        else:

            #missing modality scenario ####
            index_modality_dropped=torch.where(mod_drop_flag==True)[0] #index of the samples where modality is dropped
            index_complete_modality=torch.where(mod_drop_flag==False)[0] #index of the samples where modality is not dropped
            id_modality_dropped=index_modality_dropped.tolist()
            id_modality_complete=index_complete_modality.tolist()

            #filtered segment containing the missing modality samples 
            mmod_images=img[id_modality_dropped,:]  #image samples where modality is dropped
            mmod_text_embeds=text_embeds[id_modality_dropped,:] 
            mmod_text_masks=text_masks[id_modality_dropped,:]

            #part of the filtered segment containing the complete modality samples
            compl_images=img[id_modality_complete,:] #image samples where modality is not dropped (complete modality)
            compl_text_embeds=text_embeds[id_modality_complete,:]
            compl_text_masks=text_masks[id_modality_complete,:]
            
            if(len(id_modality_complete)==0):

                    if(self.mod_choice=="image"):

                        # text_prompt_embeddings=self.text_prompt_dropout(self.text_prompt_embeddings.expand(mmod_images.shape[0],-1,-1))

                        mmod_text_feats,mmod_image_embeds,image_mmod_mask_len = self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       image_token_type_idx
                                                       )      
                        pool_token=torch.mean(mmod_text_feats,dim=1) #average pooling of the text features
                        #we can take the first token here as well

                    elif(self.mod_choice=="text"):

                        # image_prompt_embeddings=self.image_prompt_dropout(self.image_prompt_embeddings.expand(mmod_images.shape[0],-1,-1))
                        mmod_text_embeds,mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       image_token_type_idx,
                                                       mask_image
                                                       )      
                        pool_token=torch.mean(mmod_image_feats,dim=1)

            else:

                #attention between image, text and complete prompt embeddings
                (
                    compl_image_embeds,
                    compl_image_masks,
                        patch_index,
                        image_labels,
                ) = self.transformer.visual_embed(
                        compl_images,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image
                )

                #include token type embeddings for the text and image tokens
                compl_text_embeds, compl_image_embeds = (

                    compl_text_embeds + self.token_type_embeddings(torch.zeros_like(compl_text_masks)),
                    compl_image_embeds + self.token_type_embeddings(torch.full_like(compl_image_masks, image_token_type_idx))
                )
                
                #complete masks for the prompts###
                compl_prompt_masks=torch.ones(compl_text_embeds.shape[0],self.num_prompts).to(self.config['device']) #always ones
                #concatenate the text and image embeddings with the complete prompt embeddings

                combined_embeds_compl_init=torch.cat((self.complete_prompt_dropout(self.complete_prompt_init_embeddings.expand(compl_text_embeds.shape[0],-1,-1)),
                                    compl_text_embeds,
                                    compl_image_embeds
                                    ),dim=1)
                    
                # complete embeddings + complete masks #
                combined_embeds_compl_mask=torch.cat((compl_prompt_masks,compl_text_masks,compl_image_masks),dim=1)

                #### first forward pass for the complete modality samples ####
                #### first forward pass for the complete modality samples ####
                for i in range(self.layers):
                    if(i==0):
                        combined_embeds_compl, _attn = self.transformer.blocks[i](combined_embeds_compl_init, mask=combined_embeds_compl_mask)
                    else:
                        if(i<=self.complete_prompt_deep_embeddings.shape[0]):
                            deep_complete_prompt_embed= self.complete_prompt_dropout(self.complete_prompt_deep_embeddings[i-1].expand(compl_text_embeds.shape[0],-1,-1))
                            combined_embeds_compl=torch.cat((deep_complete_prompt_embed,
                                                                combined_embeds_compl[:,self.num_prompts:self.num_prompts+text_embeds.shape[0],:],
                                                                combined_embeds_compl[:,self.num_prompts+text_embeds.shape[0]:,:]
                                                                ),dim=1)

                        combined_embeds_compl, _attn = self.transformer.blocks[i](combined_embeds_compl, mask=combined_embeds_compl_mask)


                combined_embeds_compl = self.transformer.norm(combined_embeds_compl)
                bn_image_feats_compl=combined_embeds_compl[:, compl_text_embeds.shape[1]+self.num_prompts:] #first part of the bottleneck tokens
                bn_text_feats_compl = combined_embeds_compl[:, self.num_prompts:self.num_prompts+compl_text_embeds.shape[1]]


                if(self.mod_choice=='image'):

                    mmod_text_feats,mmod_image_embeds, image_mmod_mask_len= self.handle_missing_image_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       image_token_type_idx
                                                       )

                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hs).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_feats

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(image_mmod_mask_len,bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_embeds=mmod_image_embeds[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]

                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hs).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    pool_token=torch.zeros(text_feats.shape[0],self.hs).to(self.config['device'])
                    pool_token[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    pool_token[id_modality_dropped,:]=text_feats_avg[id_modality_dropped,:]


                elif(self.mod_choice=='text'):

                    mmod_text_embeds,mmod_image_feats = self.handle_missing_text_modality(mmod_images,
                                                       mmod_text_embeds,
                                                       mmod_text_masks,
                                                       image_token_type_idx,
                                                        mask_image
                                                       )      

                    #dummy tensor where all the image features are stored
                    image_embeds_shape=min(mmod_image_feats.shape[1],bn_image_feats_compl.shape[1])
                    if(image_embeds_shape==compl_image_embeds.shape[1]):
                        #truncate the mmod_image_embeds
                        mmod_image_feats=mmod_image_feats[:,:image_embeds_shape,:]
                    else:
                        #truncate the compl_image_embeds
                        bn_image_feats_compl=bn_image_feats_compl[:,:image_embeds_shape,:]

                    #image features 
                    image_feats=torch.zeros(text_embeds.shape[0],image_embeds_shape,self.hs).to(self.config['device'])
                    image_feats[id_modality_complete,:]=bn_image_feats_compl
                    image_feats[id_modality_dropped,:]=mmod_image_feats

                    #create tensor for text features for both missing and complete modality samples
                    text_feats=torch.zeros(text_embeds.shape[0],text_embeds.shape[1],self.hs).to(self.config['device'])
                    text_feats[id_modality_complete,:]=bn_text_feats_compl
                    text_feats[id_modality_dropped,:]=mmod_text_embeds

                    #for complete modality cases 
                    image_feats_avg=torch.mean(image_feats,dim=1)
                    text_feats_avg=torch.mean(text_feats,dim=1)

                    #average the cases from image_feats_avg and text_feats_avg when where both the modalities are present
                    pool_token=torch.zeros(text_feats.shape[0],self.hs).to(self.config['device'])
                    pool_token[id_modality_complete,:]=0.5*(image_feats_avg[id_modality_complete,:]+text_feats_avg[id_modality_complete,:])
                    pool_token[id_modality_dropped,:]=image_feats_avg[id_modality_dropped,:]

            #pass it to classifier
        cls_logits=self.classifier(pool_token)

        return(cls_logits)

    def forward(self, batch):

        cls_logits=self.infer(batch)

        return cls_logits

        





        













                





            






            

            

    

    


    

























    
