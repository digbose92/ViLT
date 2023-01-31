import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

########################################################### VERSION UTILIZES IMAGE AND TEXT TOKENS (AFTER ALL OPERATIONS) TO GENERATE FINAL PREDICTIONS ###########################################################
class ViLTTransformerBN_token_average(pl.LightningModule): #uilize the token information from visual and text modalities instead of bottleneck tokens for final predictions
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
                    combined_img_mmod_embeds, _attn = blk(combined_img_mmod_embeds, mask=total_mmod_mask_fwd_pass_image)
                
                combined_img_mmod_embeds = self.transformer.norm(combined_img_mmod_embeds)
                mmod_text_feats = combined_img_mmod_embeds[:, 0:mmod_text_embeds.shape[1]] #text features for the missing modality samples

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
                    combined_text_mmod_embeds, _attn = blk(combined_text_mmod_embeds, mask=total_mmod_mask_fwd_pass_text)

                combined_text_mmod_embeds = self.transformer.norm(combined_text_mmod_embeds)
                
                #extract the image embeddings from co_text_mmod_bn_embeds
                mmod_image_feats = combined_text_mmod_embeds[:, mmod_text_embeds.shape[1]:mmod_text_embeds.shape[1]+mmod_image_embeds.shape[1]]
                ####sanity check #####
                ##text embeds from combined text mmod embeds and original text embeds should be same 
                # text_embeds_after=combined_text_mmod_embeds[:, :mmod_text_embeds.shape[1]]
                #assert torch.all(torch.eq(text_embeds_after,mmod_text_embeds))
                
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