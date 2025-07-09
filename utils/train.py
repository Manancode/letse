import os
import math
import torch
import torch.nn as nn
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilterLite
from model.embedder import SpeechEmbedder


def asymmetric_l2_loss(pred, target, alpha):
    """
    VoiceFilter-Lite 2020 Asymmetric L2 Loss
    EXACT PAPER FORMULA: 
    gasym(x, α) = x if x ≤ 0, α·x if x > 0
    Lasym = ∑∑ gasym(Scln(t,f) - Senh(t,f), α)²
    
    Args:
        pred: Enhanced features [B, T, D]
        target: Clean features [B, T, D]  
        alpha: Asymmetric parameter (α > 1, paper uses α=10)
    """
    # Note: Paper uses (clean - enhanced), we use (target - pred)
    diff = target - pred
    
    # Apply gasym function FIRST, then square
    gasym_diff = torch.where(diff <= 0, diff, alpha * diff)
    loss = gasym_diff ** 2
    
    return loss.mean()


def noise_type_loss(noise_pred, target_noise_type):
    """
    Loss for noise type prediction (speech vs non-speech)
    Uses cross-entropy loss for classification
    """
    criterion = nn.CrossEntropyLoss()
    # Reshape for cross-entropy: [B*T, C]
    noise_pred_flat = noise_pred.view(-1, noise_pred.size(-1))
    target_flat = target_noise_type.view(-1)
    return criterion(noise_pred_flat, target_flat)


def generate_noise_labels(target_features, mixed_features):
    """
    Generate noise type labels for training
    Simple heuristic: speech noise if target and mixed differ significantly
    """
    batch_size, time_steps, _ = target_features.shape
    
    # Calculate energy difference between target and mixed
    target_energy = torch.sum(target_features**2, dim=-1)  # [B, T]
    mixed_energy = torch.sum(mixed_features**2, dim=-1)    # [B, T]
    
    # If mixed has significantly more energy, likely speech interference
    energy_ratio = mixed_energy / (target_energy + 1e-8)
    
    # Label 0: Non-speech noise, Label 1: Speech noise
    noise_labels = (energy_ratio > 1.5).long()  # Threshold-based labeling
    
    return noise_labels


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilterLite(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        # VoiceFilter-Lite 2020: Use asymmetric L2 loss
        alpha = hp.train.asymmetric_loss_alpha  # 2.0 from paper
        noise_pred_weight = hp.train.noise_pred_loss_weight  # 0.1
        
        logger.info(f"Using VoiceFilter-Lite 2020 training setup:")
        logger.info(f"  - Asymmetric L2 loss with alpha={alpha}")
        logger.info(f"  - Noise prediction loss weight={noise_pred_weight}")
        
        while True:
            model.train()
            # VoiceFilter-Lite 2020: Use features instead of magnitude spectrograms
            for dvec_mels, target_features, mixed_features in trainloader:
                target_features = target_features.cuda()
                mixed_features = mixed_features.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                # VoiceFilter-Lite 2020: Get dual outputs
                mask, noise_pred = model(mixed_features, dvec)
                
                # Apply mask to mixed features
                # Note: mask is 128D, features are 384D (stacked)
                enhanced_features = model._apply_mask_to_stacked_features(mixed_features, mask)

                # VoiceFilter-Lite 2020: For loss calculation, we need to compare like-with-like
                # Convert target_features to stacked format or compare in base filterbank space
                if target_features.shape[-1] == 128:  # target is base filterbank
                    # Stack target features to match enhanced_features dimension
                    batch_size, time_steps, base_dim = target_features.shape
                    target_stacked = target_features.unsqueeze(2).repeat(1, 1, 3, 1)  # [B, T, 3, 128]
                    target_stacked = target_stacked.view(batch_size, time_steps, -1)  # [B, T, 384]
                    separation_loss = asymmetric_l2_loss(enhanced_features, target_stacked, alpha)
                else:  # target is already stacked
                    separation_loss = asymmetric_l2_loss(enhanced_features, target_features, alpha)
                
                # Generate noise type labels for training
                noise_labels = generate_noise_labels(target_features, mixed_features)
                
                # Noise type prediction loss
                noise_loss = noise_type_loss(noise_pred, noise_labels)
                
                # Combined loss
                total_loss = separation_loss + noise_pred_weight * noise_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                step += 1

                # Loss monitoring
                total_loss_val = total_loss.item()
                separation_loss_val = separation_loss.item()
                noise_loss_val = noise_loss.item()
                
                if total_loss_val > 1e8 or math.isnan(total_loss_val):
                    logger.error("Loss exploded to %.02f at step %d!" % (total_loss_val, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    writer.log_training(total_loss_val, step)
                    # Log individual loss components
                    if hasattr(writer, 'log_training_detailed'):
                        writer.log_training_detailed({
                            'total_loss': total_loss_val,
                            'separation_loss': separation_loss_val,
                            'noise_prediction_loss': noise_loss_val,
                        }, step)
                    
                    logger.info(f"Step {step}: Total={total_loss_val:.4f}, "
                              f"Sep={separation_loss_val:.4f}, Noise={noise_loss_val:.4f}")

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    validate(audio, model, embedder, testloader, writer, step, hp)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
