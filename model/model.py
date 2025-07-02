import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceFilterLite(nn.Module):
    """
    VoiceFilter-Lite 2020 Implementation
    
    PAPER SPECS:
    - Input: 128D stacked filterbank features (384D with 3x stacking)
    - Architecture: 3 uni-directional LSTM layers, 256 nodes each (for 2.2MB size)
    - Dual outputs: soft mask + noise type prediction (2x64 layers)
    - Streaming-compatible (no future context)
    - Adaptive suppression with dynamic mixing
    """
    def __init__(self, hp):
        super(VoiceFilterLite, self).__init__()
        self.hp = hp
        
        # VoiceFilter-Lite 2020 ARCHITECTURE
        # PAPER: "3 LSTM layers, each with 512 nodes" (then quantized to 2.2MB)
        self.input_dim = hp.model.input_dim  # 384 (128 filterbanks × 3 stacked frames)
        self.embedding_dim = hp.embedder.emb_dim  # 256 (d-vector dimension)
        self.lstm_dim = hp.model.lstm_dim  # 512 nodes per paper (from config)
        self.lstm_layers = hp.model.lstm_layers  # 3 layers per paper
        
        # Combined input dimension: features + speaker embedding
        combined_input_dim = self.input_dim + self.embedding_dim  # 384 + 256 = 640
        
        # 1D CNN layer as shown in Google slides architecture diagram
        # Input: [B, T, 640] → reshape to [B, 640, T] for conv1d
        self.conv1d = nn.Conv1d(
            in_channels=combined_input_dim,  # 640
            out_channels=self.lstm_dim,      # 512 
            kernel_size=3,
            padding=1
        )
        
        # PAPER: "3 LSTM layers, each with 512 nodes"
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(
                input_size=self.lstm_dim,  # CNN outputs 512, all LSTM layers use 512
                hidden_size=self.lstm_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False  # CRITICAL: No bidirectional for streaming
            ) for i in range(self.lstm_layers)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # DUAL OUTPUT SYSTEM (VoiceFilter-Lite 2020)
        # Output 1: Soft mask for speech separation
        # PAPER: "1 feedforward layer with sigmoid activation for mask prediction"
        self.mask_head = nn.Sequential(
            nn.Linear(self.lstm_dim, hp.model.mask_dim),  # 256→128 (direct)
            nn.Sigmoid()  # Soft mask values [0, 1]
        )
        
        # Output 2: Noise type prediction
        # PAPER: "2 feedforward layers, each with 64 nodes for noise type prediction"
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.lstm_dim, 64),      # 256→64 (first 64-node layer)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),                 # 64→64 (second 64-node layer)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, hp.model.noise_pred_dim),  # 64→2 (final classification)
            nn.Softmax(dim=-1)  # Probability distribution
        )
        
        # ADAPTIVE SUPPRESSION (VoiceFilter-Lite 2020)
        self.adaptive_suppression = AdaptiveSuppression(hp)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, features, dvec):
        """
        VoiceFilter-Lite 2020 Forward Pass
        
        Args:
            features: [B, T, 384] - Stacked filterbank features  
            dvec: [B, 256] - Speaker embedding (d-vector)
            
        Returns:
            mask: [B, T, 128] - Soft mask for separation
            noise_pred: [B, T, 2] - Noise type prediction (clean/non-speech vs overlapped)
        """
        batch_size, time_steps, feature_dim = features.shape
        
        # Expand d-vector to match time dimension
        dvec_expanded = dvec.unsqueeze(1).repeat(1, time_steps, 1)
        
        # Concatenate features with speaker embedding
        x = torch.cat([features, dvec_expanded], dim=-1)  # [B, T, 640]
        
        # 1D CNN processing (Google slides architecture)
        # Reshape for conv1d: [B, T, 640] → [B, 640, T]
        x = x.transpose(1, 2)  # [B, 640, T]
        x = self.conv1d(x)     # [B, 512, T]  
        x = F.relu(x)
        # Reshape back: [B, 512, T] → [B, T, 512]
        x = x.transpose(1, 2)  # [B, T, 512]
        
        # Pass through LSTM stack
        hidden_states = []
        for i, lstm_layer in enumerate(self.lstm_stack):
            x, _ = lstm_layer(x)  # [B, T, 256]
            x = self.dropout(x)
            hidden_states.append(x)
        
        # x is now [B, T, 256] after final LSTM layer
        
        # DUAL OUTPUT SYSTEM
        mask = self.mask_head(x)  # [B, T, 128]
        noise_pred = self.noise_predictor(x)  # [B, T, 2]
        
        return mask, noise_pred
    
    def separate_speech(self, features, dvec, apply_adaptive_suppression=True):
        """
        VoiceFilter-Lite 2020: Complete speech separation with adaptive suppression
        PAPER: "adaptively adjusts the suppression strength according to this prediction"
        
        Returns enhanced features for direct ASR input
        """
        mask, noise_pred = self.forward(features, dvec)
        
        # Apply mask to input features (element-wise multiplication)
        enhanced_features = self._apply_mask_to_stacked_features(features, mask)
        
        if apply_adaptive_suppression:
            # PAPER: Adaptive suppression based on noise type prediction
            final_features = self.adaptive_suppression(
                original_features=features,
                enhanced_features=enhanced_features,
                noise_pred=noise_pred
            )
        else:
            final_features = enhanced_features
        
        return final_features, noise_pred
    
    def _apply_mask_to_stacked_features(self, stacked_features, mask):
        """
        Apply 128D mask to 384D stacked features
        This requires careful handling of the stacked structure
        """
        batch_size, time_steps, _ = stacked_features.shape
        stack_size = self.hp.audio.frame_stack_size  # 3
        base_dim = self.hp.audio.num_filterbanks  # 128
        
        # Reshape stacked features to [B, T, stack_size, base_dim]
        features_reshaped = stacked_features.view(batch_size, time_steps, stack_size, base_dim)
        
        # Apply mask to each stacked frame
        mask_expanded = mask.unsqueeze(2)  # [B, T, 1, 128]
        enhanced_reshaped = features_reshaped * mask_expanded  # [B, T, 3, 128]
        
        # Reshape back to stacked format
        enhanced_features = enhanced_reshaped.view(batch_size, time_steps, -1)  # [B, T, 384]
        
        return enhanced_features


class AdaptiveSuppression(nn.Module):
    """
    VoiceFilter-Lite 2020 Adaptive Suppression Module
    
    PAPER EXACT FORMULA:
    w^(t) = β · w^(t-1) + (1-β) · (a · f_adapt(S_in^(t)) + b)
    S_out = w · S_enh + (1-w) · S_in
    
    NOISE TYPE CLASSIFICATION (PAPER):
    0 = clean speech, or containing non-speech noise
    1 = overlapped speech
    """
    def __init__(self, hp):
        super(AdaptiveSuppression, self).__init__()
        self.hp = hp
        
        # PAPER: Moving average parameters
        # w^(t) = β · w^(t-1) + (1-β) · (a · f_adapt(S_in^(t)) + b)
        self.beta = getattr(hp.adaptive_suppression, 'beta', 0.9)    # β ∈ [0,1]
        self.alpha = getattr(hp.adaptive_suppression, 'alpha', 1.0)  # a > 0
        self.bias = getattr(hp.adaptive_suppression, 'bias', 0.0)    # b ≥ 0
        
        # Validate constraints from paper
        assert 0.0 <= self.beta <= 1.0, f"β must be in [0,1], got {self.beta}"
        assert self.alpha > 0.0, f"a must be > 0, got {self.alpha}"
        assert self.bias >= 0.0, f"b must be ≥ 0, got {self.bias}"
        
        # State for streaming (moving average)
        self.register_buffer('prev_weight', torch.tensor(0.5))  # Initialize to 0.5
        
    def forward(self, original_features, enhanced_features, noise_pred):
        """
        PAPER EXACT FORMULA:
        w^(t) = β · w^(t-1) + (1-β) · (a · f_adapt(S_in^(t)) + b)
        S_out = w · S_enh + (1-w) · S_in
        
        Args:
            original_features: [B, T, 384] - Original input features
            enhanced_features: [B, T, 384] - Mask-enhanced features  
            noise_pred: [B, T, 2] - Noise type: [clean/non-speech, overlapped_speech]
            
        Returns:
            final_features: [B, T, 384] - Adaptively suppressed features
        """
        batch_size, time_steps, feature_dim = original_features.shape
        
        # Extract overlapped speech probability (index 1)
        # PAPER: Higher overlapped speech prob → stronger suppression
        overlapped_speech_prob = noise_pred[:, :, 1]  # [B, T]
        
        # PAPER EXACT FORMULA: f_adapt(S_in^(t))
        f_adapt = overlapped_speech_prob  # This is the f_adapt function
        
        # Linear transform: a · f_adapt(S_in^(t)) + b
        linear_transform = self.alpha * f_adapt + self.bias  # [B, T]
        
        # Clamp to reasonable range [0, 1]
        linear_transform = torch.clamp(linear_transform, 0.0, 1.0)
        
        # PAPER EXACT FORMULA: Moving average
        # w^(t) = β · w^(t-1) + (1-β) · (a · f_adapt(S_in^(t)) + b)
        if self.training:
            # During training, compute batch-wise weights
            current_weight = linear_transform
        else:
            # During inference, use streaming moving average (EXACT FORMULA)
            current_weight = (self.beta * self.prev_weight + 
                            (1 - self.beta) * linear_transform)
            # Update state for next frame
            self.prev_weight = current_weight.mean().detach()
        
        # PAPER EXACT FORMULA: Dynamic mixing
        # S_out = w · S_enh + (1-w) · S_in
        # Expand weight to match feature dimensions
        weight = current_weight.unsqueeze(-1)  # [B, T, 1]
        
        final_features = (weight * enhanced_features + 
                         (1 - weight) * original_features)
        
        return final_features


# Legacy compatibility alias
# This allows existing code to still work
VoiceFilter = VoiceFilterLite
