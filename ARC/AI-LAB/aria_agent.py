"""
ARIA: Adaptive Reasoning & Imagination Agent
Complete embodied AI system with multimodal reasoning, world models, planning, and RL

Architecture:
INPUT → PERCEPTION → UNDERSTANDING → REASONING → PLANNING → RL → OUTPUT

Components:
- Vision2DEncoder (ViT for RGB)
- Vision3DEncoder (PointNet for point clouds)
- SceneGraphGenerator (detect objects + relationships)
- MultimodalFusion (align vision + language)
- SceneGraphReasoner (GNN over relationships)
- StateEncoder (compact state representation)
- DynamicsModel (learn environment dynamics)
- ModelPredictiveControl (plan using world model)
- ActorCritic (RL policy and value function)
- ARIAAgent (complete integrated system)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Optional
import json
import os
from datetime import datetime


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    """Get GPU device if available"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# COMPONENT 1: VISION ENCODERS
# ============================================================================

class Vision2DEncoder(nn.Module):
    """
    Encodes RGB images to visual features using Vision Transformer
    Input: (B, 3, 224, 224) - RGB image
    Output: (B, num_patches, hidden_dim) - visual features
    """
    def __init__(self, image_size=224, patch_size=16, hidden_dim=768, depth=6, heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional encoding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) RGB image
        Returns:
            features: (B, num_patches, hidden_dim)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 224, 224) -> (B, hidden_dim, 14, 14)
        x = self.patch_embedding(x)  # (B, hidden_dim, 14, 14)
        
        # Flatten patches: (B, hidden_dim, 14, 14) -> (B, 196, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token: (B, 196, hidden_dim) -> (B, 197, hidden_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.positional_embedding
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Return all patches (excluding CLS for features)
        return x[:, 1:, :]  # (B, num_patches, hidden_dim)


class Vision3DEncoder(nn.Module):
    """
    Encodes point clouds to 3D features using PointNet architecture
    Input: (B, N, 3) - point cloud with N points
    Output: (B, hidden_dim) - global 3D features
    """
    def __init__(self, output_dim=1024):
        super().__init__()
        self.output_dim = output_dim
        
        # Input transformation (T-Net for 3x3 rotation)
        self.input_transform = self._build_transform_net(3, 3)
        
        # Feature transformation (T-Net for 64x64)
        self.feature_transform = self._build_transform_net(64, 64)
        
        # First feature extraction: 3 -> 64 -> 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Feature transform applied here (in forward)
        
        # Second feature extraction: 64 -> 128 -> 1024
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn3 = nn.BatchNorm1d(output_dim)
    
    def _build_transform_net(self, in_dim, out_dim):
        """Build T-Net for transformation matrices"""
        return nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * out_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            features: (B, output_dim) global features
        """
        B, N, C = x.shape
        
        # Transpose to (B, 3, N) for conv1d
        x = x.transpose(1, 2)
        
        # Input transform
        T = self.input_transform(x)
        T = T.view(B, 3, 3)
        x = torch.bmm(x.transpose(1, 2), T).transpose(1, 2)
        
        # First feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform
        T_feat = self.feature_transform(x)
        T_feat = T_feat.view(B, 64, 64)
        x = torch.bmm(x.transpose(1, 2), T_feat).transpose(1, 2)
        
        # Second feature extraction
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pool
        x = torch.max(x, dim=2)[0]  # (B, output_dim)
        
        return x


# ============================================================================
# COMPONENT 2: SCENE UNDERSTANDING
# ============================================================================

class SceneGraphGenerator(nn.Module):
    """
    Generates scene graphs from visual features
    - Detects objects
    - Extracts relationships
    - Creates graph representation
    
    Simplified version: uses features to predict object and relationship embeddings
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_objects=20, num_relations=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.num_relations = num_relations
        
        # Object detection: predict object features from visual features
        self.object_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relationship detection: predict pairwise relationships
        self.relationship_detector = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, num_patches, input_dim) from vision encoder
        Returns:
            scene_graph: dict with nodes and edges
                - nodes: (B, num_patches, hidden_dim) object features
                - edges: (B, num_patches, num_patches, hidden_dim) relationship features
                - adjacency: (B, num_patches, num_patches) relationship types
        """
        B, num_patches, _ = visual_features.shape
        
        # Detect objects: each patch is an object
        objects = self.object_detector(visual_features)  # (B, num_patches, hidden_dim)
        
        # Detect relationships between objects
        edges = []
        for i in range(num_patches):
            for j in range(num_patches):
                if i != j:
                    pair = torch.cat([objects[:, i, :], objects[:, j, :]], dim=-1)
                    rel = self.relationship_detector(pair)
                    edges.append(rel)
        
        # Stack edges
        edges = torch.stack(edges, dim=1)  # (B, num_pairs, hidden_dim)
        
        # Create adjacency matrix (simplified: all connected)
        adjacency = torch.ones(B, num_patches, num_patches, device=objects.device)
        
        scene_graph = {
            'nodes': objects,  # (B, num_patches, hidden_dim)
            'edges': edges,    # (B, num_pairs, hidden_dim)
            'adjacency': adjacency
        }
        
        return scene_graph


class MultimodalFusion(nn.Module):
    """
    Fuses vision and language features using cross-attention
    Aligns visual understanding with language instructions
    """
    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-attention: vision attends to language
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, visual_features, language_features):
        """
        Args:
            visual_features: (B, num_patches, hidden_dim)
            language_features: (B, lang_dim) or (B, num_words, hidden_dim)
        Returns:
            fused: (B, num_patches, hidden_dim)
        """
        # Ensure language features are 3D: (B, seq_len, hidden_dim)
        if language_features.dim() == 2:
            language_features = language_features.unsqueeze(1)
        
        # Cross-attention: visual queries, language keys/values
        attended, _ = self.cross_attention(
            visual_features,
            language_features,
            language_features
        )
        
        # Fuse
        combined = torch.cat([visual_features, attended], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused


class SceneGraphReasoner(nn.Module):
    """
    Graph Neural Network for reasoning over scene graphs
    Message passing between objects to understand relationships
    """
    def __init__(self, hidden_dim=256, num_reasoning_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_reasoning_steps)
        ])
        
        # Update functions
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_reasoning_steps)
        ])
    
    def forward(self, scene_graph):
        """
        Args:
            scene_graph: dict with nodes and edges
        Returns:
            updated_graph: dict with reasoned representations
        """
        nodes = scene_graph['nodes']  # (B, num_nodes, hidden_dim)
        
        # Message passing iterations
        for i in range(self.num_reasoning_steps):
            # Self-attention over nodes (message passing)
            attended, _ = self.gnn_layers[i](nodes, nodes, nodes)
            
            # Update node representations
            updated = self.update_mlps[i](attended)
            nodes = nodes + updated  # Residual connection
        
        scene_graph['nodes'] = nodes
        return scene_graph


class StateEncoder(nn.Module):
    """
    Encodes all perception/reasoning into compact state representation
    Used by world model, planning, and RL
    """
    def __init__(self, 
                 scene_dim=256, 
                 language_dim=256, 
                 pos_dim=8,
                 latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encode scene (aggregate graph)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Encode language
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Encode position
        self.position_encoder = nn.Sequential(
            nn.Linear(pos_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, scene_graph, language_features, agent_position):
        """
        Args:
            scene_graph: dict with nodes
            language_features: (B, language_dim)
            agent_position: (B, pos_dim) agent's position in environment
        Returns:
            state: (B, latent_dim) compact state representation
        """
        # Aggregate scene graph (max pool over nodes)
        scene_feature = scene_graph['nodes'].max(dim=1)[0]  # (B, hidden_dim)
        scene_enc = self.scene_encoder(scene_feature)
        
        # Encode language
        lang_enc = self.language_encoder(language_features)
        
        # Encode position
        pos_enc = self.position_encoder(agent_position)
        
        # Fuse all
        combined = torch.cat([scene_enc, lang_enc, pos_enc], dim=-1)
        state = self.fusion(combined)
        
        return state


# ============================================================================
# COMPONENT 3: WORLD MODEL
# ============================================================================

class DynamicsModel(nn.Module):
    """
    Learns environment dynamics in latent space
    Given state + action, predicts next state and reward
    Uses RNN to capture sequential dependencies
    """
    def __init__(self, 
                 latent_dim=256,
                 action_dim=4,
                 hidden_dim=512,
                 num_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # RNN for dynamics
        self.rnn = nn.GRU(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Predict next state
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Predict reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Predict terminal (done flag)
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action, hidden=None):
        """
        Args:
            state: (B, latent_dim) current state
            action: (B, action_dim) action taken
            hidden: (num_layers, B, hidden_dim) RNN hidden state
        Returns:
            next_state: (B, latent_dim) predicted next state
            reward: (B, 1) predicted reward
            done: (B, 1) predicted terminal probability
            hidden: updated RNN hidden state
        """
        # Accept either a single transition ([B, D]) or a sequence
        # ([B, T, D]); the latter preserves the RNN's temporal context.
        if state.dim() not in (2, 3):
            raise ValueError(
                f"State must be [B, {self.latent_dim}] or "
                f"[B, T, {self.latent_dim}], got {state.shape}"
            )
        if action.dim() != state.dim():
            raise ValueError(
                f"State and action must have the same rank, got "
                f"{state.shape} and {action.shape}"
            )
        if state.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected state dimension {self.latent_dim}, got {state.shape}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {action.shape}"
            )
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"State and action batch dimensions must match, got "
                f"{state.shape} and {action.shape}"
            )

        sequence = state.dim() == 3
        x = torch.cat([state, action], dim=-1)
        if not sequence:
            x = x.unsqueeze(1)

        output, hidden = self.rnn(x, hidden)
        if not sequence:
            output = output.squeeze(1)

        next_state = self.state_head(output)
        reward = self.reward_head(output)
        done = torch.sigmoid(self.done_head(output))
        return next_state, reward, done, hidden

class ObservationDecoder(nn.Module):
    """
    Reconstructs observations from latent state
    Used for visualization and validation of world model
    Optional component - not required for agent operation
    """
    def __init__(self, latent_dim=256, output_size=224):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size * output_size * 3),
            nn.Sigmoid()  # Images in [0, 1]
        )
        self.output_size = output_size
    
    def forward(self, state):
        """
        Args:
            state: (B, latent_dim)
        Returns:
            image: (B, 3, output_size, output_size)
        """
        x = self.decoder(state)
        image = x.view(-1, 3, self.output_size, self.output_size)
        return image


# ============================================================================
# COMPONENT 4: PLANNING
# ============================================================================

class ModelPredictiveControl(nn.Module):
    """
    Plans by rolling out trajectories in the world model
    Selects action sequence with highest estimated value
    
    Algorithm:
    1. Sample multiple action sequences
    2. Rollout each in world model
    3. Evaluate using value function
    4. Return best action sequence
    5. Execute first action
    """
    def __init__(self, 
                 dynamics_model,
                 latent_dim=256,
                 action_dim=4,
                 horizon=5,
                 num_samples=100):
        super().__init__()
        self.dynamics = dynamics_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        
        # Value function (simple network)
        self.value_fn = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def plan(self, state, hidden=None):
        """
        Plans action sequence for given state
        
        Args:
            state: (B, latent_dim) current state
            hidden: RNN hidden state
        Returns:
            best_action: (B, action_dim) action to execute
            best_actions: (B, horizon, action_dim) full trajectory
        """
        B = state.shape[0]
        device = state.device
        
        best_actions = None
        best_value = -float('inf')
        
        # Try multiple action sequences
        for _ in range(self.num_samples):
            # Sample action sequence: (horizon, B, action_dim)
            actions = torch.randn(self.horizon, B, self.action_dim, device=device)
            actions = torch.tanh(actions)  # Clip to [-1, 1]
            
            # Rollout in world model
            trajectory_value = self._rollout(state, actions, hidden)
            
            # Keep best
            if trajectory_value > best_value:
                best_value = trajectory_value
                best_actions = actions
        
        # Execute first action
        best_action = best_actions[0] if best_actions is not None else torch.zeros(
            B, self.action_dim, device=device
        )
        
        return best_action, best_actions
    
    def _rollout(self, state, actions, hidden=None):
        """
        Simulate trajectory in learned world model
        
        Args:
            state: (B, latent_dim)
            actions: (horizon, B, action_dim)
            hidden: RNN state
        Returns:
            total_value: scalar - total value of trajectory
        """
        total_value = 0.0
        current_state = state
        gamma = 0.99
        
        for t in range(self.horizon):
            action = actions[t]
            
            # Predict next state in model
            next_state, reward, done, hidden = self.dynamics(
                current_state, action, hidden
            )
            
            # Evaluate state
            value = self.value_fn(next_state)
            
            # Discount and accumulate
            discount = gamma ** t
            total_value = total_value + discount * (reward.mean() + value.mean())
            
            current_state = next_state
        
        return total_value


# ============================================================================
# COMPONENT 5: REINFORCEMENT LEARNING
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for RL
    - Actor: learns policy (action distribution)
    - Critic: learns value function
    """
    def __init__(self, latent_dim=256, action_dim=4, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Continuous actions in [-1, 1]
        )
        
        # Value head (critic)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Args:
            state: (B, latent_dim)
        Returns:
            action: (B, action_dim) deterministic action
            value: (B, 1) state value estimate
        """
        shared = self.shared_encoder(state)
        action = self.actor_head(shared)
        value = self.critic_head(shared)
        return action, value
    
    def get_action_dist(self, state, action_std=0.1):
        """
        Get action distribution for exploration
        
        Args:
            state: (B, latent_dim)
            action_std: standard deviation for exploration
        Returns:
            mean_action: (B, action_dim) mean action
            std: scalar - action standard deviation
            value: (B, 1) value estimate
        """
        shared = self.shared_encoder(state)
        mean_action = self.actor_head(shared)
        value = self.critic_head(shared)
        return mean_action, action_std, value


class ExperienceReplay:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state).detach().cpu()
        action = torch.as_tensor(action).detach().cpu()
        reward = torch.as_tensor(reward).detach().cpu()
        next_state = torch.as_tensor(next_state).detach().cpu()
        done = torch.as_tensor(done).detach().cpu()

        # If a batch is provided, store each sample separately
        if state.dim() > 1:
            batch_size = state.size(0)

            for i in range(batch_size):
                r = reward[i].reshape(-1)[0] if reward.dim() > 0 else reward
                d = done[i].reshape(-1)[0] if done.dim() > 0 else done

                self.buffer.append((
                    state[i].clone(),
                    action[i].clone(),
                    r.clone(),
                    next_state[i].clone(),
                    d.clone()
                ))

        else:
            self.buffer.append((
                state.clone(),
                action.clone(),
                reward.reshape(-1)[0].clone(),
                next_state.clone(),
                done.reshape(-1)[0].clone()
            ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer: "
                f"{len(self.buffer)} < {batch_size}"
            )

        batch = random.sample(self.buffer, batch_size)

        states = torch.stack([x[0] for x in batch])
        actions = torch.stack([x[1] for x in batch])
        rewards = torch.stack([x[2] for x in batch]).float()
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.stack([x[4] for x in batch]).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
# ============================================================================
# COMPONENT 6: COMPLETE ARIA AGENT
# ============================================================================

class ARIAAgent(nn.Module):
    """
    ARIA: Adaptive Reasoning & Imagination Agent
    
    Complete embodied AI system integrating:
    - Perception (Vision + Scene Understanding)
    - Reasoning (Multimodal Fusion + Graph Networks)
    - World Model (Dynamics + Planning)
    - RL (Actor-Critic + Experience Replay)
    
    Input: observation + instruction
    Output: action
    """
    def __init__(self, 
                 image_size=224,
                 latent_dim=256,
                 hidden_dim=256,
                 action_dim=4,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # ===== PERCEPTION =====
        self.vision_2d = Vision2DEncoder(
            image_size=image_size,
            hidden_dim=hidden_dim,
            depth=6
        )
        self.vision_3d = Vision3DEncoder(output_dim=hidden_dim)
        
        # ===== UNDERSTANDING =====
        self.scene_graph_gen = SceneGraphGenerator(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        self.multimodal_fusion = MultimodalFusion(hidden_dim=hidden_dim)
        self.scene_reasoner = SceneGraphReasoner(hidden_dim=hidden_dim)
        
        # Language encoder (simple linear)
        self.language_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),  # Assume CLIP text encoder output
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ===== STATE ENCODING =====
        self.state_encoder = StateEncoder(
            scene_dim=hidden_dim,
            language_dim=hidden_dim,
            pos_dim=8,
            latent_dim=latent_dim
        )
        
        # ===== WORLD MODEL =====
        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # ===== PLANNING =====
        self.mpc = ModelPredictiveControl(
            dynamics_model=self.dynamics,
            latent_dim=latent_dim,
            action_dim=action_dim,
            horizon=5,
            num_samples=50
        )
        
        # ===== RL =====
        self.actor_critic = ActorCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        self.replay_buffer = ExperienceReplay(capacity=100000)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.world_model_optimizer = optim.Adam(
            list(self.dynamics.parameters()),
            lr=1e-4
        )
        
        self.to(device)
    
    def encode_instruction(self, instruction_text):
        """
        Encode text instruction to features
        
        In practice, you'd use CLIP text encoder or similar
        For now, using simple learned embedding
        
        Args:
            instruction_text: (B,) string instruction
        Returns:
            features: (B, hidden_dim)
        """
        # Placeholder: in real implementation, use CLIP
        B = len(instruction_text)
        features = torch.randn(B, 768, device=self.device)
        return self.language_encoder(features)
    
    def step(self, observation, instruction, agent_position=None):
        """
        One environment step: observation + instruction -> action
        
        Args:
            observation: dict with 'rgb' and optional 'points'
            instruction: string goal description
            agent_position: (B, 8) agent's position/state
        Returns:
            action: (B, action_dim) action to take
            state: (B, latent_dim) encoded state
        """
        with torch.no_grad():
            # Default agent position if not provided
            if agent_position is None:
                agent_position = torch.zeros(
                    observation['rgb'].shape[0], 8, device=self.device
                )
            
            # ===== PERCEPTION =====
            # Process RGB image
            rgb = observation['rgb']  # (B, 3, 224, 224)
            visual_2d = self.vision_2d(rgb)  # (B, num_patches, hidden_dim)
            
            # Process point cloud if available
            if 'points' in observation:
                points = observation['points']  # (B, N, 3)
                visual_3d = self.vision_3d(points)  # (B, hidden_dim)
                visual_3d = visual_3d.unsqueeze(1)  # (B, 1, hidden_dim)
                visual_all = torch.cat([visual_2d, visual_3d], dim=1)
            else:
                visual_all = visual_2d
            
            # ===== UNDERSTANDING =====
            # Build scene graph
            scene_graph = self.scene_graph_gen(visual_all)
            
            # Encode instruction
            instruction_features = self.encode_instruction([instruction] * rgb.shape[0])
            
            # Fuse vision and language
            fused = self.multimodal_fusion(visual_all, instruction_features)
            
            # Update scene graph with reasoning
            reasoned_graph = self.scene_reasoner(scene_graph)
            
            # ===== STATE ENCODING =====
            state = self.state_encoder(reasoned_graph, instruction_features, agent_position)
            
            # ===== PLANNING & RL DECISION =====
            # Plan using world model
            planned_action, _ = self.mpc.plan(state)
            
            # Get RL policy action
            learned_action, _ = self.actor_critic(state)
            
            # Blend planned and learned actions
            alpha = 0.6  # Weight for planning vs learning
            final_action = alpha * planned_action + (1 - alpha) * learned_action
        
        return final_action, state
    
    def update_world_model(self, batch_size=32):
        """
        Train world model on collected experiences
        
        Args:
            batch_size: mini-batch size
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Predict next state and reward
        pred_next_states, pred_rewards, pred_dones, _ = self.dynamics(
            states, actions
        )
        
        # World model loss
        state_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards.squeeze(-1), rewards.squeeze(-1))
        pred_dones = pred_dones.squeeze(-1)

        if pred_dones.dim() > 1:
            pred_dones = pred_dones[:, 0]

        dones = dones.float().view(-1)

        done_loss = F.binary_cross_entropy(
            pred_dones,
            dones
        )
        
        world_loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss
        
        # Backprop
        self.world_model_optimizer.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.world_model_optimizer.step()
        
        return {
            'world_loss': world_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item()
        }
    def update_policy(self, batch_size=32, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        states = states.to(self.device).float()
        actions = actions.to(self.device).float()
        rewards = rewards.to(self.device).float().view(-1)
        next_states = next_states.to(self.device).float()
        dones = dones.to(self.device).float().view(-1)

        # Current values
        _, values = self.actor_critic(states)

        # Next-state values
        _, next_values = self.actor_critic(next_states)

        values = values.view(-1)
        next_values = next_values.view(-1)

        # TD target
        target_values = (
            rewards
            + gamma * next_values.detach() * (1.0 - dones)
        )

        # Advantage
        advantages = target_values - values

        # Predicted actions
        predicted_actions, _ = self.actor_critic(states)

        # Actor loss
        action_loss = F.mse_loss(
            predicted_actions,
            actions
        )

        policy_loss = (
            action_loss
            * (-advantages.detach().mean())
        )

        # Critic loss
        value_loss = F.mse_loss(
            values,
            target_values
        )

        # Total loss
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }

    def encode_observation(self, observation, instruction, agent_position=None):
        with torch.no_grad():

            rgb = observation['rgb'].to(self.device)

            if agent_position is None:
                agent_position = torch.zeros(
                    rgb.shape[0],
                    8,
                    device=self.device
                )

            # Vision
            visual_2d = self.vision_2d(rgb)

            if 'points' in observation:
                points = observation['points'].to(self.device)

                visual_3d = self.vision_3d(points)
                visual_3d = visual_3d.unsqueeze(1)

                visual_all = torch.cat(
                    [visual_2d, visual_3d],
                    dim=1
                )
            else:
                visual_all = visual_2d

            # Language
            instruction_features = self.encode_instruction(
                [instruction] * rgb.shape[0]
            )

            # Multimodal fusion
            fused = self.multimodal_fusion(
                visual_all,
                instruction_features
            )

            # Scene graph should use fused features
            scene_graph = self.scene_graph_gen(fused)

            # Reasoning
            reasoned_graph = self.scene_reasoner(scene_graph)

            # State
            state = self.state_encoder(
                reasoned_graph,
                instruction_features,
                agent_position
            )

            return state
    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'world_optimizer_state': self.world_model_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_optimizer_state'])
        print(f"Model loaded from {path}")


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(agent, env, num_episodes=10, update_frequency=5):
    """
    Train ARIA using real transitions from the environment.
    """

    metrics = {
        'episode_rewards': [],
        'world_losses': [],
        'policy_losses': []
    }

    agent.train()

    for episode in range(num_episodes):

        obs = env.reset()
        instruction = env.get_instruction()

        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < 100:

            # ============================================================
            # 1. Encode current observation and select action
            # ============================================================

            action, state = agent.step(
                obs,
                instruction
            )

            # ============================================================
            # 2. Execute action in environment
            # ============================================================

            next_obs, reward, done, info = env.step(action)

            episode_reward += float(reward)

            # ============================================================
            # 3. Encode NEXT observation
            # ============================================================

            next_state = agent.encode_observation(
                next_obs,
                instruction
            )

            # ============================================================
            # 4. Store real transition
            # ============================================================

            agent.replay_buffer.add(
                state,
                action,
                torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=agent.device
                ),
                next_state,
                torch.tensor(
                    float(done),
                    dtype=torch.float32,
                    device=agent.device
                )
            )

            # ============================================================
            # 5. Train world model + policy
            # ============================================================

            if step % update_frequency == 0:

                world_metrics = agent.update_world_model(
                    batch_size=16
                )

                policy_metrics = agent.update_policy(
                    batch_size=16
                )

                if world_metrics is not None:
                    metrics['world_losses'].append(
                        world_metrics['world_loss']
                    )

                if policy_metrics is not None:
                    metrics['policy_losses'].append(
                        policy_metrics['policy_loss']
                    )

            # ============================================================
            # 6. Move to next state
            # ============================================================

            obs = next_obs

            step += 1

        metrics['episode_rewards'].append(
            episode_reward
        )

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {episode_reward:.3f} | "
            f"Steps: {step} | "
            f"Replay: {len(agent.replay_buffer)}"
        )

    return metrics

# ============================================================================
# MAIN - EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ARIA: Adaptive Reasoning & Imagination Agent")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")
    
    # Initialize agent
    print("Initializing ARIA Agent...")
    agent = ARIAAgent(
        image_size=224,
        latent_dim=256,
        hidden_dim=256,
        action_dim=4,
        device=device
    )
    print(f"Total parameters: {sum(p.numel() for p in agent.parameters()):,}\n")
    
    # Count parameters per component
    print("Parameter breakdown:")
    print(f"  Vision 2D: {sum(p.numel() for p in agent.vision_2d.parameters()):,}")
    print(f"  Vision 3D: {sum(p.numel() for p in agent.vision_3d.parameters()):,}")
    print(f"  Scene Graph: {sum(p.numel() for p in agent.scene_graph_gen.parameters()):,}")
    print(f"  Fusion: {sum(p.numel() for p in agent.multimodal_fusion.parameters()):,}")
    print(f"  Reasoner: {sum(p.numel() for p in agent.scene_reasoner.parameters()):,}")
    print(f"  State Encoder: {sum(p.numel() for p in agent.state_encoder.parameters()):,}")
    print(f"  Dynamics: {sum(p.numel() for p in agent.dynamics.parameters()):,}")
    print(f"  MPC: {sum(p.numel() for p in agent.mpc.parameters()):,}")
    print(f"  Actor-Critic: {sum(p.numel() for p in agent.actor_critic.parameters()):,}\n")
    
    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    
    # Create dummy input
    dummy_rgb = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_points = torch.randn(batch_size, 100, 3, device=device)
    dummy_pos = torch.randn(batch_size, 8, device=device)
    
    observation = {
        'rgb': dummy_rgb,
        'points': dummy_points
    }
    
    # Forward pass
    action, state = agent.step(
        observation,
        "Pick up the cup",
        agent_position=dummy_pos
    )
    
    print(f"Output shapes:")
    print(f"  Action: {action.shape}")
    print(f"  State: {state.shape}\n")
    
    # Test replay buffer
    print("Testing experience replay...")
    for _ in range(10):
        action = torch.randn(batch_size, 4, device=device)
        reward = torch.randn(batch_size, 1, device=device)
        next_state = torch.randn(batch_size, 256, device=device)
        done = torch.bernoulli(torch.ones(batch_size) * 0.1)
        
        agent.replay_buffer.add(state, action, reward, next_state, done)
    
    print(f"Replay buffer size: {len(agent.replay_buffer)}\n")
    
    # Test updates
    print("Testing world model update...")
    world_loss = agent.update_world_model(batch_size=4)
    print(f"World model loss: {world_loss}\n")
    
    print("Testing policy update...")
    policy_loss = agent.update_policy(batch_size=4)
    print(f"Policy loss: {policy_loss}\n")
    
    # Save model
    print("Saving model...")
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/aria_v0.pt")
    
    print("\n" + "=" * 80)
    print("ARIA Agent ready for training!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your environment (Habitat, PyBullet, etc.)")
    print("2. Create a wrapper that returns observations in the format:")
    print("   {'rgb': (B, 3, 224, 224), 'points': (B, N, 3)}")
    print("3. Run training loop using train_epoch(agent, env)")
    print("4. Evaluate using evaluate(agent, env)")
    print("\nFor detailed implementation, see:")
    print("- aria_agent.py: Complete agent code")
    print("- Check each module's docstring for usage details")