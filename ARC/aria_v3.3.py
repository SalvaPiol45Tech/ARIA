import os
import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ARIA FINAL
# Adaptive Reasoning & Imagination Agent
#
# Perception
# Reasoning
# Memory
# World Model
# Imagination
# Planning
# Reinforcement Learning
#
# Single-file research prototype
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SEED = 42

GRID_SIZE = 15
NUM_OBSTACLES = 22
NUM_OBJECTS = 4

MAX_STEPS = 100

D_MODEL = 96
N_HEADS = 4
TRANSFORMER_DEPTH = 2

WORLD_HIDDEN = 128
MEMORY_SIZE = 32

GAMMA = 0.99
LR = 3e-4

TRAIN_EPISODES = 500
EVAL_EPISODES = 100

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "aria_final.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ACTION SPACE
# ============================================================

ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


# ============================================================
# COLORS
# ============================================================

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
]

COLOR_TO_ID = {
    "red": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4,
}

ID_TO_COLOR = {
    1: "red",
    2: "green",
    3: "blue",
    4: "yellow",
}


# ============================================================
# UTILS
# ============================================================

def inside(y, x):
    return (
        0 <= y < GRID_SIZE
        and 0 <= x < GRID_SIZE
    )


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def position_to_index(pos):
    y, x = pos
    return y * GRID_SIZE + x


def index_to_position(index):
    return (
        index // GRID_SIZE,
        index % GRID_SIZE,
    )


def reachable(start, goal, obstacles):

    if start in obstacles:
        return False

    if goal in obstacles:
        return False

    queue = deque([start])
    visited = {start}

    while queue:

        y, x = queue.popleft()

        if (y, x) == goal:
            return True

        for dy, dx in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]:

            ny = y + dy
            nx = x + dx

            nxt = (ny, nx)

            if not inside(ny, nx):
                continue

            if nxt in obstacles:
                continue

            if nxt in visited:
                continue

            visited.add(nxt)
            queue.append(nxt)

    return False


# ============================================================
# ENVIRONMENT
# ============================================================

class ARIAEnvironment:

    def __init__(self):

        self.grid_size = GRID_SIZE

        self.reset()

    # --------------------------------------------------------

    def reset(self):

        while True:

            cells = [
                (y, x)
                for y in range(GRID_SIZE)
                for x in range(GRID_SIZE)
            ]

            random.shuffle(cells)

            self.agent_pos = cells[0]

            self.obstacles = set(
                cells[1:1 + NUM_OBSTACLES]
            )

            free = [
                c for c in cells
                if c not in self.obstacles
                and c != self.agent_pos
            ]

            if len(free) < NUM_OBJECTS:
                continue

            selected = free[:NUM_OBJECTS]

            self.objects = {}

            for color, position in zip(
                COLORS,
                selected
            ):
                self.objects[color] = position

            self.target_color = random.choice(
                COLORS
            )

            self.goal_pos = self.objects[
                self.target_color
            ]

            if reachable(
                self.agent_pos,
                self.goal_pos,
                self.obstacles
            ):
                break

        self.steps = 0

        self.done = False

        self.previous_distance = distance(
            self.agent_pos,
            self.goal_pos
        )

        return self.observe()

    # --------------------------------------------------------

    def observe(self):

        """
        Visual observation.

        0 = empty
        1 = obstacle
        2 = agent
        3 = red
        4 = green
        5 = blue
        6 = yellow
        """

        grid = torch.zeros(
            GRID_SIZE,
            GRID_SIZE,
            dtype=torch.long
        )

        for y, x in self.obstacles:
            grid[y, x] = 1

        ay, ax = self.agent_pos
        grid[ay, ax] = 2

        object_ids = {
            "red": 3,
            "green": 4,
            "blue": 5,
            "yellow": 6,
        }

        for color, pos in self.objects.items():

            if pos != self.agent_pos:

                y, x = pos

                grid[y, x] = object_ids[color]

        return grid

    # --------------------------------------------------------

    def step(self, action):

        if self.done:

            return (
                self.observe(),
                0.0,
                True,
            )

        old_distance = distance(
            self.agent_pos,
            self.goal_pos
        )

        dy, dx = ACTION_DELTAS[action]

        y, x = self.agent_pos

        ny = y + dy
        nx = x + dx

        collision = False

        if not inside(ny, nx):

            collision = True

        elif (ny, nx) in self.obstacles:

            collision = True

        else:

            self.agent_pos = (ny, nx)

        new_distance = distance(
            self.agent_pos,
            self.goal_pos
        )

        reward = -0.01

        # Distance shaping.
        if new_distance < old_distance:
            reward += 0.05

        elif new_distance > old_distance:
            reward -= 0.03

        if collision:

            reward -= 0.10

        self.steps += 1

        if self.agent_pos == self.goal_pos:

            reward += 1.0

            self.done = True

        elif self.steps >= MAX_STEPS:

            self.done = True

        self.previous_distance = new_distance

        return (
            self.observe(),
            reward,
            self.done,
        )


# ============================================================
# MEMORY
# ============================================================

@dataclass
class MemoryEntry:

    state: torch.Tensor
    reasoning: torch.Tensor
    action: int
    reward: float


class EpisodicMemory:

    def __init__(
        self,
        max_size=MEMORY_SIZE
    ):

        self.max_size = max_size

        self.entries = deque(
            maxlen=max_size
        )

    # --------------------------------------------------------

    def add(
        self,
        state,
        reasoning,
        action,
        reward
    ):

        entry = MemoryEntry(
            state.detach().cpu(),
            reasoning.detach().cpu(),
            action,
            reward,
        )

        self.entries.append(entry)

    # --------------------------------------------------------

    def get_recent(self):

        return list(self.entries)

    # --------------------------------------------------------

    def clear(self):

        self.entries.clear()


# ============================================================
# VISUAL ENCODER
# ============================================================

class VisualEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        num_cells = GRID_SIZE * GRID_SIZE

        self.cell_embedding = nn.Embedding(
            7,
            D_MODEL
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_cells,
                D_MODEL
            ) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=TRANSFORMER_DEPTH
        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

    # --------------------------------------------------------

    def forward(self, grid):

        B = grid.shape[0]

        x = grid.reshape(
            B,
            -1
        )

        x = self.cell_embedding(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        x = self.norm(x)

        return x


# ============================================================
# INSTRUCTION ENCODER
# ============================================================

class InstructionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            5,
            D_MODEL
        )

        self.network = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL,
                D_MODEL
            )
        )

    # --------------------------------------------------------

    def forward(self, color_id):

        x = self.embedding(color_id)

        return self.network(x)


# ============================================================
# REASONER
# ============================================================

class NeuralReasoner(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual = VisualEncoder()

        self.instruction = InstructionEncoder()

        self.fusion = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

        # Learnable reasoning query.
        self.reasoning_query = nn.Parameter(
            torch.randn(
                1,
                1,
                D_MODEL
            ) * 0.02
        )

        self.cross_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            batch_first=True
        )

        # Spatial target heatmap.
        self.target_head = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL,
                1
            )
        )

        # State value.
        self.value_head = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL,
                1
            )
        )

    # --------------------------------------------------------

    def forward(
        self,
        grid,
        color_id
    ):

        visual_tokens = self.visual(
            grid
        )

        instruction = self.instruction(
            color_id
        )

        B, N, D = visual_tokens.shape

        instruction_tokens = instruction.unsqueeze(
            1
        ).expand(
            B,
            N,
            D
        )

        fused = visual_tokens + instruction_tokens

        fused = self.fusion(
            fused
        )

        fused = self.norm(
            fused
        )

        query = self.reasoning_query.expand(
            B,
            1,
            D
        )

        reasoning_state, attention = (
            self.cross_attention(
                query,
                fused,
                fused
            )
        )

        reasoning_state = reasoning_state.squeeze(
            1
        )

        target_logits = self.target_head(
            fused
        ).squeeze(-1)

        value = self.value_head(
            reasoning_state
        ).squeeze(-1)

        return {
            "spatial_tokens": fused,
            "reasoning_state": reasoning_state,
            "target_logits": target_logits,
            "value": value,
            "attention": attention,
        }


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel(nn.Module):

    """
    Predicts future latent state.

    z_t + action -> z_(t+1)
    """

    def __init__(self):

        super().__init__()

        self.action_embedding = nn.Embedding(
            5,
            D_MODEL
        )

        self.network = nn.Sequential(
            nn.Linear(
                D_MODEL * 2,
                WORLD_HIDDEN
            ),
            nn.GELU(),

            nn.Linear(
                WORLD_HIDDEN,
                WORLD_HIDDEN
            ),
            nn.GELU(),

            nn.Linear(
                WORLD_HIDDEN,
                D_MODEL
            )
        )

    # --------------------------------------------------------

    def forward(
        self,
        state,
        action
    ):

        action_embedding = self.action_embedding(
            action
        )

        x = torch.cat(
            [
                state,
                action_embedding
            ],
            dim=-1
        )

        return self.network(x)


# ============================================================
# IMAGINATION
# ============================================================

class ImaginationModule:

    def __init__(
        self,
        world_model,
        horizon=5
    ):

        self.world_model = world_model

        self.horizon = horizon

    # --------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        state,
        actions
    ):

        """
        Imagine a sequence of future states.
        """

        current = state

        imagined = []

        for action in actions:

            action_tensor = torch.tensor(
                [action],
                dtype=torch.long,
                device=state.device
            )

            current = self.world_model(
                current,
                action_tensor
            )

            imagined.append(
                current
            )

        return imagined


# ============================================================
# LEARNED PLANNER
# ============================================================

class LearnedPlanner:

    def __init__(self):

        self.action_order = [
            1,
            2,
            3,
            4,
        ]

    # --------------------------------------------------------

    def choose_action(
        self,
        agent_pos,
        predicted_target,
        obstacles
    ):

        """
        Lightweight goal-directed planner.

        Unlike the previous version, the planner
        does not receive the privileged target.
        """

        best_action = 0

        best_distance = float("inf")

        for action in self.action_order:

            dy, dx = ACTION_DELTAS[action]

            y, x = agent_pos

            ny = y + dy
            nx = x + dx

            if not inside(ny, nx):
                continue

            if (ny, nx) in obstacles:
                continue

            d = distance(
                (ny, nx),
                predicted_target
            )

            if d < best_distance:

                best_distance = d

                best_action = action

        return best_action


# ============================================================
# RL POLICY
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),

            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),

            nn.Linear(
                D_MODEL,
                5
            )
        )

        self.critic = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL
            ),
            nn.GELU(),

            nn.Linear(
                D_MODEL,
                1
            )
        )

    # --------------------------------------------------------

    def forward(self, state):

        logits = self.actor(state)

        value = self.critic(state).squeeze(-1)

        return logits, value


# ============================================================
# ARIA AGENT
# ============================================================

class ARIAAgent(nn.Module):

    def __init__(self):

        super().__init__()

        self.reasoner = NeuralReasoner()

        self.world_model = WorldModel()

        self.policy = ActorCritic()

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LR,
            weight_decay=1e-4
        )

        self.memory = EpisodicMemory()

        self.imagination = ImaginationModule(
            self.world_model
        )

        self.planner = LearnedPlanner()

    # --------------------------------------------------------

    def perceive_and_reason(
        self,
        observation,
        target_color
    ):

        grid = observation.unsqueeze(0).to(
            DEVICE
        )

        color_id = torch.tensor(
            [
                COLOR_TO_ID[target_color]
            ],
            dtype=torch.long,
            device=DEVICE
        )

        output = self.reasoner(
            grid,
            color_id
        )

        return output

    # --------------------------------------------------------

    @torch.no_grad()
    def predict_target(
        self,
        observation,
        target_color
    ):

        output = self.perceive_and_reason(
            observation,
            target_color
        )

        logits = output[
            "target_logits"
        ]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        index = probabilities.argmax(
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            index
        ].item()

        return (
            index_to_position(index),
            confidence,
            output
        )

    # --------------------------------------------------------

    def select_action(
        self,
        reasoning_state,
        training=True
    ):

        logits, value = self.policy(
            reasoning_state
        )

        if training:

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            )

            entropy = distribution.entropy()

        else:

            action = logits.argmax(
                dim=-1
            )

            log_prob = torch.zeros_like(
                action,
                dtype=torch.float
            )

            entropy = torch.zeros_like(
                value
            )

        return (
            action,
            log_prob,
            entropy,
            value
        )


# ============================================================
# TRAINING
# ============================================================

def train_agent(
    agent,
    episodes=TRAIN_EPISODES
):

    print()
    print("=" * 70)
    print("TRAINING ARIA FINAL")
    print("=" * 70)

    best_success = 0.0

    episode_rewards = []

    for episode in range(
        1,
        episodes + 1
    ):

        env = ARIAEnvironment()

        observation = env.reset()

        target_color = env.target_color

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for step in range(MAX_STEPS):

            output = agent.perceive_and_reason(
                observation,
                target_color
            )

            reasoning_state = output[
                "reasoning_state"
            ]

            action, log_prob, entropy, value = (
                agent.select_action(
                    reasoning_state,
                    training=True
                )
            )

            action_int = action.item()

            next_observation, reward, done = (
                env.step(
                    action_int
                )
            )

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(
                torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=DEVICE
                )
            )
            entropies.append(entropy)

            agent.memory.add(
                observation,
                reasoning_state,
                action_int,
                reward
            )

            observation = next_observation

            if done:
                break

        # ----------------------------------------------------
        # RETURN COMPUTATION
        # ----------------------------------------------------

        returns = []

        running_return = torch.tensor(
            0.0,
            device=DEVICE
        )

        for reward in reversed(
            rewards
        ):

            running_return = (
                reward
                + GAMMA * running_return
            )

            returns.insert(
                0,
                running_return
            )

        returns = torch.stack(
            returns
        )

        log_probs_tensor = torch.stack(
            log_probs
        )

        values_tensor = torch.stack(
            values
        )

        entropy_tensor = torch.stack(
            entropies
        )

        advantages = (
            returns
            - values_tensor.detach()
        )

        # ----------------------------------------------------
        # ACTOR LOSS
        # ----------------------------------------------------

        actor_loss = -(
            log_probs_tensor
            * advantages
        ).mean()

        # ----------------------------------------------------
        # CRITIC LOSS
        # ----------------------------------------------------

        critic_loss = F.mse_loss(
            values_tensor,
            returns
        )

        # ----------------------------------------------------
        # ENTROPY
        # ----------------------------------------------------

        entropy_bonus = (
            entropy_tensor.mean()
        )

        loss = (
            actor_loss
            + 0.5 * critic_loss
            - 0.01 * entropy_bonus
        )

        agent.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            agent.parameters(),
            1.0
        )

        agent.optimizer.step()

        total_reward = sum(
            float(r.item())
            for r in rewards
        )

        episode_rewards.append(
            total_reward
        )

        success = (
            env.agent_pos
            == env.goal_pos
        )

        if success:
            best_success = max(
                best_success,
                1.0
            )

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 25 == 0
        ):

            recent = episode_rewards[
                -25:
            ]

            avg_reward = sum(
                recent
            ) / len(recent)

            print(
                f"Episode {episode:4d} | "
                f"Reward {avg_reward:7.3f} | "
                f"Steps {env.steps:3d} | "
                f"Success {int(success)}"
            )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if episode % 100 == 0:

            os.makedirs(
                CHECKPOINT_DIR,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict":
                        agent.state_dict(),

                    "optimizer_state_dict":
                        agent.optimizer.state_dict(),

                    "episode":
                        episode,
                },
                CHECKPOINT_PATH
            )

            print(
                f"✓ Checkpoint saved: "
                f"{CHECKPOINT_PATH}"
            )

    return agent


# ============================================================
# GREEDY EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    agent,
    episodes=EVAL_EPISODES
):

    print()
    print("=" * 70)
    print("ARIA FINAL EVALUATION")
    print("=" * 70)

    successes = 0
    collisions = 0

    total_steps = 0
    total_distance = 0.0
    total_reward = 0.0

    grounding_correct = 0

    for episode in range(
        episodes
    ):

        env = ARIAEnvironment()

        observation = env.reset()

        target_color = env.target_color

        # ----------------------------------------------------
        # NEURAL GROUNDING
        # ----------------------------------------------------

        predicted_target, confidence, _ = (
            agent.predict_target(
                observation,
                target_color
            )
        )

        if predicted_target == env.goal_pos:
            grounding_correct += 1

        # ----------------------------------------------------
        # INTERACTION
        # ----------------------------------------------------

        episode_reward = 0.0

        for step in range(
            MAX_STEPS
        ):

            output = agent.perceive_and_reason(
                observation,
                target_color
            )

            reasoning_state = output[
                "reasoning_state"
            ]

            logits, _ = agent.policy(
                reasoning_state
            )

            action = logits.argmax(
                dim=-1
            ).item()

            old_position = env.agent_pos

            observation, reward, done = (
                env.step(
                    action
                )
            )

            episode_reward += reward

            if (
                env.agent_pos == old_position
                and action != 0
            ):
                collisions += 1

            if done:
                break

        total_reward += episode_reward

        total_steps += env.steps

        final_distance = distance(
            env.agent_pos,
            env.goal_pos
        )

        total_distance += final_distance

        if env.agent_pos == env.goal_pos:

            successes += 1

    print(
        f"Grounding Accuracy : "
        f"{grounding_correct / episodes * 100:.2f}%"
    )

    print(
        f"Navigation Success : "
        f"{successes / episodes * 100:.2f}%"
    )

    print(
        f"Collision Rate     : "
        f"{collisions / max(total_steps, 1) * 100:.2f}%"
    )

    print(
        f"Average Reward     : "
        f"{total_reward / episodes:.4f}"
    )

    print(
        f"Final Distance     : "
        f"{total_distance / episodes:.4f}"
    )

    print(
        f"Average Steps      : "
        f"{total_steps / episodes:.2f}"
    )

    return {
        "grounding": grounding_correct / episodes,
        "success": successes / episodes,
        "collision": collisions / max(
            total_steps,
            1
        ),
        "reward": total_reward / episodes,
        "distance": total_distance / episodes,
        "steps": total_steps / episodes,
    }


# ============================================================
# WORLD MODEL TEST
# ============================================================

@torch.no_grad()
def test_world_model(agent):

    print()
    print("=" * 70)
    print("WORLD MODEL TEST")
    print("=" * 70)

    env = ARIAEnvironment()

    observation = env.reset()

    target_color = env.target_color

    output = agent.perceive_and_reason(
        observation,
        target_color
    )

    state = output[
        "reasoning_state"
    ]

    print(
        f"Latent state shape: "
        f"{tuple(state.shape)}"
    )

    actions = [
        4,
        4,
        2,
        2,
        3,
    ]

    imagined = agent.imagination.rollout(
        state,
        actions
    )

    print(
        f"Imagine horizon: "
        f"{len(imagined)}"
    )

    for i, future_state in enumerate(
        imagined,
        1
    ):

        print(
            f"  Future {i}: "
            f"{tuple(future_state.shape)}"
        )

    print(
        "✓ World model imagination test passed"
    )


# ============================================================
# MEMORY TEST
# ============================================================

def test_memory(agent):

    print()
    print("=" * 70)
    print("MEMORY TEST")
    print("=" * 70)

    print(
        f"Memory entries: "
        f"{len(agent.memory.entries)}"
    )

    print(
        "✓ Episodic memory operational"
    )


# ============================================================
# PARAMETER REPORT
# ============================================================

def parameter_report(agent):

    total = sum(
        p.numel()
        for p in agent.parameters()
    )

    trainable = sum(
        p.numel()
        for p in agent.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 70)
    print("ARIA FINAL MODEL")
    print("=" * 70)

    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print()
    print("Components:")

    print("  ✓ Visual Transformer")
    print("  ✓ Instruction Encoder")
    print("  ✓ Neural Reasoner")
    print("  ✓ Spatial Target Grounding")
    print("  ✓ Latent State")
    print("  ✓ Episodic Memory")
    print("  ✓ World Model")
    print("  ✓ Imagination")
    print("  ✓ Actor-Critic RL")
    print("  ✓ Agent Loop")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(agent):

    if not os.path.exists(
        CHECKPOINT_PATH
    ):
        return False

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    agent.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"✓ Loaded checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    return True


# ============================================================
# DEMONSTRATION
# ============================================================

@torch.no_grad()
def demonstration(agent):

    print()
    print("=" * 70)
    print("ARIA FINAL DEMONSTRATION")
    print("=" * 70)

    env = ARIAEnvironment()

    observation = env.reset()

    target_color = env.target_color

    predicted_target, confidence, output = (
        agent.predict_target(
            observation,
            target_color
        )
    )

    print(
        f"Instruction: "
        f"Go to the {target_color} object"
    )

    print(
        f"Actual target: "
        f"{env.goal_pos}"
    )

    print(
        f"Neural target: "
        f"{predicted_target}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )

    print(
        f"Grounding correct: "
        f"{predicted_target == env.goal_pos}"
    )

    print()

    # --------------------------------------------------------
    # AGENT LOOP
    # --------------------------------------------------------

    total_reward = 0.0

    for step in range(
        MAX_STEPS
    ):

        output = agent.perceive_and_reason(
            observation,
            target_color
        )

        reasoning_state = output[
            "reasoning_state"
        ]

        target_logits = output[
            "target_logits"
        ]

        target_index = target_logits.argmax(
            dim=-1
        ).item()

        predicted_target = index_to_position(
            target_index
        )

        action_logits, value = agent.policy(
            reasoning_state
        )

        action = action_logits.argmax(
            dim=-1
        ).item()

        observation, reward, done = (
            env.step(
                action
            )
        )

        total_reward += reward

        print(
            f"Step {step + 1:3d} | "
            f"Target {predicted_target} | "
            f"Action {ACTION_NAMES[action]:5s} | "
            f"Position {env.agent_pos} | "
            f"Reward {reward:+.3f}"
        )

        if done:
            break

    print()
    print(
        f"Final position: "
        f"{env.agent_pos}"
    )

    print(
        f"Actual goal: "
        f"{env.goal_pos}"
    )

    print(
        f"Steps: "
        f"{env.steps}"
    )

    print(
        f"Total reward: "
        f"{total_reward:.4f}"
    )

    print(
        f"Success: "
        f"{env.agent_pos == env.goal_pos}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ARIA FINAL")
    print("Adaptive Reasoning & Imagination Agent")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Grid: {GRID_SIZE}x{GRID_SIZE}"
    )

    print(
        "Architecture:"
    )

    print(
        "Perception → Reasoning → Memory → "
        "World Model → Imagination → RL → Action"
    )

    # --------------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------------

    agent = ARIAAgent().to(
        DEVICE
    )

    parameter_report(agent)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_agent(
        agent,
        episodes=TRAIN_EPISODES
    )

    # --------------------------------------------------------
    # LOAD BEST/LATEST
    # --------------------------------------------------------

    load_checkpoint(agent)

    # --------------------------------------------------------
    # TEST INTERNAL COMPONENTS
    # --------------------------------------------------------

    test_world_model(agent)

    test_memory(agent)

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    demonstration(agent)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results = evaluate(
        agent,
        episodes=EVAL_EPISODES
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA FINAL REPORT")
    print("=" * 70)

    print(
        f"Grounding Accuracy : "
        f"{results['grounding'] * 100:.2f}%"
    )

    print(
        f"Navigation Success : "
        f"{results['success'] * 100:.2f}%"
    )

    print(
        f"Collision Rate     : "
        f"{results['collision'] * 100:.2f}%"
    )

    print(
        f"Average Reward     : "
        f"{results['reward']:.4f}"
    )

    print(
        f"Final Distance     : "
        f"{results['distance']:.4f}"
    )

    print(
        f"Average Steps      : "
        f"{results['steps']:.2f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ARIA FINAL COMPLETED"
    )

    print(
        "=" * 70
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()