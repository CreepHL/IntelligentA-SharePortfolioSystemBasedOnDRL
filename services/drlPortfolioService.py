import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


class PortfolioStateEncoder:
    """Encode agent signals into RL state representation"""

    def encode_state(self) -> np.ndarray:
        """Extract features from agent signals"""
        # Get agent signals (similar to portfolio_manager.py)
        messages = state.get("messages", [])
        portfolio = state.get("data", {}).get("portfolio", {})

        # Extract signal features
        features = []

        # Portfolio state
        features.extend([
            portfolio.get("cash", 0) / 1000000,  # Normalized cash
            portfolio.get("stock", 0) / 1000,  # Normalized shares
        ])

        # Agent signals (convert to numerical)
        for agent_name in ["technical_analyst_agent", "fundamentals_agent",
                           "sentiment_agent", "valuation_agent", "macro_analyst_agent"]:
            signal = self._extract_signal(messages, agent_name)
            features.extend(signal)

        return np.array(features, dtype=np.float32)

    def _extract_signal(self, messages, agent_name):
        """Extract numerical signal from agent message"""
        # Implementation to parse agent signals
        # Returns [bullish_score, bearish_score, confidence]
        pass


class RLPolicyNetwork(nn.Module):
    """Deep Q-Network for portfolio decisions"""

    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class DRLPortfolio:
    """DRL-based portfolio management agent"""

    def __init__(self, model_path: str = None):
        self.state_encoder = PortfolioStateEncoder()
        self.policy_net = RLPolicyNetwork(state_dim=20)
        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path))

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Make portfolio decision using trained policy"""
        # Encode current state
        state_tensor = torch.FloatTensor(self.state_encoder.encode_state(state))

        # Get action from policy
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()

            # Map action to decision
        actions = ["hold", "buy", "sell"]
        action = actions[action_idx]

        # Calculate quantity (simplified)
        portfolio = state["data"]["portfolio"]
        if action == "buy" and portfolio["cash"] > 0:
            quantity = min(100, int(portfolio["cash"] / 100))  # Buy with 10% of cash
        elif action == "sell" and portfolio["stock"] > 0:
            quantity = min(100, portfolio["stock"] // 10)  # Sell 10% of holdings
        else:
            quantity = 0

        return {
            "messages": state["messages"] + [HumanMessage(
                content=json.dumps({
                    "action": action,
                    "quantity": quantity,
                    "confidence": 0.8,
                    "agent_signals": [],  # Can include original signals for transparency
                    "reasoning": f"DRL policy decision: {action}"
                }),
                name="rl_portfolio_agent"
            )],
            "data": state["data"],
            "metadata": state["metadata"]
        }