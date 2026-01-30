import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict


# -----------------------------
# 1) Utility / Reward (profit high better, safety high safer)
# -----------------------------

@dataclass
class UserPref:
    risk_pref: float  # 0=aggressive (profit), 1=conservative (safety)


def utility(
    sel_mask: torch.Tensor,   # [B, N] 0/1
    profit: torch.Tensor,     # [N]
    safety: torch.Tensor,     # [N] higher=safer
    pref: UserPref,
    size_penalty: float = 0.05,
) -> torch.Tensor:
    """
    U(S) = (1-rho)*profit_sum + rho*safety_sum - size_penalty*|S|
    """
    rho = float(pref.risk_pref)
    profit_sum = (sel_mask * profit).sum(dim=-1)
    safety_sum = (sel_mask * safety).sum(dim=-1)
    size = sel_mask.sum(dim=-1)
    return (1 - rho) * profit_sum + rho * safety_sum - size_penalty * size


def log_reward_from_utility(U: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    # log R = beta * U, with R = exp(beta U)
    return beta * U


# -----------------------------
# 2) GFlowNet for subset/path generation (Trajectory Balance baseline)
# -----------------------------

class SubsetGFlowNetTB(nn.Module):
    """
    Path: start -> add stock -> ... -> STOP
    Actions: AddStock(i) for i not selected, or STOP.
    """
    def __init__(self, n_assets: int, kmax: int = 8, hidden: int = 128):
        super().__init__()
        self.N = n_assets
        self.Kmax = kmax
        state_dim = n_assets + 2  # sel_mask + k_sel + step
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),  # +1 for risk_pref as context
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_assets + 1)  # N add + STOP
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def _mask_actions(self, sel_mask: torch.Tensor, k_sel: torch.Tensor) -> torch.Tensor:
        B, N = sel_mask.shape
        mask = torch.ones(B, N + 1, device=sel_mask.device)
        mask[:, :N] = 1.0 - sel_mask                     # cannot re-select
        full = (k_sel.squeeze(-1) >= self.Kmax).float().unsqueeze(-1)
        mask[:, :N] = mask[:, :N] * (1.0 - full)         # full => must stop
        empty = (k_sel.squeeze(-1) <= 0).float()
        mask[:, N] = 1.0 - empty                         # empty => cannot stop
        return mask

    @staticmethod
    def _sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        logits = logits.masked_fill(mask <= 0.0, -1e9)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def _logits(self, sel_mask, k_sel, step, risk_pref):
        x = torch.cat([sel_mask, k_sel, step, risk_pref], dim=-1)
        h = self.net(x)
        return self.head(h)

    @torch.no_grad()
    def sample_paths(self, risk_pref: float, n_samples: int, temperature: float = 0.8, device="cpu"):
        B, N = n_samples, self.N
        sel_mask = torch.zeros(B, N, device=device)
        actions = torch.full((B, self.Kmax + 1), -1, dtype=torch.long, device=device)

        k_sel = torch.zeros(B, 1, device=device)
        step = torch.zeros(B, 1, device=device)
        rp = torch.full((B, 1), float(risk_pref), device=device)

        for t in range(self.Kmax + 1):
            logits = self._logits(sel_mask, k_sel, step, rp)
            mask = self._mask_actions(sel_mask, k_sel)
            a = self._sample(logits, mask, temperature)
            actions[:, t] = a

            stop = (a == N)
            idx = torch.arange(B, device=device)
            sel_mask[idx[~stop], a[~stop]] = 1.0

            k_sel = sel_mask.sum(dim=-1, keepdim=True)
            step = step + 1.0
            if stop.all():
                break

        return sel_mask, actions

    def logprob_paths(self, risk_pref: float, actions: torch.Tensor) -> torch.Tensor:
        device = actions.device
        B, L = actions.shape
        N = self.N

        sel_mask = torch.zeros(B, N, device=device)
        k_sel = torch.zeros(B, 1, device=device)
        step = torch.zeros(B, 1, device=device)
        rp = torch.full((B, 1), float(risk_pref), device=device)

        logpf = torch.zeros(B, device=device)

        for t in range(L):
            a = actions[:, t]
            alive = (a >= 0)
            if not alive.any():
                break

            logits = self._logits(sel_mask, k_sel, step, rp)
            mask = self._mask_actions(sel_mask, k_sel)
            logits = logits.masked_fill(mask <= 0.0, -1e9)
            logp = F.log_softmax(logits, dim=-1)

            idx = torch.arange(B, device=device)
            logpf[alive] += logp[idx[alive], a[alive]]

            stop = (a == N) & alive
            add = (~stop) & alive
            sel_mask[idx[add], a[add]] = 1.0

            k_sel = sel_mask.sum(dim=-1, keepdim=True)
            step = step + 1.0

        return logpf


def train_gflownet(
    profit_scores: List[float],
    safety_scores: List[float],
    risk_pref: float,
    kmax: int = 6,
    beta: float = 3.0,
    size_penalty: float = 0.05,
    steps: int = 2500,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: str = "cpu",
):
    profit = torch.tensor(profit_scores, dtype=torch.float32, device=device)
    safety = torch.tensor(safety_scores, dtype=torch.float32, device=device)
    pref = UserPref(risk_pref=risk_pref)

    model = SubsetGFlowNetTB(n_assets=len(profit_scores), kmax=kmax, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for s in range(1, steps + 1):
        sel_mask, actions = model.sample_paths(risk_pref, n_samples=batch_size, temperature=1.0, device=device)

        U = utility(sel_mask, profit, safety, pref, size_penalty=size_penalty)
        logR = log_reward_from_utility(U, beta=beta)
        logpf = model.logprob_paths(risk_pref, actions)

        tb = model.logZ + logpf - logR
        loss = (tb ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 300 == 0:
            with torch.no_grad():
                sm, ac = model.sample_paths(risk_pref, n_samples=200, temperature=0.8, device=device)
                U_s = utility(sm, profit, safety, pref, size_penalty=size_penalty)
                print(f"[step {s:4d}] loss={loss.item():.4f} logZ={model.logZ.item():.3f} U_best={U_s.max().item():.3f} U_mean={U_s.mean().item():.3f}")

    return model


def decode_path(actions_row: torch.Tensor, stop_id: int) -> List[int]:
    path = []
    for a in actions_row.tolist():
        if a < 0 or a == stop_id:
            break
        path.append(a)
    return path


def generate_ranked_paths(
    model: SubsetGFlowNetTB,
    profit_scores: List[float],
    safety_scores: List[float],
    risk_pref: float,
    n_generate: int = 500,
    top_k: int = 30,
    kmax: int = 6,
    beta: float = 3.0,
    size_penalty: float = 0.4,
    sort_by: str = "profit",  # "profit" or "utility"
    device: str = "cpu",
) -> List[Dict]:
    profit = torch.tensor(profit_scores, dtype=torch.float32, device=device)
    safety = torch.tensor(safety_scores, dtype=torch.float32, device=device)
    pref = UserPref(risk_pref=risk_pref)

    sel_mask, actions = model.sample_paths(risk_pref, n_samples=n_generate, temperature=0.8, device=device)

    profit_sum = (sel_mask * profit).sum(dim=-1).cpu().numpy()
    safety_sum = (sel_mask * safety).sum(dim=-1).cpu().numpy()
    U = utility(sel_mask, profit, safety, pref, size_penalty=size_penalty).cpu().numpy()

    items = []
    for i in range(n_generate):
        chosen = np.where(sel_mask[i].cpu().numpy() > 0.5)[0].tolist()
        path = decode_path(actions[i], stop_id=model.N)
        items.append({
            "path": path,                 # 选股顺序（路径）
            "portfolio": chosen,          # 最终组合（集合）
            "profit_sum": float(profit_sum[i]),
            "safety_sum": float(safety_sum[i]),
            "utility": float(U[i]),
        })

    key = "profit_sum" if sort_by == "profit" else "utility"
    items.sort(key=lambda d: d[key], reverse=True)

    # 去重（同一个组合不重复输出）
    seen = set()
    uniq = []
    for it in items:
        sig = tuple(sorted(it["portfolio"]))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(it)
        if len(uniq) >= top_k:
            break
    return uniq


def run_gflownets(profit_scores, safety_scores):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # user risk preference: 0 aggressive, 1 conservative
    risk_pref = 0.7   # 偏保守 => 更看安全分数

    model = train_gflownet(
        profit_scores, safety_scores,
        risk_pref=risk_pref,
        kmax=6, beta=3.0,
        size_penalty=0.4,
        steps=2000, batch_size=64,
        lr=2e-4, device=device
    )

    # 你要求“按盈利降序输出多条路径”
    top = generate_ranked_paths(
        model, profit_scores, safety_scores, risk_pref,
        n_generate=800, top_k=8,
        sort_by="profit",
        device=device
    )

    print("\nTop paths by PROFIT (descending):")
    for i, d in enumerate(top, 1):
        print(i, d)
