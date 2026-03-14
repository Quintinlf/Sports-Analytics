"""
Hierarchical Bayesian basketball model (experimental).

This is a streamlined implementation preserving the public interface used
in the legacy monolith while keeping runtime practical.
"""

import pickle
from typing import Dict, List

import numpy as np


class BayesianBasketballHierarchical:
    """
    Experimental hierarchical model with Gibbs-style posterior sampling.

    Expected input for fit_gibbs:
        team_shot_data[team_id] = {
            'M_ik': np.ndarray shape (K,),
            'N_ik': np.ndarray shape (K,),
            'points_per_game': float,
        }
    """

    def __init__(self, L: int = 10, J: int = 10, K: int = 7):
        self.L = L
        self.J = J
        self.K = K

        self.team_ids: List[int] = []
        self.theta_i: Dict[int, float] = {}
        self.z_i: Dict[int, int] = {}
        self.w_i: Dict[int, int] = {}
        self.mu_j = np.zeros((J, K))
        self.eta_l = np.zeros((L, K))

        self.posterior_samples = {'theta': [], 'z': [], 'w': [], 'mu': [], 'eta': []}
        self.epaa_stats: Dict[int, Dict] = {}
        self.cluster_assignments: Dict[int, Dict] = {}

    def fit_gibbs(
        self,
        team_shot_data: Dict,
        n_iterations: int = 5000,
        burn_in: int = 1500,
        thin: int = 1,
        verbose: bool = False,
    ) -> 'BayesianBasketballHierarchical':
        """
        Fit model and collect posterior draws.

        The routine uses simple conjugate draws for cluster-level shooting
        profiles and Normal draws for EPAA to keep runtime tractable.
        """
        self.team_ids = list(team_shot_data.keys())
        if not self.team_ids:
            return self

        ppg = np.array([team_shot_data[t]['points_per_game'] for t in self.team_ids], dtype=float)
        league_avg = float(np.mean(ppg))

        # Initialize random cluster assignments
        for tid in self.team_ids:
            self.z_i[tid] = int(np.random.randint(0, self.J))
            self.w_i[tid] = int(np.random.randint(0, self.L))
            self.theta_i[tid] = float(team_shot_data[tid]['points_per_game'] - league_avg)

        self.mu_j = np.random.dirichlet(np.ones(self.K), size=self.J)
        self.eta_l = np.random.beta(4.5, 5.5, size=(self.L, self.K))

        if verbose:
            print(f'Gibbs sampling: teams={len(self.team_ids)}, iter={n_iterations}')

        for it in range(n_iterations):
            # Update team-level latent assignments
            for tid in self.team_ids:
                attempts = np.asarray(team_shot_data[tid]['N_ik'], dtype=float)
                makes = np.asarray(team_shot_data[tid]['M_ik'], dtype=float)

                # Shot-selection cluster
                z_scores = np.array([
                    np.sum(attempts * np.log(self.mu_j[j] + 1e-10))
                    for j in range(self.J)
                ])
                z_probs = np.exp(z_scores - np.max(z_scores))
                z_probs /= z_probs.sum()
                self.z_i[tid] = int(np.random.choice(self.J, p=z_probs))

                # Accuracy cluster
                w_scores = np.array([
                    np.sum(
                        makes * np.log(self.eta_l[l] + 1e-10)
                        + (attempts - makes) * np.log(1.0 - self.eta_l[l] + 1e-10)
                    )
                    for l in range(self.L)
                ])
                w_probs = np.exp(w_scores - np.max(w_scores))
                w_probs /= w_probs.sum()
                self.w_i[tid] = int(np.random.choice(self.L, p=w_probs))

                # EPAA draw
                observed_epaa = float(team_shot_data[tid]['points_per_game'] - league_avg)
                self.theta_i[tid] = float(np.random.normal(observed_epaa, 2.0))

            # Update cluster parameters
            for j in range(self.J):
                teams = [tid for tid in self.team_ids if self.z_i[tid] == j]
                if teams:
                    total_attempts = np.sum([team_shot_data[t]['N_ik'] for t in teams], axis=0)
                    self.mu_j[j] = np.random.dirichlet(1.0 + np.asarray(total_attempts, dtype=float))
                else:
                    self.mu_j[j] = np.random.dirichlet(np.ones(self.K))

            for l in range(self.L):
                teams = [tid for tid in self.team_ids if self.w_i[tid] == l]
                if teams:
                    total_makes = np.sum([team_shot_data[t]['M_ik'] for t in teams], axis=0)
                    total_attempts = np.sum([team_shot_data[t]['N_ik'] for t in teams], axis=0)
                else:
                    total_makes = np.zeros(self.K)
                    total_attempts = np.zeros(self.K)
                a = 4.5 + total_makes
                b = 5.5 + np.maximum(0, total_attempts - total_makes)
                self.eta_l[l] = np.random.beta(a, b)

            if it >= burn_in and ((it - burn_in) % max(thin, 1) == 0):
                self.posterior_samples['theta'].append(self.theta_i.copy())
                self.posterior_samples['z'].append(self.z_i.copy())
                self.posterior_samples['w'].append(self.w_i.copy())
                self.posterior_samples['mu'].append(self.mu_j.copy())
                self.posterior_samples['eta'].append(self.eta_l.copy())

        self._compute_posterior_statistics()
        return self

    def _compute_posterior_statistics(self) -> None:
        n = len(self.posterior_samples['theta'])
        if n == 0 or not self.team_ids:
            return

        self.epaa_stats = {}
        self.cluster_assignments = {}

        for tid in self.team_ids:
            theta_samples = np.array([draw[tid] for draw in self.posterior_samples['theta']], dtype=float)
            z_samples = np.array([draw[tid] for draw in self.posterior_samples['z']], dtype=int)
            w_samples = np.array([draw[tid] for draw in self.posterior_samples['w']], dtype=int)

            self.epaa_stats[tid] = {
                'mean': float(np.mean(theta_samples)),
                'std': float(np.std(theta_samples)),
                'median': float(np.median(theta_samples)),
                'q025': float(np.percentile(theta_samples, 2.5)),
                'q975': float(np.percentile(theta_samples, 97.5)),
            }

            z_counts = np.bincount(z_samples, minlength=self.J)
            w_counts = np.bincount(w_samples, minlength=self.L)

            self.cluster_assignments[tid] = {
                'shot_selection': {
                    'most_likely': int(np.argmax(z_counts)),
                    'probabilities': z_counts / max(n, 1),
                },
                'accuracy': {
                    'most_likely': int(np.argmax(w_counts)),
                    'probabilities': w_counts / max(n, 1),
                },
            }

    def predict_team_performance(self, team_id):
        if team_id not in self.epaa_stats:
            raise ValueError(f'Team {team_id} not in fitted model')

        st = self.epaa_stats[team_id]
        cl = self.cluster_assignments[team_id]
        z = cl['shot_selection']['most_likely']
        w = cl['accuracy']['most_likely']
        expected_fg_pct = float(np.sum(self.mu_j[z] * self.eta_l[w]))

        return {
            'epaa_mean': st['mean'],
            'epaa_std': st['std'],
            'epaa_ci': (st['q025'], st['q975']),
            'shot_cluster': z,
            'accuracy_cluster': w,
            'expected_fg_pct': expected_fg_pct,
            'cluster_probabilities': {
                'shot_selection': cl['shot_selection']['probabilities'],
                'accuracy': cl['accuracy']['probabilities'],
            },
        }

    def get_epaa_rankings(self):
        rankings = [(tid, st['mean'], st['std']) for tid, st in self.epaa_stats.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_cluster_profiles(self):
        if self.posterior_samples['mu']:
            mu = np.mean(np.asarray(self.posterior_samples['mu']), axis=0)
            eta = np.mean(np.asarray(self.posterior_samples['eta']), axis=0)
        else:
            mu = self.mu_j
            eta = self.eta_l
        return {'shot_selection': mu, 'accuracy': eta}

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'BayesianBasketballHierarchical':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, BayesianBasketballHierarchical):
            raise TypeError('Loaded object is not BayesianBasketballHierarchical')
        return obj


def calculate_epaa(team_shot_data: Dict, league_avg_ppg: float = None) -> Dict:
    if not team_shot_data:
        return {}
    if league_avg_ppg is None:
        league_avg_ppg = float(np.mean([d['points_per_game'] for d in team_shot_data.values()]))

    out = {}
    for tid, data in team_shot_data.items():
        epaa = float(data['points_per_game'] - league_avg_ppg)
        out[tid] = {'epaa': epaa, 'ppg': float(data['points_per_game']), 'league_avg': league_avg_ppg}
    return out


def compare_team_matchup(model: BayesianBasketballHierarchical, home_team_id, away_team_id) -> Dict:
    home = model.predict_team_performance(home_team_id)
    away = model.predict_team_performance(away_team_id)

    epaa_diff = float(home['epaa_mean'] - away['epaa_mean'])
    home_advantage = 3.0
    spread = epaa_diff + home_advantage
    spread_std = float(np.sqrt(home['epaa_std'] ** 2 + away['epaa_std'] ** 2))

    return {
        'home_epaa': home,
        'away_epaa': away,
        'epaa_diff': epaa_diff,
        'predicted_spread': spread,
        'spread_std': spread_std,
        'spread_ci': (spread - 1.96 * spread_std, spread + 1.96 * spread_std),
        'home_advantage': home_advantage,
    }
