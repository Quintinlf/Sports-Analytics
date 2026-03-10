"""
Hierarchical Bayesian MCMC Model for NBA Basketball Analytics

This module implements a Gibbs sampler for hierarchical Bayesian modeling of NBA team performance.
The model captures:
- Shot selection patterns (z_i): Which shot types teams prefer
- Shooting accuracy (w_i): How accurate teams are from different locations
- EPAA (Expected Points Above Average): Overall team efficiency metric

The hierarchical structure allows for:
- Team-specific parameters
- Population-level priors
- Uncertainty quantification via posterior sampling
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pickle
from typing import Dict, Tuple, Optional, List


class BayesianBasketballHierarchical:
    """
    Hierarchical Bayesian model for NBA team performance using Gibbs sampling.
    
    This model uses MCMC (Markov Chain Monte Carlo) to estimate:
    - Team offensive efficiency (EPAA - Expected Points Above Average)
    - Shot selection clusters (which shot types teams prefer)
    - Accuracy clusters (how accurate teams are from different court regions)
    
    Parameters
    ----------
    L : int, default=10
        Number of accuracy clusters (how many distinct accuracy profiles)
    J : int, default=10
        Number of shot selection clusters (how many distinct shot selection patterns)
    K : int, default=7
        Number of court regions (e.g., paint, mid-range, 3PT corner, etc.)
    
    Attributes
    ----------
    team_ids : list
        List of team IDs in the model
    theta_i : dict
        Team-specific offensive efficiency parameters (EPAA)
    z_i : dict
        Shot selection cluster assignments for each team
    w_i : dict
        Accuracy cluster assignments for each team
    mu_j : np.ndarray
        Shot selection profiles (J x K): proportion of shots from each region
    eta_l : np.ndarray
        Accuracy profiles (L x K): field goal percentage for each region
    """
    
    def __init__(self, L=10, J=10, K=7):
        """
        Initialize the hierarchical Bayesian model.
        
        Parameters
        ----------
        L : int
            Number of accuracy clusters
        J : int
            Number of shot selection clusters
        K : int
            Number of court regions
        """
        self.L = L  # Number of accuracy clusters
        self.J = J  # Number of shot selection clusters
        self.K = K  # Number of court regions
        
        # Model parameters (will be set during fitting)
        self.team_ids = None
        self.theta_i = {}  # Team offensive efficiency (EPAA)
        self.z_i = {}      # Shot selection cluster assignment
        self.w_i = {}      # Accuracy cluster assignment
        self.mu_j = None   # Shot selection profiles (J x K)
        self.eta_l = None  # Accuracy profiles (L x K)
        
        # Hyperparameters
        self.alpha_z = np.ones(J)  # Dirichlet prior for shot selection
        self.alpha_w = np.ones(L)  # Dirichlet prior for accuracy
        
        # Posterior samples for uncertainty quantification
        self.posterior_samples = {
            'theta': [],
            'z': [],
            'w': [],
            'mu': [],
            'eta': []
        }
        
    def fit_gibbs(self, team_shot_data, n_iterations=5000, burn_in=1500, thin=1):
        """
        Fit the model using Gibbs sampling.
        
        This is the core MCMC algorithm that alternates between sampling:
        1. Team assignments (z_i, w_i)
        2. Cluster parameters (mu_j, eta_l)
        3. Team efficiency (theta_i)
        
        Parameters
        ----------
        team_shot_data : dict
            Dictionary mapping team_id to shot data:
            {
                team_id: {
                    'M_ik': np.ndarray of shape (K,)  # Made shots per region
                    'N_ik': np.ndarray of shape (K,)  # Attempted shots per region
                    'points_per_game': float          # Average points scored
                }
            }
        n_iterations : int, default=5000
            Total number of Gibbs sampling iterations
        burn_in : int, default=1500
            Number of initial iterations to discard
        thin : int, default=1
            Keep every thin-th sample (for reducing autocorrelation)
            
        Returns
        -------
        self : BayesianBasketballHierarchical
            Fitted model with posterior samples
        """
        self.team_ids = list(team_shot_data.keys())
        n_teams = len(self.team_ids)
        
        print(f"🔬 Starting Gibbs Sampling with {n_iterations} iterations...")
        print(f"   Teams: {n_teams}")
        print(f"   Burn-in: {burn_in}, Thinning: {thin}")
        print(f"   Clusters: {self.J} shot selection, {self.L} accuracy")
        
        # Initialize parameters randomly
        self._initialize_parameters(team_shot_data)
        
        # Store data for sampling
        self.data = team_shot_data
        
        # Gibbs sampling loop
        for iteration in range(n_iterations):
            if iteration % 500 == 0:
                print(f"   Iteration {iteration}/{n_iterations}...")
            
            # Step 1: Sample cluster assignments for each team
            for team_id in self.team_ids:
                self._sample_z_i(team_id)
                self._sample_w_i(team_id)
            
            # Step 2: Sample cluster parameters
            self._sample_mu_j()
            self._sample_eta_l()
            
            # Step 3: Sample team efficiency parameters
            self._sample_theta_i()
            
            # Store samples after burn-in (with thinning)
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                self._store_sample()
        
        print(f"✅ Gibbs sampling complete!")
        print(f"   Collected {len(self.posterior_samples['theta'])} posterior samples")
        
        # Compute posterior means and credible intervals
        self._compute_posterior_statistics()
        
        return self
    
    def _initialize_parameters(self, team_shot_data):
        """Initialize all parameters randomly."""
        # Initialize cluster assignments randomly
        for team_id in self.team_ids:
            self.z_i[team_id] = np.random.randint(0, self.J)
            self.w_i[team_id] = np.random.randint(0, self.L)
        
        # Initialize shot selection profiles (Dirichlet priors)
        self.mu_j = np.random.dirichlet([1.0] * self.K, size=self.J)
        
        # Initialize accuracy profiles (Beta priors, centered around 0.45)
        self.eta_l = np.random.beta(4.5, 5.5, size=(self.L, self.K))
        
        # Initialize team efficiency based on actual points per game
        league_avg_ppg = np.mean([data['points_per_game'] 
                                   for data in team_shot_data.values()])
        
        for team_id in self.team_ids:
            ppg = team_shot_data[team_id]['points_per_game']
            self.theta_i[team_id] = ppg - league_avg_ppg  # EPAA
    
    def _sample_z_i(self, team_id):
        """
        Sample shot selection cluster assignment for team i.
        
        Uses the observed shot distribution to compute posterior probabilities
        for each cluster, then samples from the categorical distribution.
        """
        data = self.data[team_id]
        N_ik = data['N_ik']  # Shot attempts per region
        
        # Compute log probabilities for each cluster
        log_probs = np.zeros(self.J)
        
        for j in range(self.J):
            # Multinomial likelihood: N_ik | mu_j[k]
            # Log probability of observed shot distribution given cluster j
            log_probs[j] = np.sum(N_ik * np.log(self.mu_j[j] + 1e-10))
            
            # Add prior (Dirichlet)
            log_probs[j] += np.log(self.alpha_z[j] + 1e-10)
        
        # Normalize to get probabilities
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Sample new cluster assignment
        self.z_i[team_id] = np.random.choice(self.J, p=probs)
    
    def _sample_w_i(self, team_id):
        """
        Sample accuracy cluster assignment for team i.
        
        Uses the observed makes/attempts to compute posterior probabilities
        for each accuracy cluster.
        """
        data = self.data[team_id]
        M_ik = data['M_ik']  # Made shots per region
        N_ik = data['N_ik']  # Attempted shots per region
        
        # Compute log probabilities for each accuracy cluster
        log_probs = np.zeros(self.L)
        
        for l in range(self.L):
            # Binomial likelihood: M_ik | N_ik, eta_l[k]
            for k in range(self.K):
                if N_ik[k] > 0:
                    log_probs[l] += stats.binom.logpmf(
                        int(M_ik[k]), 
                        int(N_ik[k]), 
                        self.eta_l[l, k]
                    )
            
            # Add prior (Dirichlet)
            log_probs[l] += np.log(self.alpha_w[l] + 1e-10)
        
        # Normalize to get probabilities
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Sample new cluster assignment
        self.w_i[team_id] = np.random.choice(self.L, p=probs)
    
    def _sample_mu_j(self):
        """
        Sample shot selection profiles for all clusters.
        
        For each cluster j, aggregate all teams assigned to that cluster
        and sample from the Dirichlet posterior.
        """
        for j in range(self.J):
            # Find teams assigned to cluster j
            teams_in_cluster = [tid for tid in self.team_ids if self.z_i[tid] == j]
            
            if len(teams_in_cluster) == 0:
                # No teams in this cluster, sample from prior
                self.mu_j[j] = np.random.dirichlet(self.alpha_z)
            else:
                # Aggregate shot counts from all teams in cluster
                total_shots = np.zeros(self.K)
                for team_id in teams_in_cluster:
                    total_shots += self.data[team_id]['N_ik']
                
                # Sample from Dirichlet posterior
                posterior_alpha = self.alpha_z + total_shots
                self.mu_j[j] = np.random.dirichlet(posterior_alpha)
    
    def _sample_eta_l(self):
        """
        Sample accuracy profiles for all clusters.
        
        For each cluster l and region k, aggregate makes/attempts and
        sample from the Beta posterior.
        """
        for l in range(self.L):
            # Find teams assigned to cluster l
            teams_in_cluster = [tid for tid in self.team_ids if self.w_i[tid] == l]
            
            for k in range(self.K):
                if len(teams_in_cluster) == 0:
                    # No teams in cluster, sample from prior Beta(4.5, 5.5)
                    self.eta_l[l, k] = np.random.beta(4.5, 5.5)
                else:
                    # Aggregate makes and attempts
                    total_makes = sum(self.data[tid]['M_ik'][k] 
                                     for tid in teams_in_cluster)
                    total_attempts = sum(self.data[tid]['N_ik'][k] 
                                        for tid in teams_in_cluster)
                    
                    # Sample from Beta posterior
                    # Beta(a + makes, b + (attempts - makes))
                    alpha_post = 4.5 + total_makes
                    beta_post = 5.5 + (total_attempts - total_makes)
                    self.eta_l[l, k] = np.random.beta(alpha_post, beta_post)
    
    def _sample_theta_i(self):
        """
        Sample team efficiency parameters (EPAA).
        
        This uses a Normal likelihood based on actual points per game
        and the expected points from the team's shot profile.
        """
        league_avg_ppg = np.mean([self.data[tid]['points_per_game'] 
                                   for tid in self.team_ids])
        
        for team_id in self.team_ids:
            observed_ppg = self.data[team_id]['points_per_game']
            
            # Expected points from shot profile
            z = self.z_i[team_id]
            w = self.w_i[team_id]
            
            # Calculate expected points: sum over regions of
            # (shot_proportion * accuracy * points_per_shot)
            expected_points = 0.0
            for k in range(self.K):
                shot_proportion = self.mu_j[z, k]
                accuracy = self.eta_l[w, k]
                # Assume regions 0-3 are 2PT (paint, mid), 4-6 are 3PT
                points_value = 3.0 if k >= 4 else 2.0
                expected_points += shot_proportion * accuracy * points_value
            
            # Scale to per-game basis (assume ~80 FGA per game)
            expected_points *= 80
            
            # Sample theta from Normal distribution
            # theta represents deviation from league average
            observed_epaa = observed_ppg - league_avg_ppg
            expected_epaa = expected_points - league_avg_ppg
            
            # Normal posterior with observed data
            # Prior: N(0, 5^2), Likelihood variance: 3^2
            prior_mean = 0.0
            prior_var = 25.0
            likelihood_var = 9.0
            
            # Posterior is also Normal with updated parameters
            post_var = 1.0 / (1.0/prior_var + 1.0/likelihood_var)
            post_mean = post_var * (prior_mean/prior_var + observed_epaa/likelihood_var)
            
            self.theta_i[team_id] = np.random.normal(post_mean, np.sqrt(post_var))
    
    def _store_sample(self):
        """Store current parameter values as a posterior sample."""
        self.posterior_samples['theta'].append(self.theta_i.copy())
        self.posterior_samples['z'].append(self.z_i.copy())
        self.posterior_samples['w'].append(self.w_i.copy())
        self.posterior_samples['mu'].append(self.mu_j.copy())
        self.posterior_samples['eta'].append(self.eta_l.copy())
    
    def _compute_posterior_statistics(self):
        """Compute summary statistics from posterior samples."""
        n_samples = len(self.posterior_samples['theta'])
        
        # Compute EPAA statistics for each team
        self.epaa_stats = {}
        for team_id in self.team_ids:
            theta_samples = [sample[team_id] 
                            for sample in self.posterior_samples['theta']]
            
            self.epaa_stats[team_id] = {
                'mean': np.mean(theta_samples),
                'std': np.std(theta_samples),
                'median': np.median(theta_samples),
                'q025': np.percentile(theta_samples, 2.5),
                'q975': np.percentile(theta_samples, 97.5)
            }
        
        # Compute most likely cluster assignments
        self.cluster_assignments = {}
        for team_id in self.team_ids:
            z_samples = [sample[team_id] 
                        for sample in self.posterior_samples['z']]
            w_samples = [sample[team_id] 
                        for sample in self.posterior_samples['w']]
            
            # Most frequent cluster assignment
            z_counts = np.bincount(z_samples, minlength=self.J)
            w_counts = np.bincount(w_samples, minlength=self.L)
            
            self.cluster_assignments[team_id] = {
                'shot_selection': {
                    'most_likely': np.argmax(z_counts),
                    'probabilities': z_counts / n_samples
                },
                'accuracy': {
                    'most_likely': np.argmax(w_counts),
                    'probabilities': w_counts / n_samples
                }
            }
    
    def predict_team_performance(self, team_id):
        """
        Predict team performance metrics.
        
        Parameters
        ----------
        team_id : int or str
            Team identifier
            
        Returns
        -------
        dict
            Dictionary with:
            - 'epaa_mean': Expected points above average (mean)
            - 'epaa_std': Standard deviation of EPAA
            - 'epaa_ci': 95% credible interval
            - 'shot_cluster': Most likely shot selection cluster
            - 'accuracy_cluster': Most likely accuracy cluster
            - 'expected_fg_pct': Expected field goal percentage
        """
        if team_id not in self.team_ids:
            raise ValueError(f"Team {team_id} not in fitted model")
        
        stats = self.epaa_stats[team_id]
        clusters = self.cluster_assignments[team_id]
        
        # Calculate expected FG%
        z = clusters['shot_selection']['most_likely']
        w = clusters['accuracy']['most_likely']
        
        expected_fg_pct = np.sum(self.mu_j[z] * self.eta_l[w])
        
        return {
            'epaa_mean': stats['mean'],
            'epaa_std': stats['std'],
            'epaa_ci': (stats['q025'], stats['q975']),
            'shot_cluster': z,
            'accuracy_cluster': w,
            'expected_fg_pct': expected_fg_pct,
            'cluster_probabilities': {
                'shot_selection': clusters['shot_selection']['probabilities'],
                'accuracy': clusters['accuracy']['probabilities']
            }
        }
    
    def get_epaa_rankings(self):
        """
        Get teams ranked by EPAA (Expected Points Above Average).
        
        Returns
        -------
        list of tuples
            List of (team_id, epaa_mean, epaa_std) sorted by epaa_mean
        """
        rankings = []
        for team_id in self.team_ids:
            stats = self.epaa_stats[team_id]
            rankings.append((team_id, stats['mean'], stats['std']))
        
        # Sort by EPAA (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_cluster_profiles(self):
        """
        Get the mean profiles for each cluster.
        
        Returns
        -------
        dict
            Dictionary with:
            - 'shot_selection': Mean shot selection profiles (J x K)
            - 'accuracy': Mean accuracy profiles (L x K)
        """
        # Average over posterior samples
        mu_mean = np.mean(self.posterior_samples['mu'], axis=0)
        eta_mean = np.mean(self.posterior_samples['eta'], axis=0)
        
        return {
            'shot_selection': mu_mean,
            'accuracy': eta_mean
        }
    
    def save(self, filepath):
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        BayesianBasketballHierarchical
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")
        return model


def calculate_epaa(team_shot_data, league_avg_ppg=None):
    """
    Calculate EPAA (Expected Points Above Average) for teams.
    
    This is a simplified calculation that doesn't require MCMC,
    useful for quick estimates or initialization.
    
    Parameters
    ----------
    team_shot_data : dict
        Dictionary mapping team_id to shot data with 'points_per_game'
    league_avg_ppg : float, optional
        League average points per game (computed if not provided)
        
    Returns
    -------
    dict
        Dictionary mapping team_id to EPAA value
    """
    if league_avg_ppg is None:
        league_avg_ppg = np.mean([data['points_per_game'] 
                                   for data in team_shot_data.values()])
    
    epaa_results = {}
    for team_id, data in team_shot_data.items():
        epaa = data['points_per_game'] - league_avg_ppg
        epaa_results[team_id] = {
            'epaa': epaa,
            'ppg': data['points_per_game'],
            'league_avg': league_avg_ppg
        }
    
    return epaa_results


def compare_team_matchup(model, home_team_id, away_team_id):
    """
    Compare two teams using the MCMC model.
    
    Parameters
    ----------
    model : BayesianBasketballHierarchical
        Fitted MCMC model
    home_team_id : int or str
        Home team identifier
    away_team_id : int or str
        Away team identifier
        
    Returns
    -------
    dict
        Dictionary with:
        - 'home_epaa': Home team EPAA statistics
        - 'away_epaa': Away team EPAA statistics
        - 'epaa_diff': Difference in EPAA (home - away)
        - 'predicted_spread': Predicted point spread
        - 'home_advantage': Typical home court advantage (~3 pts)
    """
    home_pred = model.predict_team_performance(home_team_id)
    away_pred = model.predict_team_performance(away_team_id)
    
    epaa_diff = home_pred['epaa_mean'] - away_pred['epaa_mean']
    home_advantage = 3.0  # Typical home court advantage
    
    predicted_spread = epaa_diff + home_advantage
    
    # Uncertainty in the spread
    spread_std = np.sqrt(home_pred['epaa_std']**2 + away_pred['epaa_std']**2)
    
    return {
        'home_epaa': home_pred,
        'away_epaa': away_pred,
        'epaa_diff': epaa_diff,
        'predicted_spread': predicted_spread,
        'spread_std': spread_std,
        'spread_ci': (
            predicted_spread - 1.96 * spread_std,
            predicted_spread + 1.96 * spread_std
        ),
        'home_advantage': home_advantage
    }


# Example usage and testing
if __name__ == "__main__":
    print("🏀 MCMC Basketball Model - Example Usage\n")
    
    # Create synthetic test data
    np.random.seed(42)
    K_REGIONS = 7  # Number of court regions
    
    # Simulate data for 10 teams
    test_data = {}
    for i in range(10):
        team_id = 1610612700 + i  # Example team IDs
        
        # Random shot distribution
        N_ik = np.random.multinomial(800, np.ones(K_REGIONS) / K_REGIONS)
        
        # Random makes (with some regions being better)
        fg_pcts = np.random.beta(4.5, 5.5, size=K_REGIONS)
        M_ik = np.array([np.random.binomial(n, p) 
                         for n, p in zip(N_ik, fg_pcts)])
        
        # Points per game
        ppg = np.random.normal(110, 8)
        
        test_data[team_id] = {
            'M_ik': M_ik,
            'N_ik': N_ik,
            'points_per_game': ppg
        }
    
    print("✅ Test data created for 10 teams\n")
    
    # Fit the model
    model = BayesianBasketballHierarchical(L=5, J=5, K=K_REGIONS)
    model.fit_gibbs(test_data, n_iterations=1000, burn_in=300)
    
    print("\n📊 EPAA Rankings:")
    print("="*50)
    rankings = model.get_epaa_rankings()
    for rank, (team_id, epaa, std) in enumerate(rankings, 1):
        print(f"{rank}. Team {team_id}: {epaa:+.2f} ± {std:.2f} EPAA")
    
    print("\n🎯 Example Matchup Prediction:")
    print("="*50)
    home_id = rankings[0][0]  # Best team
    away_id = rankings[-1][0]  # Worst team
    
    matchup = compare_team_matchup(model, home_id, away_id)
    print(f"Home Team {home_id} EPAA: {matchup['home_epaa']['epaa_mean']:+.2f}")
    print(f"Away Team {away_id} EPAA: {matchup['away_epaa']['epaa_mean']:+.2f}")
    print(f"Predicted Spread: {matchup['predicted_spread']:.2f} points")
    print(f"95% CI: ({matchup['spread_ci'][0]:.2f}, {matchup['spread_ci'][1]:.2f})")
