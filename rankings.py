import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict

@dataclass
class HeadToHead:
    """Stores head-to-head record between two teams"""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def points(self) -> float:
        """Returns points (wins + 0.5 * draws)"""
        return self.wins + 0.5 * self.draws
    
    @property
    def record_string(self) -> str:
        """Returns string like '3.5-1.5 (3-1-1)'"""
        points_against = self.total_games - self.points
        return f"{self.points}-{points_against} ({self.wins}-{self.losses}-{self.draws})"

class Team:
    def __init__(self, name: str):
        self.name = name
        self._records: Dict[str, HeadToHead] = defaultdict(HeadToHead)
    
    def versus(self, opponent: 'Team', wins: int, losses: int, draws: int = 0) -> None:
        """Record head-to-head results against an opponent"""
        # Record from this team's perspective
        self._records[opponent.name].wins += wins
        self._records[opponent.name].losses += losses
        self._records[opponent.name].draws += draws
        
        # Record from opponent's perspective
        opponent._records[self.name].wins += losses
        opponent._records[self.name].losses += wins
        opponent._records[self.name].draws += draws
    
    @property
    def total_games(self) -> int:
        return sum(h2h.total_games for h2h in self._records.values())
    
    @property
    def total_points(self) -> float:
        return sum(h2h.points for h2h in self._records.values())
    
    def winning_percentage(self) -> float:
        """Calculate the winning percentage"""
        if self.total_games == 0:
            return 0.0
        return (self.total_points / self.total_games) * 100

    def get_transition_probabilities(self, all_teams: List[str]) -> List[float]:
        """Calculate Markov transition probabilities to all teams"""
        if self.total_games == 0:
            # If team hasn't played, distribute evenly to all other teams
            probs = [1/(len(all_teams)-1) if team != self.name else 0 for team in all_teams]
            return probs
        
        # Calculate probability of transitioning to each team
        probs = []
        for team in all_teams:
            if team == self.name:
                # Probability of staying (winning percentage)
                prob = self.total_points / self.total_games
            else:
                # Probability of transitioning (losing to this specific team)
                h2h = self._records[team]
                prob = (h2h.losses + 0.5 * h2h.draws) / self.total_games if h2h.total_games > 0 else 0
            probs.append(prob)
        
        return probs
    
    def __str__(self) -> str:
        """Pretty print team's record against all opponents with alignment"""
        if not self._records:
            return f"{self.name}: No games played"
        
        # Align opponent names and records
        lines = [f"{self.name} vs ..."]
        max_opp_name_length = max(len(opp) for opp in self._records.keys()) if self._records else 0
        for opp_name, h2h in sorted(self._records.items()):
            lines.append(
                f"{opp_name:<{max_opp_name_length}}: {h2h.record_string:<15}"
            )
        
        # Add total record
        total_against = self.total_games - self.total_points
        total_record = f"{self.total_points}-{total_against} ({sum(h2h.wins for h2h in self._records.values())}-"
        total_record += f"{sum(h2h.losses for h2h in self._records.values())}-"
        total_record += f"{sum(h2h.draws for h2h in self._records.values())})"
        win_percent = f"{self.winning_percentage():.1f}%"
        lines.append(
            f"{'Total':<{max_opp_name_length}}: {total_record:<15} ({win_percent})"
        )
        return "\n".join(lines)

class League:
    def __init__(self, name: str):
        self.name = name
        self._teams: Dict[str, Team] = {}
    
    def add_team(self, name: str) -> Team:
        """Add a team to the league"""
        if name in self._teams:
            raise ValueError(f"Team {name} already exists in {self.name}")
        team = Team(name)
        self._teams[name] = team
        return team
    
    def get_team(self, name: str) -> Team:
        """Get a team by name"""
        return self._teams[name]
    
    def calculate_rankings(self) -> List[Tuple[str, float]]:
        """Calculate rankings using Markov chain steady state"""
        team_names = sorted(self._teams.keys())
        n_teams = len(team_names)
        
        # Build transition matrix
        P = np.zeros((n_teams, n_teams))
        for i, team in enumerate(team_names):
            P[i] = self._teams[team].get_transition_probabilities(team_names)
        
        # Calculate steady state
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        index = np.argmin(np.abs(eigenvalues - 1))
        steady_state = np.real(eigenvectors[:, index])
        steady_state = steady_state / np.sum(steady_state)
        
        # Return sorted team names and probabilities
        rankings = list(zip(team_names, steady_state))
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def print_rankings(self) -> None:
        """Print rankings in a formatted way"""
        rankings = self.calculate_rankings()
        print(f"\n{self.name} Rankings:")
        print("-" * 40)
        for i, (team, prob) in enumerate(rankings, 1):
            print(f"{i}. {team}: {prob:.3f}")
    
    def print_all_records(self) -> None:
        """Print all team records"""
        print(f"\n{self.name} Team Records:")
        print("-" * 40)
        team_names = sorted(self._teams.keys())
        for team_name in team_names:
            team = self._teams[team_name]
            # Ensure all teams are included in the output
            for other_name in team_names:
                if other_name not in team._records and other_name != team.name:
                    team._records[other_name] = HeadToHead()  # Add a placeholder for teams that haven't played
            print(f"\n{team}")

def premierStandard():
    # Create a league
    league = League("Premier League - 2022")
    
    # Add teams
    Arsenal = league.add_team("Arsenal")
    AstonVilla = league.add_team("Aston Villa")
    Bournemouth = league.add_team("Bournemouth")
    Brentford = league.add_team("Brentford")
    BrightonAndHoveAlbion = league.add_team("Brighton & Hove Albion")
    Chelsea = league.add_team("Chelsea")
    CrystalPalace = league.add_team("Crystal Palace")
    Everton = league.add_team("Everton")
    Fulham = league.add_team("Fulham")
    LeedsUnited = league.add_team("Leeds United")
    LeicesterCity = league.add_team("Leicester City")
    Liverpool = league.add_team("Liverpool")
    ManchesterCity = league.add_team("Manchester City")
    ManchesterUnited = league.add_team("Manchester United")
    NewcastleUnited = league.add_team("Newcastle United")
    NottinghamForest = league.add_team("Nottingham Forest")
    Southampton = league.add_team("Southampton")
    TottenhamHotspur = league.add_team("Tottenham Hotspur")
    WestHamUnited = league.add_team("West Ham United")
    WolverhamptonWanderers = league.add_team("Wolverhampton Wanderers")
    
    # Add head-to-head results
    Arsenal.versus(AstonVilla, 2, 0, 0)
    Arsenal.versus(Bournemouth, 2, 0, 0)
    Arsenal.versus(Brentford, 1.5, 0.5, 0)
    Arsenal.versus(BrightonAndHoveAlbion, 1, 1, 0)
    Arsenal.versus(Chelsea, 2, 0, 0)
    Arsenal.versus(CrystalPalace, 2, 0, 0)
    Arsenal.versus(Everton, 1, 1, 0)
    Arsenal.versus(Fulham, 2, 0, 0)
    Arsenal.versus(LeedsUnited, 2, 0, 0)
    Arsenal.versus(LeicesterCity, 2, 0, 0)
    Arsenal.versus(Liverpool, 1.5, 0.5, 0)
    Arsenal.versus(ManchesterCity, 0, 2, 0)
    Arsenal.versus(ManchesterUnited, 1, 1, 0)
    Arsenal.versus(NewcastleUnited, 1.5, 0.5, 0)
    Arsenal.versus(NottinghamForest, 1, 1, 0)
    Arsenal.versus(Southampton, 1, 1, 0)
    Arsenal.versus(TottenhamHotspur, 2, 0, 0)
    Arsenal.versus(WestHamUnited, 1.5, 0.5, 0)
    Arsenal.versus(WolverhamptonWanderers, 2, 0, 0)

    AstonVilla.versus(Bournemouth, 1, 1, 0)
    AstonVilla.versus(Brentford, 1.5, 0.5, 0)
    AstonVilla.versus(BrightonAndHoveAlbion, 2, 0, 0)
    AstonVilla.versus(Chelsea, 1, 1, 0)
    AstonVilla.versus(CrystalPalace, 1, 1, 0)
    AstonVilla.versus(Everton, 2, 0, 0)
    AstonVilla.versus(Fulham, 1, 1, 0)
    AstonVilla.versus(LeedsUnited, 1.5, 0.5, 0)
    AstonVilla.versus(LeicesterCity, 1, 1, 0)
    AstonVilla.versus(Liverpool, 0.5, 1.5, 0)
    AstonVilla.versus(ManchesterCity, 0.5, 1.5, 0)
    AstonVilla.versus(ManchesterUnited, 1, 1, 0)
    AstonVilla.versus(NewcastleUnited, 1, 1, 0)
    AstonVilla.versus(NottinghamForest, 1.5, 0.5, 0)
    AstonVilla.versus(Southampton, 2, 0, 0)
    AstonVilla.versus(TottenhamHotspur, 2, 0, 0)
    AstonVilla.versus(WestHamUnited, 0.5, 1.5, 0)
    AstonVilla.versus(WolverhamptonWanderers, 0.5, 1.5, 0)

    Bournemouth.versus(Brentford, 0.5, 1.5, 0)
    Bournemouth.versus(BrightonAndHoveAlbion, 0, 2, 0)
    Bournemouth.versus(Chelsea, 0, 2, 0)
    Bournemouth.versus(CrystalPalace, 0, 2, 0)
    Bournemouth.versus(Everton, 1, 1, 0)
    Bournemouth.versus(Fulham, 1.5, 0.5, 0)
    Bournemouth.versus(LeedsUnited, 1, 1, 0)
    Bournemouth.versus(LeicesterCity, 2, 0, 0)
    Bournemouth.versus(Liverpool, 1, 1, 0)
    Bournemouth.versus(ManchesterCity, 0, 2, 0)
    Bournemouth.versus(ManchesterUnited, 0, 2, 0)
    Bournemouth.versus(NewcastleUnited, 1, 1, 0)
    Bournemouth.versus(NottinghamForest, 1.5, 0.5, 0)
    Bournemouth.versus(Southampton, 1, 1, 0)
    Bournemouth.versus(TottenhamHotspur, 1, 1, 0)
    Bournemouth.versus(WestHamUnited, 0, 2, 0)
    Bournemouth.versus(WolverhamptonWanderers, 1.5, 0.5, 0)

    Brentford.versus(BrightonAndHoveAlbion, 1.5, 0.5, 0)
    Brentford.versus(Chelsea, 1.5, 0.5, 0)
    Brentford.versus(CrystalPalace, 1, 1, 0)
    Brentford.versus(Everton, 0.5, 1.5, 0)
    Brentford.versus(Fulham, 1, 1, 0)
    Brentford.versus(LeedsUnited, 1.5, 0.5, 0)
    Brentford.versus(LeicesterCity, 1, 1, 0)
    Brentford.versus(Liverpool, 1, 1, 0)
    Brentford.versus(ManchesterCity, 2, 0, 0)
    Brentford.versus(ManchesterUnited, 1, 1, 0)
    Brentford.versus(NewcastleUnited, 0, 2, 0)
    Brentford.versus(NottinghamForest, 1.5, 0.5, 0)
    Brentford.versus(Southampton, 2, 0, 0)
    Brentford.versus(TottenhamHotspur, 1.5, 0.5, 0)
    Brentford.versus(WestHamUnited, 2, 0, 0)
    Brentford.versus(WolverhamptonWanderers, 0.5, 1.5, 0)
        
    BrightonAndHoveAlbion.versus(Chelsea, 2, 0, 0)
    BrightonAndHoveAlbion.versus(CrystalPalace, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(Everton, 1, 1, 0)
    BrightonAndHoveAlbion.versus(Fulham, 0, 2, 0)
    BrightonAndHoveAlbion.versus(LeedsUnited, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(LeicesterCity, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(Liverpool, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(ManchesterCity, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(ManchesterUnited, 2, 0, 0)
    BrightonAndHoveAlbion.versus(NewcastleUnited, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(NottinghamForest, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(Southampton, 2, 0, 0)
    BrightonAndHoveAlbion.versus(TottenhamHotspur, 0, 2, 0)
    BrightonAndHoveAlbion.versus(WestHamUnited, 2, 0, 0)
    BrightonAndHoveAlbion.versus(WolverhamptonWanderers, 2, 0, 0)

    Chelsea.versus(CrystalPalace, 2, 0, 0)
    Chelsea.versus(Everton, 1.5, 0.5, 0)
    Chelsea.versus(Fulham, 0.5, 1.5, 0)
    Chelsea.versus(LeedsUnited, 1, 1, 0)
    Chelsea.versus(LeicesterCity, 2, 0, 0)
    Chelsea.versus(Liverpool, 1, 1, 0)
    Chelsea.versus(ManchesterCity, 0, 2, 0)
    Chelsea.versus(ManchesterUnited, 0.5, 1.5, 0)
    Chelsea.versus(NewcastleUnited, 0.5, 1.5, 0)
    Chelsea.versus(NottinghamForest, 1, 1, 0)
    Chelsea.versus(Southampton, 0, 2, 0)
    Chelsea.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Chelsea.versus(WestHamUnited, 1.5, 0.5, 0)
    Chelsea.versus(WolverhamptonWanderers, 1, 1, 0)

    CrystalPalace.versus(Everton, 0.5, 1.5, 0)
    CrystalPalace.versus(Fulham, 0.5, 1.5, 0)
    CrystalPalace.versus(LeedsUnited, 2, 0, 0)
    CrystalPalace.versus(LeicesterCity, 1.5, 0.5, 0)
    CrystalPalace.versus(Liverpool, 1, 1, 0)
    CrystalPalace.versus(ManchesterCity, 0, 2, 0)
    CrystalPalace.versus(ManchesterUnited, 0.5, 1.5, 0)
    CrystalPalace.versus(NewcastleUnited, 1, 1, 0)
    CrystalPalace.versus(NottinghamForest, 0.5, 1.5, 0)
    CrystalPalace.versus(Southampton, 2, 0, 0)
    CrystalPalace.versus(TottenhamHotspur, 0, 2, 0)
    CrystalPalace.versus(WestHamUnited, 2, 0, 0)
    CrystalPalace.versus(WolverhamptonWanderers, 1, 1, 0)

    Everton.versus(Fulham, 0.5, 1.5, 0)
    Everton.versus(LeedsUnited, 1.5, 0.5, 0)
    Everton.versus(LeicesterCity, 0.5, 1.5, 0)
    Everton.versus(Liverpool, 0.5, 1.5, 0)
    Everton.versus(ManchesterCity, 0.5, 1.5, 0)
    Everton.versus(ManchesterUnited, 0, 2, 0)
    Everton.versus(NewcastleUnited, 0, 2, 0)
    Everton.versus(NottinghamForest, 1, 1, 0)
    Everton.versus(Southampton, 1, 1, 0)
    Everton.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Everton.versus(WestHamUnited, 1, 1, 0)
    Everton.versus(WolverhamptonWanderers, 0.5, 1.5, 0)

    Fulham.versus(LeedsUnited, 2, 0, 0)
    Fulham.versus(LeicesterCity, 2, 0, 0)
    Fulham.versus(Liverpool, 0.5, 1.5, 0)
    Fulham.versus(ManchesterCity, 0, 2, 0)
    Fulham.versus(ManchesterUnited, 0, 2, 0)
    Fulham.versus(NewcastleUnited, 0, 2, 0)
    Fulham.versus(NottinghamForest, 2, 0, 0)
    Fulham.versus(Southampton, 2, 0, 0)
    Fulham.versus(TottenhamHotspur, 0, 2, 0)
    Fulham.versus(WestHamUnited, 0, 2, 0)
    Fulham.versus(WolverhamptonWanderers, 1, 1, 0)

    LeedsUnited.versus(LeicesterCity, 0.5, 1.5, 0)
    LeedsUnited.versus(Liverpool, 1, 1, 0)
    LeedsUnited.versus(ManchesterCity, 0, 2, 0)
    LeedsUnited.versus(ManchesterUnited, 0.5, 1.5, 0)
    LeedsUnited.versus(NewcastleUnited, 1, 1, 0)
    LeedsUnited.versus(NottinghamForest, 1, 1, 0)
    LeedsUnited.versus(Southampton, 1.5, 0.5, 0)
    LeedsUnited.versus(TottenhamHotspur, 0, 2, 0)
    LeedsUnited.versus(WestHamUnited, 0.5, 1.5, 0)
    LeedsUnited.versus(WolverhamptonWanderers, 2, 0, 0)

    LeicesterCity.versus(Liverpool, 0, 2, 0)
    LeicesterCity.versus(ManchesterCity, 0, 2, 0)
    LeicesterCity.versus(ManchesterUnited, 0, 2, 0)
    LeicesterCity.versus(NewcastleUnited, 0.5, 1.5, 0)
    LeicesterCity.versus(NottinghamForest, 1, 1, 0)
    LeicesterCity.versus(Southampton, 0, 2, 0)
    LeicesterCity.versus(TottenhamHotspur, 1, 1, 0)
    LeicesterCity.versus(WestHamUnited, 2, 0, 0)
    LeicesterCity.versus(WolverhamptonWanderers, 2, 0, 0)

    Liverpool.versus(ManchesterCity, 1, 1, 0)
    Liverpool.versus(ManchesterUnited, 1, 1, 0)
    Liverpool.versus(NewcastleUnited, 2, 0, 0)
    Liverpool.versus(NottinghamForest, 1, 1, 0)
    Liverpool.versus(Southampton, 1.5, 0.5, 0)
    Liverpool.versus(TottenhamHotspur, 2, 0, 0)
    Liverpool.versus(WestHamUnited, 2, 0, 0)
    Liverpool.versus(WolverhamptonWanderers, 1, 1, 0)

    ManchesterCity.versus(ManchesterUnited, 1, 1, 0)
    ManchesterCity.versus(NewcastleUnited, 1.5, 0.5, 0)
    ManchesterCity.versus(NottinghamForest, 1.5, 0.5, 0)
    ManchesterCity.versus(Southampton, 2, 0, 0)
    ManchesterCity.versus(TottenhamHotspur, 1, 1, 0)
    ManchesterCity.versus(WestHamUnited, 2, 0, 0)
    ManchesterCity.versus(WolverhamptonWanderers, 2, 0, 0)

    ManchesterUnited.versus(NewcastleUnited, 0.5, 1.5, 0)
    ManchesterUnited.versus(NottinghamForest, 2, 0, 0)
    ManchesterUnited.versus(Southampton, 1.5, 0.5, 0)
    ManchesterUnited.versus(TottenhamHotspur, 1.5, 0.5, 0)
    ManchesterUnited.versus(WestHamUnited, 1, 1, 0)
    ManchesterUnited.versus(WolverhamptonWanderers, 2, 0, 0)

    NewcastleUnited.versus(NottinghamForest, 2, 0, 0)
    NewcastleUnited.versus(Southampton, 2, 0, 0)
    NewcastleUnited.versus(TottenhamHotspur, 2, 0, 0)
    NewcastleUnited.versus(WestHamUnited, 1.5, 0.5, 0)
    NewcastleUnited.versus(WolverhamptonWanderers, 1.5, 0.5, 0)

    NottinghamForest.versus(Southampton, 2, 0, 0)
    NottinghamForest.versus(TottenhamHotspur, 0, 2, 0)
    NottinghamForest.versus(WestHamUnited, 1, 1, 0)
    NottinghamForest.versus(WolverhamptonWanderers, 0.5, 1.5, 0)

    Southampton.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Southampton.versus(WestHamUnited, 0.5, 1.5, 0)
    Southampton.versus(WolverhamptonWanderers, 0, 2, 0)

    TottenhamHotspur.versus(WestHamUnited, 1.5, 0.5, 0)
    TottenhamHotspur.versus(WolverhamptonWanderers, 1, 1, 0)

    WestHamUnited.versus(WolverhamptonWanderers, 1, 1, 0)

    # Print all records
    league.print_all_records()
    
    # Calculate and print rankings
    league.print_rankings()

def premierTest():
    # Create a league
    league = League("Premier League - Example")
    
    # Add teams
    Arsenal = league.add_team("Arsenal")
    AstonVilla = league.add_team("Aston Villa")
    Bournemouth = league.add_team("Bournemouth")
    Brentford = league.add_team("Brentford")
    BrightonAndHoveAlbion = league.add_team("Brighton & Hove Albion")
    Chelsea = league.add_team("Chelsea")
    CrystalPalace = league.add_team("Crystal Palace")
    Everton = league.add_team("Everton")
    Fulham = league.add_team("Fulham")
    LeedsUnited = league.add_team("Leeds United")
    LeicesterCity = league.add_team("Leicester City")
    Liverpool = league.add_team("Liverpool")
    ManchesterCity = league.add_team("Manchester City")
    ManchesterUnited = league.add_team("Manchester United")
    NewcastleUnited = league.add_team("Newcastle United")
    NottinghamForest = league.add_team("Nottingham Forest")
    Southampton = league.add_team("Southampton")
    TottenhamHotspur = league.add_team("Tottenham Hotspur")
    WestHamUnited = league.add_team("West Ham United")
    WolverhamptonWanderers = league.add_team("Wolverhampton Wanderers")
    
    # Add head-to-head results

    # all wins
    # Arsenal.versus(AstonVilla, 2, 0, 0)
    Arsenal.versus(Bournemouth, 2, 0, 0)
    Arsenal.versus(Brentford, 2, 0, 0)
    Arsenal.versus(BrightonAndHoveAlbion, 2, 0, 0)
    Arsenal.versus(Chelsea, 2, 0, 0)
    Arsenal.versus(CrystalPalace, 2, 0, 0)
    Arsenal.versus(Everton, 2, 0, 0)
    Arsenal.versus(Fulham, 2, 0, 0)
    Arsenal.versus(LeedsUnited, 2, 0, 0)
    Arsenal.versus(LeicesterCity, 2, 0, 0)
    Arsenal.versus(Liverpool, 2, 0, 0)
    Arsenal.versus(ManchesterCity, 2, 0, 0)
    Arsenal.versus(ManchesterUnited, 2, 0, 0)
    Arsenal.versus(NewcastleUnited, 2, 0, 0)
    Arsenal.versus(NottinghamForest, 2, 0, 0)
    Arsenal.versus(Southampton, 2, 0, 0)
    Arsenal.versus(TottenhamHotspur, 2, 0, 0)
    Arsenal.versus(WestHamUnited, 2, 0, 0)
    Arsenal.versus(WolverhamptonWanderers, 2, 0, 0)

    # all wins
    AstonVilla.versus(Bournemouth, 2, 0, 0)
    AstonVilla.versus(Brentford, 2, 0, 0)
    AstonVilla.versus(BrightonAndHoveAlbion, 2, 0, 0)
    AstonVilla.versus(Chelsea, 2, 0, 0)
    AstonVilla.versus(CrystalPalace, 2, 0, 0)
    AstonVilla.versus(Everton, 2, 0, 0)
    AstonVilla.versus(Fulham, 2, 0, 0)
    AstonVilla.versus(LeedsUnited, 2, 0, 0)
    AstonVilla.versus(LeicesterCity, 2, 0, 0)
    AstonVilla.versus(Liverpool, 2, 0, 0)
    AstonVilla.versus(ManchesterCity, 2, 0, 0)
    AstonVilla.versus(ManchesterUnited, 2, 0, 0)
    AstonVilla.versus(NewcastleUnited, 2, 0, 0)
    AstonVilla.versus(NottinghamForest, 2, 0, 0)
    AstonVilla.versus(Southampton, 2, 0, 0)
    AstonVilla.versus(TottenhamHotspur, 2, 0, 0)
    AstonVilla.versus(WestHamUnited, 2, 0, 0)
    AstonVilla.versus(WolverhamptonWanderers, 2, 0, 0)

    Bournemouth.versus(Brentford, 0.5, 1.5, 0)
    Bournemouth.versus(BrightonAndHoveAlbion, 0, 2, 0)
    Bournemouth.versus(Chelsea, 0, 2, 0)
    Bournemouth.versus(CrystalPalace, 0, 2, 0)
    Bournemouth.versus(Everton, 1, 1, 0)
    Bournemouth.versus(Fulham, 1.5, 0.5, 0)
    Bournemouth.versus(LeedsUnited, 1, 1, 0)
    Bournemouth.versus(LeicesterCity, 2, 0, 0)
    Bournemouth.versus(Liverpool, 1, 1, 0)
    Bournemouth.versus(ManchesterCity, 0, 2, 0)
    Bournemouth.versus(ManchesterUnited, 0, 2, 0)
    Bournemouth.versus(NewcastleUnited, 1, 1, 0)
    Bournemouth.versus(NottinghamForest, 1.5, 0.5, 0)
    Bournemouth.versus(Southampton, 1, 1, 0)
    Bournemouth.versus(TottenhamHotspur, 1, 1, 0)
    Bournemouth.versus(WestHamUnited, 0, 2, 0)
    Bournemouth.versus(WolverhamptonWanderers, 1.5, 0.5, 0)

    Brentford.versus(BrightonAndHoveAlbion, 1.5, 0.5, 0)
    Brentford.versus(Chelsea, 1.5, 0.5, 0)
    Brentford.versus(CrystalPalace, 1, 1, 0)
    Brentford.versus(Everton, 0.5, 1.5, 0)
    Brentford.versus(Fulham, 1, 1, 0)
    Brentford.versus(LeedsUnited, 1.5, 0.5, 0)
    Brentford.versus(LeicesterCity, 1, 1, 0)
    Brentford.versus(Liverpool, 1, 1, 0)
    Brentford.versus(ManchesterCity, 2, 0, 0)
    Brentford.versus(ManchesterUnited, 1, 1, 0)
    Brentford.versus(NewcastleUnited, 0, 2, 0)
    Brentford.versus(NottinghamForest, 1.5, 0.5, 0)
    Brentford.versus(Southampton, 2, 0, 0)
    Brentford.versus(TottenhamHotspur, 1.5, 0.5, 0)
    Brentford.versus(WestHamUnited, 2, 0, 0)
    Brentford.versus(WolverhamptonWanderers, 0.5, 1.5, 0)
        
    BrightonAndHoveAlbion.versus(Chelsea, 2, 0, 0)
    BrightonAndHoveAlbion.versus(CrystalPalace, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(Everton, 1, 1, 0)
    BrightonAndHoveAlbion.versus(Fulham, 0, 2, 0)
    BrightonAndHoveAlbion.versus(LeedsUnited, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(LeicesterCity, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(Liverpool, 1.5, 0.5, 0)
    BrightonAndHoveAlbion.versus(ManchesterCity, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(ManchesterUnited, 2, 0, 0)
    BrightonAndHoveAlbion.versus(NewcastleUnited, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(NottinghamForest, 0.5, 1.5, 0)
    BrightonAndHoveAlbion.versus(Southampton, 2, 0, 0)
    BrightonAndHoveAlbion.versus(TottenhamHotspur, 0, 2, 0)
    BrightonAndHoveAlbion.versus(WestHamUnited, 2, 0, 0)
    BrightonAndHoveAlbion.versus(WolverhamptonWanderers, 2, 0, 0)

    Chelsea.versus(CrystalPalace, 2, 0, 0)
    Chelsea.versus(Everton, 1.5, 0.5, 0)
    Chelsea.versus(Fulham, 0.5, 1.5, 0)
    Chelsea.versus(LeedsUnited, 1, 1, 0)
    Chelsea.versus(LeicesterCity, 2, 0, 0)
    Chelsea.versus(Liverpool, 1, 1, 0)
    Chelsea.versus(ManchesterCity, 0, 2, 0)
    Chelsea.versus(ManchesterUnited, 0.5, 1.5, 0)
    Chelsea.versus(NewcastleUnited, 0.5, 1.5, 0)
    Chelsea.versus(NottinghamForest, 1, 1, 0)
    Chelsea.versus(Southampton, 0, 2, 0)
    Chelsea.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Chelsea.versus(WestHamUnited, 1.5, 0.5, 0)
    Chelsea.versus(WolverhamptonWanderers, 1, 1, 0)

    CrystalPalace.versus(Everton, 0.5, 1.5, 0)
    CrystalPalace.versus(Fulham, 0.5, 1.5, 0)
    CrystalPalace.versus(LeedsUnited, 2, 0, 0)
    CrystalPalace.versus(LeicesterCity, 1.5, 0.5, 0)
    CrystalPalace.versus(Liverpool, 1, 1, 0)
    CrystalPalace.versus(ManchesterCity, 0, 2, 0)
    CrystalPalace.versus(ManchesterUnited, 0.5, 1.5, 0)
    CrystalPalace.versus(NewcastleUnited, 1, 1, 0)
    CrystalPalace.versus(NottinghamForest, 0.5, 1.5, 0)
    CrystalPalace.versus(Southampton, 2, 0, 0)
    CrystalPalace.versus(TottenhamHotspur, 0, 2, 0)
    CrystalPalace.versus(WestHamUnited, 2, 0, 0)
    CrystalPalace.versus(WolverhamptonWanderers, 1, 1, 0)

    Everton.versus(Fulham, 0.5, 1.5, 0)
    Everton.versus(LeedsUnited, 1.5, 0.5, 0)
    Everton.versus(LeicesterCity, 0.5, 1.5, 0)
    Everton.versus(Liverpool, 0.5, 1.5, 0)
    Everton.versus(ManchesterCity, 0.5, 1.5, 0)
    Everton.versus(ManchesterUnited, 0, 2, 0)
    Everton.versus(NewcastleUnited, 0, 2, 0)
    Everton.versus(NottinghamForest, 1, 1, 0)
    Everton.versus(Southampton, 1, 1, 0)
    Everton.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Everton.versus(WestHamUnited, 1, 1, 0)
    Everton.versus(WolverhamptonWanderers, 0.5, 1.5, 0)

    Fulham.versus(LeedsUnited, 2, 0, 0)
    Fulham.versus(LeicesterCity, 2, 0, 0)
    Fulham.versus(Liverpool, 0.5, 1.5, 0)
    Fulham.versus(ManchesterCity, 0, 2, 0)
    Fulham.versus(ManchesterUnited, 0, 2, 0)
    Fulham.versus(NewcastleUnited, 0, 2, 0)
    Fulham.versus(NottinghamForest, 2, 0, 0)
    Fulham.versus(Southampton, 2, 0, 0)
    Fulham.versus(TottenhamHotspur, 0, 2, 0)
    Fulham.versus(WestHamUnited, 0, 2, 0)
    Fulham.versus(WolverhamptonWanderers, 1, 1, 0)

    LeedsUnited.versus(LeicesterCity, 0.5, 1.5, 0)
    LeedsUnited.versus(Liverpool, 1, 1, 0)
    LeedsUnited.versus(ManchesterCity, 0, 2, 0)
    LeedsUnited.versus(ManchesterUnited, 0.5, 1.5, 0)
    LeedsUnited.versus(NewcastleUnited, 1, 1, 0)
    LeedsUnited.versus(NottinghamForest, 1, 1, 0)
    LeedsUnited.versus(Southampton, 1.5, 0.5, 0)
    LeedsUnited.versus(TottenhamHotspur, 0, 2, 0)
    LeedsUnited.versus(WestHamUnited, 0.5, 1.5, 0)
    LeedsUnited.versus(WolverhamptonWanderers, 2, 0, 0)

    LeicesterCity.versus(Liverpool, 0, 2, 0)
    LeicesterCity.versus(ManchesterCity, 0, 2, 0)
    LeicesterCity.versus(ManchesterUnited, 0, 2, 0)
    LeicesterCity.versus(NewcastleUnited, 0.5, 1.5, 0)
    LeicesterCity.versus(NottinghamForest, 1, 1, 0)
    LeicesterCity.versus(Southampton, 0, 2, 0)
    LeicesterCity.versus(TottenhamHotspur, 1, 1, 0)
    LeicesterCity.versus(WestHamUnited, 2, 0, 0)
    LeicesterCity.versus(WolverhamptonWanderers, 2, 0, 0)

    Liverpool.versus(ManchesterCity, 1, 1, 0)
    Liverpool.versus(ManchesterUnited, 1, 1, 0)
    Liverpool.versus(NewcastleUnited, 2, 0, 0)
    Liverpool.versus(NottinghamForest, 1, 1, 0)
    Liverpool.versus(Southampton, 1.5, 0.5, 0)
    Liverpool.versus(TottenhamHotspur, 2, 0, 0)
    Liverpool.versus(WestHamUnited, 2, 0, 0)
    Liverpool.versus(WolverhamptonWanderers, 1, 1, 0)

    ManchesterCity.versus(ManchesterUnited, 1, 1, 0)
    ManchesterCity.versus(NewcastleUnited, 1.5, 0.5, 0)
    ManchesterCity.versus(NottinghamForest, 1.5, 0.5, 0)
    ManchesterCity.versus(Southampton, 2, 0, 0)
    ManchesterCity.versus(TottenhamHotspur, 1, 1, 0)
    ManchesterCity.versus(WestHamUnited, 2, 0, 0)
    ManchesterCity.versus(WolverhamptonWanderers, 2, 0, 0)

    ManchesterUnited.versus(NewcastleUnited, 0.5, 1.5, 0)
    ManchesterUnited.versus(NottinghamForest, 2, 0, 0)
    ManchesterUnited.versus(Southampton, 1.5, 0.5, 0)
    ManchesterUnited.versus(TottenhamHotspur, 1.5, 0.5, 0)
    ManchesterUnited.versus(WestHamUnited, 1, 1, 0)
    ManchesterUnited.versus(WolverhamptonWanderers, 2, 0, 0)

    NewcastleUnited.versus(NottinghamForest, 2, 0, 0)
    NewcastleUnited.versus(Southampton, 2, 0, 0)
    NewcastleUnited.versus(TottenhamHotspur, 2, 0, 0)
    NewcastleUnited.versus(WestHamUnited, 1.5, 0.5, 0)
    NewcastleUnited.versus(WolverhamptonWanderers, 1.5, 0.5, 0)

    NottinghamForest.versus(Southampton, 2, 0, 0)
    NottinghamForest.versus(TottenhamHotspur, 0, 2, 0)
    NottinghamForest.versus(WestHamUnited, 1, 1, 0)
    NottinghamForest.versus(WolverhamptonWanderers, 0.5, 1.5, 0)

    Southampton.versus(TottenhamHotspur, 0.5, 1.5, 0)
    Southampton.versus(WestHamUnited, 0.5, 1.5, 0)
    Southampton.versus(WolverhamptonWanderers, 0, 2, 0)

    TottenhamHotspur.versus(WestHamUnited, 1.5, 0.5, 0)
    TottenhamHotspur.versus(WolverhamptonWanderers, 1, 1, 0)

    WestHamUnited.versus(WolverhamptonWanderers, 1, 1, 0)

    # Print all records
    league.print_all_records()
    
    # Calculate and print rankings
    league.print_rankings()

def exampleTest():
    # Create a league
    league = League("Example League")
    
    # Add teams
    TeamA = league.add_team("Team A")
    TeamB = league.add_team("Team B")
    TeamC = league.add_team("Team C")
    TeamD = league.add_team("Team D")
    
    # Add head-to-head results
    TeamA.versus(TeamC, 1, 0, 0)
    TeamA.versus(TeamD, 1, 0, 0)
    TeamB.versus(TeamC, 1, 0, 0)
    TeamB.versus(TeamD, 1, 0, 0)
    TeamC.versus(TeamD, 1, 0, 0)

    # Print all records
    league.print_all_records()
    
    # Calculate and print rankings
    league.print_rankings()

if __name__ == "__main__":
    exampleTest()