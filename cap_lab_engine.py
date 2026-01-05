"""
SDIS Cap Lab Engine
====================
NBA Salary Cap calculations, projections, and trade validation.
Data sourced from Spotrac with local caching.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - 2024-25 NBA CBA
# =============================================================================

CAP_NUMBERS = {
    "2024-25": {
        "salary_cap": 140_588_000,
        "luxury_tax": 170_814_000,
        "first_apron": 195_945_000,
        "second_apron": 207_824_000,
    },
    "2025-26": {
        "salary_cap": 146_000_000,
        "luxury_tax": 177_000_000,
        "first_apron": 203_000_000,
        "second_apron": 215_000_000,
    },
    "2026-27": {
        "salary_cap": 152_000_000,
        "luxury_tax": 184_000_000,
        "first_apron": 211_000_000,
        "second_apron": 224_000_000,
    },
    "2027-28": {
        "salary_cap": 158_000_000,
        "luxury_tax": 191_000_000,
        "first_apron": 219_000_000,
        "second_apron": 233_000_000,
    },
    "2028-29": {
        "salary_cap": 164_000_000,
        "luxury_tax": 199_000_000,
        "first_apron": 228_000_000,
        "second_apron": 242_000_000,
    },
}

# Tax brackets: (threshold, rate)
TAX_BRACKETS = [
    (5_000_000, 1.50),
    (5_000_000, 1.75),
    (5_000_000, 2.50),
    (5_000_000, 3.25),
]
TAX_RATE_ABOVE = 3.75
TAX_INCREMENT = 0.50

TEAM_SLUGS = {
    'ATL': 'atlanta-hawks', 'BOS': 'boston-celtics', 'BKN': 'brooklyn-nets',
    'CHA': 'charlotte-hornets', 'CHI': 'chicago-bulls', 'CLE': 'cleveland-cavaliers',
    'DAL': 'dallas-mavericks', 'DEN': 'denver-nuggets', 'DET': 'detroit-pistons',
    'GSW': 'golden-state-warriors', 'HOU': 'houston-rockets', 'IND': 'indiana-pacers',
    'LAC': 'los-angeles-clippers', 'LAL': 'los-angeles-lakers', 'MEM': 'memphis-grizzlies',
    'MIA': 'miami-heat', 'MIL': 'milwaukee-bucks', 'MIN': 'minnesota-timberwolves',
    'NOP': 'new-orleans-pelicans', 'NYK': 'new-york-knicks', 'OKC': 'oklahoma-city-thunder',
    'ORL': 'orlando-magic', 'PHI': 'philadelphia-76ers', 'PHX': 'phoenix-suns',
    'POR': 'portland-trail-blazers', 'SAC': 'sacramento-kings', 'SAS': 'san-antonio-spurs',
    'TOR': 'toronto-raptors', 'UTA': 'utah-jazz', 'WAS': 'washington-wizards'
}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CapLabError(Exception):
    """Base exception for Cap Lab errors."""
    pass


class ScrapingError(CapLabError):
    """Raised when web scraping fails."""
    pass


class ValidationError(CapLabError):
    """Raised when input validation fails."""
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TeamCapData:
    """Structured cap data for a single team."""
    team: str
    active_roster: float = 0.0
    dead_money: float = 0.0
    cap_holds: float = 0.0
    cap_space: float = 0.0
    cap_maximum: float = 0.0
    players: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'Team': self.team,
            'Active Roster': self.active_roster,
            'Dead Money': self.dead_money,
            'Cap Space': self.cap_space,
            'Cap Holds': self.cap_holds,
        }


# =============================================================================
# CAP CALCULATOR
# =============================================================================

class CapCalculator:
    """Calculates cap status, tax bills, and tier restrictions."""

    def __init__(self, season: str = "2024-25"):
        if season not in CAP_NUMBERS:
            raise ValidationError(f"Unknown season: {season}")
        self.season = season
        self.caps = CAP_NUMBERS[season]

    def get_tax_bill(self, total_salary: float, is_repeater: bool = False) -> float:
        """Calculate luxury tax bill based on salary over tax line."""
        over_tax = total_salary - self.caps["luxury_tax"]
        if over_tax <= 0:
            return 0.0

        bill = 0.0
        remaining = over_tax

        # Apply bracket rates
        for threshold, rate in TAX_BRACKETS:
            if remaining <= 0:
                break
            taxable = min(remaining, threshold)
            bill += taxable * rate
            remaining -= taxable

        # Beyond brackets: incremental rate increases
        if remaining > 0:
            rate = TAX_RATE_ABOVE
            while remaining > 0:
                taxable = min(remaining, 5_000_000)
                bill += taxable * rate
                remaining -= taxable
                rate += TAX_INCREMENT

        return bill * 1.5 if is_repeater else bill

    def get_tier(self, total_salary: float) -> str:
        """Determine cap tier based on total salary."""
        if total_salary <= self.caps["salary_cap"]:
            return "Under Cap"
        elif total_salary <= self.caps["luxury_tax"]:
            return "Over Cap"
        elif total_salary <= self.caps["first_apron"]:
            return "In Tax"
        elif total_salary <= self.caps["second_apron"]:
            return "First Apron"
        return "Second Apron"

    def get_status(self, total_salary: float) -> Dict:
        """Get comprehensive cap status."""
        return {
            "total_salary": total_salary,
            "salary_cap": self.caps["salary_cap"],
            "cap_space": self.caps["salary_cap"] - total_salary,
            "luxury_tax": self.caps["luxury_tax"],
            "tax_space": self.caps["luxury_tax"] - total_salary,
            "first_apron": self.caps["first_apron"],
            "second_apron": self.caps["second_apron"],
            "tier": self.get_tier(total_salary),
            "tax_bill": self.get_tax_bill(total_salary),
            "in_tax": total_salary > self.caps["luxury_tax"],
        }

    # Backward-compatible aliases
    def get_cap_status(self, total_salary: float) -> Dict:
        """Alias for get_status (backward compatibility)."""
        return self.get_status(total_salary)

    def get_apron_restrictions(self, total_salary: float) -> List[str]:
        """Alias for get_restrictions (backward compatibility)."""
        return self.get_restrictions(total_salary)

    def get_restrictions(self, total_salary: float) -> List[str]:
        """Get active CBA restrictions based on cap tier."""
        restrictions = []

        if total_salary > self.caps["first_apron"]:
            restrictions.extend([
                "No Bi-Annual Exception",
                "No salary aggregation in trades",
                "Cannot receive sign-and-trade",
                "Taxpayer MLE only (~$5.2M)",
            ])

        if total_salary > self.caps["second_apron"]:
            restrictions.extend([
                "7-year limit on trading first-round picks",
                "No traded player exception",
                "Pick swap restrictions",
            ])

        return restrictions


# =============================================================================
# TRADE MATCHER
# =============================================================================

class TradeMatcher:
    """Validates trade salary matching under CBA rules."""

    def __init__(self, season: str = "2024-25"):
        if season not in CAP_NUMBERS:
            raise ValidationError(f"Unknown season: {season}")
        self.caps = CAP_NUMBERS[season]

    def get_matching_range(self, outgoing: float, team_salary: float) -> Tuple[float, float, str]:
        """
        Calculate allowable incoming salary range.
        Returns: (min_incoming, max_incoming, rule_description)
        """
        cap = self.caps["salary_cap"]
        tax = self.caps["luxury_tax"]
        apron = self.caps["first_apron"]

        if team_salary <= cap:
            # Under cap: can absorb up to cap space + outgoing
            cap_space = cap - team_salary
            return 0, cap_space + outgoing, "Under cap - flexible"

        if team_salary <= tax:
            # Standard matching
            if outgoing <= 7_500_000:
                max_in = outgoing * 1.75 + 250_000
            elif outgoing <= 29_000_000:
                max_in = outgoing + 5_750_000
            else:
                max_in = outgoing * 1.25
            return outgoing * 0.75, max_in, "Standard (175%+$250k / +$5.75M / 125%)"

        if team_salary <= apron:
            # Tax apron matching
            return outgoing * 0.75, outgoing * 1.10 + 100_000, "Tax apron (110% + $100k)"

        # Above first apron - strictest
        return outgoing * 0.90, outgoing * 1.10, "Above apron (110% max)"

    def validate(self, t1_out: float, t1_in: float, t1_salary: float, t2_salary: float) -> Dict:
        """Validate a two-team trade."""
        t1_min, t1_max, t1_rule = self.get_matching_range(t1_out, t1_salary)
        t2_min, t2_max, t2_rule = self.get_matching_range(t1_in, t2_salary)

        t1_valid = t1_min <= t1_in <= t1_max
        t2_valid = t2_min <= t1_out <= t2_max

        return {
            "valid": t1_valid and t2_valid,
            "team1": {"valid": t1_valid, "min": t1_min, "max": t1_max, "rule": t1_rule},
            "team2": {"valid": t2_valid, "min": t2_min, "max": t2_max, "rule": t2_rule},
        }

    def validate_trade(self, t1_out: float, t1_in: float, t1_salary: float, t2_salary: float) -> Dict:
        """Validate trade with legacy response format."""
        t1_min, t1_max, t1_rule = self.get_matching_range(t1_out, t1_salary)
        t2_min, t2_max, t2_rule = self.get_matching_range(t1_in, t2_salary)

        t1_valid = t1_min <= t1_in <= t1_max
        t2_valid = t2_min <= t1_out <= t2_max

        # Determine cap situation label
        def get_situation(salary):
            if salary <= self.caps["salary_cap"]:
                return "Under Cap"
            elif salary <= self.caps["luxury_tax"]:
                return "Over Cap (below tax)"
            elif salary <= self.caps["first_apron"]:
                return "In Tax (below apron)"
            return "Above First Apron"

        return {
            "trade_valid": t1_valid and t2_valid,
            "team1_valid": t1_valid,
            "team2_valid": t2_valid,
            "team1_requirements": {
                "min_incoming": t1_min,
                "max_incoming": t1_max,
                "rule": t1_rule,
                "team_cap_situation": get_situation(t1_salary),
            },
            "team2_requirements": {
                "min_incoming": t2_min,
                "max_incoming": t2_max,
                "rule": t2_rule,
                "team_cap_situation": get_situation(t2_salary),
            },
        }


# Backward-compatible alias
TradeSalaryMatcher = TradeMatcher


# =============================================================================
# SCRAPER
# =============================================================================

class SpotracScraper:
    """Scrapes salary cap data from Spotrac with caching."""

    CACHE_DAYS = 7
    REQUEST_DELAY = 1.0

    # Rows to skip when parsing
    SKIP_PATTERNS = frozenset([
        'total', 'cap', 'space', 'roster', 'maximum', 'apron', 'mid-level',
        'bi-annual', 'trade', 'exception', 'room', 'luxury', 'tax', 'dead',
        'retained', 'waived', 'active', 'non-taxpayer', 'taxpayer'
    ])

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
        })

    def _parse_salary(self, text: str) -> Optional[float]:
        """Parse salary string to float. Returns None on failure."""
        if not text or text.strip() in ('', '-'):
            return None
        try:
            clean = re.sub(r'[^\d.\-]', '', text)
            return float(clean) if clean else None
        except ValueError:
            return None

    def _is_player_row(self, name: str) -> bool:
        """Check if row represents an actual player (not summary/exception)."""
        if not name or len(name) < 3:
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in self.SKIP_PATTERNS):
            return False
        if name[0].isdigit():  # Date like "6/29/2026"
            return False
        if '/' in name and any(c.isdigit() for c in name):
            return False
        return True

    def scrape_team(self, team: str) -> Optional[TeamCapData]:
        """Scrape cap data for a single team."""
        slug = TEAM_SLUGS.get(team)
        if not slug:
            logger.warning(f"Unknown team: {team}")
            return None

        url = f"https://www.spotrac.com/nba/{slug}/cap/"

        try:
            # Warm up session
            self.session.get("https://www.spotrac.com/nba/cap/", timeout=30)
            time.sleep(0.5)

            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch {team}: HTTP {resp.status_code}")
                return None

            soup = BeautifulSoup(resp.content, 'html.parser')
            data = TeamCapData(team=team)

            for row in soup.select('table tbody tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                # Get name
                anchor = cells[0].find('a')
                name = (anchor.get_text(strip=True) if anchor
                        else cells[0].get_text(strip=True))

                # Get salary
                salary = None
                for cell in cells[1:]:
                    text = cell.get_text(strip=True)
                    if '$' in text:
                        salary = self._parse_salary(text)
                        break

                if salary is None:
                    continue

                # Categorize
                name_lower = name.lower().strip()

                if 'active roster' in name_lower:
                    data.active_roster = salary
                elif 'dead' in name_lower:
                    data.dead_money = salary
                elif name_lower == 'space' and data.cap_space == 0:
                    data.cap_space = salary
                elif 'cap hold' in name_lower:
                    data.cap_holds = salary
                elif 'cap maximum' in name_lower and 'summary' not in name_lower:
                    data.cap_maximum = salary
                elif self._is_player_row(name):
                    data.players.append({'name': name, 'cap_hit': salary})

            return data

        except requests.RequestException as e:
            logger.error(f"Network error scraping {team}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {team}: {e}")
            return None

    def scrape_all(self, delay: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scrape all teams.
        Returns: (players_df, summary_df)
        """
        delay = delay or self.REQUEST_DELAY
        players = []
        summaries = []

        for team in TEAM_SLUGS:
            logger.info(f"Scraping {team}...")
            data = self.scrape_team(team)

            if data:
                summaries.append(data.to_dict())
                for p in data.players:
                    players.append({
                        'Player': p['name'],
                        'Cap Hit': p['cap_hit'],
                        'Team': team
                    })

            time.sleep(delay)

        players_df = pd.DataFrame(players)
        summary_df = pd.DataFrame(summaries)

        # Save to cache
        if not players_df.empty:
            players_df['2024-25'] = players_df['Cap Hit'].apply(lambda x: f"${x:,.0f}")
            players_df[['Player', '2024-25', 'Team']].to_excel(
                os.path.join(self.cache_dir, "nba_salaries.xlsx"), index=False
            )
        if not summary_df.empty:
            summary_df.to_excel(
                os.path.join(self.cache_dir, "nba_cap_summary.xlsx"), index=False
            )

        logger.info(f"Scraped {len(summaries)} teams, {len(players)} players")
        return players_df, summary_df

    def load_cached(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load cached data if fresh enough."""
        salary_file = os.path.join(self.cache_dir, "nba_salaries.xlsx")
        summary_file = os.path.join(self.cache_dir, "nba_cap_summary.xlsx")

        if not os.path.exists(salary_file):
            return None, None

        age_days = (datetime.now().timestamp() - os.path.getmtime(salary_file)) / 86400
        if age_days > self.CACHE_DAYS:
            logger.info(f"Cache expired ({age_days:.0f} days old)")
            return None, None

        players_df = pd.read_excel(salary_file)
        summary_df = pd.read_excel(summary_file) if os.path.exists(summary_file) else None

        return players_df, summary_df


# =============================================================================
# PUBLIC API (for backward compatibility with Streamlit app)
# =============================================================================

class SalaryDataScraper(SpotracScraper):
    """Backward-compatible wrapper for the Streamlit app."""

    def scrape_all_teams(self, delay: float = 1.0) -> pd.DataFrame:
        """Legacy method - returns just players DataFrame."""
        players_df, _ = self.scrape_all(delay)
        return players_df if players_df is not None else pd.DataFrame()


def get_cap_numbers(season: str = "2024-25") -> Dict:
    """Get cap numbers for a season."""
    return CAP_NUMBERS.get(season, CAP_NUMBERS["2024-25"])
