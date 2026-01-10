"""
SPRS - Daily Incremental Data Update
=====================================
Only pulls new games since the last update.
Runs quickly (few minutes) compared to full pull (hours).
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
from nba_api.stats.static import teams
import time
import re
import os
from datetime import datetime, timedelta

# Configuration
DATA_DIR = r"C:\Users\user"
COMBINED_FILE = os.path.join(DATA_DIR, "NBA_Quarter_ALL_Combined.xlsx")
INJURIES_FILE = os.path.join(DATA_DIR, "NBA_Injuries_Combined.xlsx")
REQUEST_DELAY = 0.6
CURRENT_SEASON = "2025-26"

print("=" * 60)
print("SPRS - Daily Incremental Update")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ============ HELPER FUNCTIONS (same as full pull) ============

def get_boxscore_from_cdn(game_id):
    url = f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_play_by_play(game_id):
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=120)
        return pbp.get_data_frames()[0]
    except:
        return pd.DataFrame()

def parse_clock_to_seconds(clock_str):
    if not clock_str or pd.isna(clock_str):
        return 0
    match = re.match(r'PT(\d+)M([\d.]+)S', str(clock_str))
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    return 0

def calculate_quarter_minutes(pbp_df, player_id, period):
    period_plays = pbp_df[pbp_df['period'] == period]
    player_events = period_plays[period_plays['personId'] == player_id]
    if player_events.empty:
        return 0.0
    first_clock = player_events['clock'].iloc[0] if len(player_events) > 0 else None
    last_clock = player_events['clock'].iloc[-1] if len(player_events) > 0 else None
    period_length = 720 if period <= 4 else 300
    if first_clock and last_clock:
        start_secs = parse_clock_to_seconds(first_clock)
        end_secs = parse_clock_to_seconds(last_clock)
        minutes = abs(start_secs - end_secs) / 60.0
        minutes = min(minutes * 1.3, period_length / 60.0)
        return round(minutes, 1)
    return 0.0

def extract_quarter_stats(pbp_df, player_id, player_name, period):
    period_plays = pbp_df[(pbp_df['period'] == period) & (pbp_df['personId'] == player_id)]
    stats = {'pts': 0, 'fgm': 0, 'fga': 0, 'trb': 0, 'ast': 0, 'stl': 0, 'blk': 0, 'tov': 0, 'pf': 0}

    for _, play in period_plays.iterrows():
        action = str(play.get('actionType', '')).lower()
        sub_type = str(play.get('subType', '')).lower()
        shot_result = str(play.get('shotResult', '')).lower()
        shot_value = play.get('shotValue', 0) or 0

        if action == 'made shot':
            stats['fgm'] += 1
            stats['fga'] += 1
            stats['pts'] += int(shot_value)
        elif action == 'missed shot':
            stats['fga'] += 1
        elif action == 'free throw':
            if shot_result == 'made':
                stats['pts'] += 1
        elif action == 'rebound':
            stats['trb'] += 1
        elif action == 'turnover':
            stats['tov'] += 1
        elif action == 'foul' and ('personal' in sub_type or 'shooting' in sub_type):
            stats['pf'] += 1

    all_period = pbp_df[pbp_df['period'] == period]
    for _, play in all_period.iterrows():
        desc = str(play.get('description', ''))
        name_parts = player_name.lower().split()
        desc_lower = desc.lower()

        if 'AST)' in desc:
            for part in name_parts:
                if len(part) > 2 and part in desc_lower:
                    stats['ast'] += 1
                    break
        if 'steal' in desc_lower:
            for part in name_parts:
                if len(part) > 2 and part in desc_lower:
                    stats['stl'] += 1
                    break
        if 'block' in desc_lower and play.get('personId') != player_id:
            for part in name_parts:
                if len(part) > 2 and part in desc_lower:
                    stats['blk'] += 1
                    break
    return stats


# ============ INCREMENTAL UPDATE LOGIC ============

def get_last_game_date():
    """Get the most recent game date from existing data."""
    if os.path.exists(COMBINED_FILE):
        try:
            df = pd.read_excel(COMBINED_FILE)
            df['game_date'] = pd.to_datetime(df['game_date'])
            return df['game_date'].max()
        except:
            pass
    return None

def get_existing_game_ids():
    """Get set of game IDs already in the data."""
    if os.path.exists(COMBINED_FILE):
        try:
            df = pd.read_excel(COMBINED_FILE)
            return set(df['game_id'].astype(str).unique())
        except:
            pass
    return set()

def pull_new_games():
    """Pull only games that aren't already in the data."""

    existing_ids = get_existing_game_ids()
    last_date = get_last_game_date()

    if last_date:
        print(f"Last game in data: {last_date.strftime('%Y-%m-%d')}")
        # Start from 1 day before last date to catch any missed games
        start_date = last_date - timedelta(days=1)
    else:
        print("No existing data found. Run full pull first.")
        return pd.DataFrame()

    print(f"Checking for new games since {start_date.strftime('%Y-%m-%d')}...")

    # Get recent games
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=CURRENT_SEASON,
        league_id_nullable='00',
        timeout=180
    )
    games_df = finder.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)

    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    # Filter to new games only
    new_games = games_df[games_df['GAME_DATE'] >= start_date].drop_duplicates(subset=['GAME_ID'])
    new_games = new_games[~new_games['GAME_ID'].astype(str).isin(existing_ids)]

    if len(new_games) == 0:
        print("No new games to process.")
        return pd.DataFrame()

    print(f"Found {len(new_games)} new games to process")

    # Process new games
    all_data = []
    for idx, (_, game_row) in enumerate(new_games.iterrows()):
        game_id = str(game_row['GAME_ID']).zfill(10)
        game_date = game_row['GAME_DATE']

        print(f"  [{idx+1}/{len(new_games)}] Game {game_id}...", end=' ', flush=True)

        cdn_data = get_boxscore_from_cdn(game_id)
        time.sleep(REQUEST_DELAY)

        if not cdn_data:
            print('no CDN data')
            continue

        pbp = get_play_by_play(game_id)
        time.sleep(REQUEST_DELAY)

        if pbp.empty:
            print('no PBP data')
            continue

        game_info = cdn_data.get('game', {})
        game_records = []

        # Extract final game scores
        home_team_info = game_info.get('homeTeam', {})
        away_team_info = game_info.get('awayTeam', {})
        home_team = home_team_info.get('teamTricode', '')
        away_team = away_team_info.get('teamTricode', '')
        home_score = home_team_info.get('score', 0)
        away_score = away_team_info.get('score', 0)

        for team_key in ['homeTeam', 'awayTeam']:
            team_data = game_info.get(team_key, {})
            team_abbrev = team_data.get('teamTricode', '')
            players = team_data.get('players', [])

            team_game = games_df[(games_df['GAME_ID'] == int(game_id)) & (games_df['TEAM_ABBREVIATION'] == team_abbrev)]
            win_loss = team_game['WL'].iloc[0] if len(team_game) > 0 else ''

            periods = sorted(pbp['period'].unique())

            for player in players:
                if not player.get('played', False):
                    continue

                player_id = player.get('personId')
                player_name = player.get('name', '')

                for period in periods:
                    qtr_label = f'Q{period}' if period <= 4 else f'OT{period-4}'
                    stats = extract_quarter_stats(pbp, player_id, player_name, period)
                    minutes = calculate_quarter_minutes(pbp, player_id, period)

                    if stats['pts'] > 0 or stats['fga'] > 0 or stats['trb'] > 0 or minutes > 0:
                        game_records.append({
                            'player': player_name,
                            'game_date': game_date,
                            'game_id': game_id,
                            'qtr': qtr_label,
                            'qtr_num': period,
                            'pts': stats['pts'],
                            'trb': stats['trb'],
                            'ast': stats['ast'],
                            'stl': stats['stl'],
                            'blk': stats['blk'],
                            'tov': stats['tov'],
                            'pf': stats['pf'],
                            'fgm': stats['fgm'],
                            'fga': stats['fga'],
                            'minutes': minutes,
                            'team': team_abbrev,
                            'win_loss': win_loss,
                            'dataset': f'{team_abbrev} {CURRENT_SEASON}',
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score
                        })

        all_data.extend(game_records)
        print(f'{len(game_records)} records')

    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()


def pull_new_injuries():
    """Pull injury reports for the last 3 days."""

    TEAM_NAMES = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }

    def parse_injury_pdf(pdf_content):
        pdf = PdfReader(BytesIO(pdf_content))
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + " "
        text = re.sub(r'\s+', ' ', all_text)
        records = []

        for full_name, abbrev in TEAM_NAMES.items():
            team_pattern = full_name.replace(' ', r'\s+')
            team_matches = list(re.finditer(team_pattern, text, re.IGNORECASE))

            for match in team_matches:
                start_pos = match.end()
                next_team_pos = len(text)

                for other_name in TEAM_NAMES.keys():
                    if other_name == full_name:
                        continue
                    other_pattern = other_name.replace(' ', r'\s+')
                    other_match = re.search(other_pattern, text[start_pos:], re.IGNORECASE)
                    if other_match:
                        next_team_pos = min(next_team_pos, start_pos + other_match.start())

                team_section = text[start_pos:next_team_pos]
                pattern = r'([A-Z][a-z]+(?:\s+(?:Jr\.|Sr\.|III|II|IV))?),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(Out|Questionable|Doubtful|Available|Probable)\s+([^A-Z][^,]*?)(?=[A-Z][a-z]+,|$)'

                for m in re.finditer(pattern, team_section):
                    records.append({
                        'team': abbrev,
                        'player': f'{m.group(1)}, {m.group(2)}',
                        'status': m.group(3),
                        'reason': m.group(4).strip()[:80]
                    })

        seen = set()
        unique = []
        for r in records:
            key = (r['team'], r['player'], r['status'])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    print("\nPulling recent injury reports...")
    all_records = []

    for days_ago in range(3):
        date = datetime.now() - timedelta(days=days_ago)
        date_str = date.strftime("%Y-%m-%d")

        for suffix in ["07_00PM", "06_00PM", "05_00PM"]:
            url = f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date_str}_{suffix}.pdf"
            headers = {'User-Agent': 'Mozilla/5.0'}

            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    records = parse_injury_pdf(response.content)
                    for r in records:
                        r['report_date'] = date_str
                    all_records.extend(records)
                    print(f"  {date_str}: {len(records)} records")
                    break
            except:
                pass

        time.sleep(0.3)

    if all_records:
        return pd.DataFrame(all_records)
    return pd.DataFrame()


def main():
    # Pull new quarter data
    print("\n[1/2] Checking for new games...")
    new_quarter_data = pull_new_games()

    if not new_quarter_data.empty:
        # Append to existing file
        if os.path.exists(COMBINED_FILE):
            existing = pd.read_excel(COMBINED_FILE)
            combined = pd.concat([existing, new_quarter_data], ignore_index=True)
            # Remove any duplicates (by game_id + player + qtr)
            combined = combined.drop_duplicates(subset=['game_id', 'player', 'qtr'], keep='last')
        else:
            combined = new_quarter_data

        combined.to_excel(COMBINED_FILE, index=False)
        print(f"\nQuarter data updated: {len(new_quarter_data)} new records added")
        print(f"Total records: {len(combined)}")

    # Pull new injury data
    print("\n[2/2] Updating injury data...")
    new_injuries = pull_new_injuries()

    if not new_injuries.empty:
        if os.path.exists(INJURIES_FILE):
            existing = pd.read_excel(INJURIES_FILE)
            combined = pd.concat([existing, new_injuries], ignore_index=True)
            combined = combined.drop_duplicates(subset=['report_date', 'team', 'player'], keep='last')
        else:
            combined = new_injuries

        combined.to_excel(INJURIES_FILE, index=False)
        print(f"Injury data updated: {len(new_injuries)} new records")

    print("\n" + "=" * 60)
    print(f"Daily update complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
