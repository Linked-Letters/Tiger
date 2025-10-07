# This file is part of the Tiger software, a collection of python tools for
# scraping sports data, and generating team ratings and game predictions.
# 
# Copyright (C) 2025 George Limpert
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>. 

import joblib
import json
import math
import numbers
import numpy as np
import scipy.stats as stats
import sys
import tqdm

def simulate_game (home_offense_rating, home_defense_rating, home_teamid, away_offense_rating, away_defense_rating, away_teamid, points_mean, points_stdev, points_df, points_use_norm_dist, home_advantage, is_neutral_site, season_rng):
	if not is_neutral_site:
		cur_homeadv = home_advantage
	else:
		cur_homeadv = 0
	cur_homepts_mean = home_offense_rating + cur_homeadv / 2 - away_defense_rating + points_mean
	cur_awaypts_mean = away_offense_rating - cur_homeadv / 2 - home_defense_rating + points_mean
	picked_winner = False
	while not picked_winner:
		if points_use_norm_dist:
			cur_homepts = max(season_rng.normal(loc = 0, scale = 1) * points_stdev + cur_homepts_mean, 0)
			cur_awaypts = max(season_rng.normal(loc = 0, scale = 1) * points_stdev + cur_awaypts_mean, 0)
		else:
			cur_homepts = max(season_rng.standard_t(df = points_df) * points_stdev + cur_homepts_mean, 0)
			cur_awaypts = max(season_rng.standard_t(df = points_df) * points_stdev + cur_awaypts_mean, 0)
		if cur_homepts != cur_awaypts:
			picked_winner = True
	if cur_homepts > cur_awaypts:
		return home_teamid, away_teamid
	else:
		return away_teamid, home_teamid

def simulate_season (seasonid, prediction_season, input_data):
	# Get a list of teams, and set some other basic information
	teamid_list = list(input_data['TeamRatings'].keys())
	teamid_divisions = dict(zip(teamid_list, [input_data['TeamRatings'][x]['Division'] for x in teamid_list]))
	teamid_conferences = dict(zip(teamid_list, [input_data['TeamRatings'][x]['Conference'] for x in teamid_list]))
	divisions_list = list(set([input_data['TeamRatings'][x]['Division'] for x in teamid_list]))
	conferences_list = list(set([input_data['TeamRatings'][x]['Conference'] for x in teamid_list]))
	#home_advantage = input_data['HomeAdvantage']
	tie_cdf_bound = input_data['TieCDFBound']
	points_mean = input_data['ScoreMean']
	points_use_norm_dist = input_data['IsScoreErrorNormal']
	points_df = input_data['ScoreErrorDF']
	points_stdev = input_data['ScoreErrorStDev']

	# Choose different team ratings in different seasons
	rating_id = seasonid % input_data['NumberOfRatingAttempts']
	home_advantage = input_data['HomeAdvantageList'][rating_id]

	# Simulate the actual season
	season_data = {}
	# Prepare an empty data structure for each team
	for teamid in teamid_list:
		season_data[teamid] = {'Wins': 0, 'Losses': 0, 'Ties': 0, 'PointsFor': 0, 'PointsAgainst': 0, 'WinsHeadToHead': dict(zip(teamid_list, [0] * len(teamid_list))), 'LossesHeadToHead': dict(zip(teamid_list, [0] * len(teamid_list))), 'TiesHeadToHead': dict(zip(teamid_list, [0] * len(teamid_list))), 'PointsForHeadToHead': dict(zip(teamid_list, [0] * len(teamid_list))), 'PointsAgainstHeadToHead': dict(zip(teamid_list, [0] * len(teamid_list))), 'WinsDivision': 0, 'LossesDivision': 0, 'TiesDivision': 0, 'WinsConference': 0, 'LossesConference': 0, 'TiesConference': 0, 'PointsForDivision': 0, 'PointsAgainstDivision': 0, 'PointsForConference': 0, 'PointsAgainstConference': 0, 'PointDifferential': 0.0, 'Win%': 0.0, 'Win%HeadToHead': dict(zip(teamid_list, [0.0] * len(teamid_list))), 'PointDifferentialHeadToHead': dict(zip(teamid_list, [0.0] * len(teamid_list))), 'Win%Conference': 0.0, 'Win%Division': 0.0, 'PointDifferentialConference': 0.0, 'PointDifferentialDivision': 0.0, 'WonConference': False, 'WonDivision': False, 'WildCard': False, 'MadePlayoffs': False, 'PlayoffSeed': 0, 'MadeDivisionRound': False, 'MadeConferenceFinal': False, 'MadeSuperBowl': False, 'WonSuperBowl': False}
	season_rng = np.random.default_rng(seed = [seasonid])
	# Loop through past and future games
	input_games = input_data['PastSchedule'] + input_data['FutureSchedule']
	for cur_game in input_games:
		if (cur_game['Season'] == prediction_season) and (not cur_game['IsPreseason']):
			cur_homeid = cur_game['HomeID']
			cur_awayid = cur_game['AwayID']
			# Determine if there's a home advantage
			if not cur_game['IsNeutralSite']:
				cur_homeadv = home_advantage
			else:
				cur_homeadv = 0
			# If the game has been played, use the score of the game
			if (list(cur_game.keys()).count('HomeScore') > 0) and (list(cur_game.keys()).count('AwayScore') > 0) and (cur_game['HomeScore'] is not None) and (cur_game['AwayScore'] is not None) and (isinstance(cur_game['HomeScore'], numbers.Number)) and (isinstance(cur_game['AwayScore'], numbers.Number)):
				cur_wasplayed = True
				cur_homepts = cur_game['HomeScore']
				cur_awaypts = cur_game['AwayScore']
			# Otherwise, predict the score of the game
			else:
				cur_wasplayed = False
				cur_homepts_mean = input_data['TeamRatings'][cur_homeid]['OffenseRatingList'][rating_id] + cur_homeadv / 2 - input_data['TeamRatings'][cur_awayid]['DefenseRatingList'][rating_id] + points_mean
				cur_awaypts_mean = input_data['TeamRatings'][cur_awayid]['OffenseRatingList'][rating_id] - cur_homeadv / 2 - input_data['TeamRatings'][cur_homeid]['DefenseRatingList'][rating_id] + points_mean
				if points_use_norm_dist:
					cur_homepts = max(season_rng.normal(loc = 0, scale = 1) * points_stdev + cur_homepts_mean, 0)
					cur_awaypts = max(season_rng.normal(loc = 0, scale = 1) * points_stdev + cur_awaypts_mean, 0)
				else:
					cur_homepts = max(season_rng.standard_t(df = points_df) * points_stdev + cur_homepts_mean, 0)
					cur_awaypts = max(season_rng.standard_t(df = points_df) * points_stdev + cur_awaypts_mean, 0)
			# Determine if it's a conference game and a division game
			cur_isconferencegame = (cur_game['HomeConference'] == cur_game['AwayConference'])
			cur_isdivisiongame = (cur_game['HomeDivision'] == cur_game['AwayDivision'])
			# Determine if the game should be treated as a tie
			if cur_wasplayed:
				cur_istie = (cur_homepts == cur_awaypts)
			else:
				cur_istie = abs(cur_homepts - cur_awaypts) < tie_cdf_bound
			if cur_istie:
				cur_homewin = False
				cur_homeloss = False
			else:
				cur_homewin = (cur_homepts > cur_awaypts)
				cur_homeloss = (cur_homepts < cur_awaypts)
			# Record wins, losses, and ties
			if cur_istie:
				season_data[cur_homeid]['Ties'] += 1
				season_data[cur_awayid]['Ties'] += 1
				season_data[cur_homeid]['TiesHeadToHead'][cur_awayid] += 1
				season_data[cur_awayid]['TiesHeadToHead'][cur_homeid] += 1
				if cur_isconferencegame:
					season_data[cur_homeid]['TiesConference'] += 1
					season_data[cur_awayid]['TiesConference'] += 1
				if cur_isdivisiongame:
					season_data[cur_homeid]['TiesDivision'] += 1
					season_data[cur_awayid]['TiesDivision'] += 1
			elif cur_homewin:
				season_data[cur_homeid]['Wins'] += 1
				season_data[cur_awayid]['Losses'] += 1
				season_data[cur_homeid]['WinsHeadToHead'][cur_awayid] += 1
				season_data[cur_awayid]['LossesHeadToHead'][cur_homeid] += 1
				if cur_isconferencegame:
					season_data[cur_homeid]['WinsConference'] += 1
					season_data[cur_awayid]['LossesConference'] += 1
				if cur_isdivisiongame:
					season_data[cur_homeid]['WinsDivision'] += 1
					season_data[cur_awayid]['LossesDivision'] += 1
			elif cur_homeloss:
				season_data[cur_homeid]['Losses'] += 1
				season_data[cur_awayid]['Wins'] += 1
				season_data[cur_homeid]['LossesHeadToHead'][cur_awayid] += 1
				season_data[cur_awayid]['WinsHeadToHead'][cur_homeid] += 1
				if cur_isconferencegame:
					season_data[cur_homeid]['LossesConference'] += 1
					season_data[cur_awayid]['WinsConference'] += 1
				if cur_isdivisiongame:
					season_data[cur_homeid]['LossesDivision'] += 1
					season_data[cur_awayid]['WinsDivision'] += 1
			# Record points
			season_data[cur_homeid]['PointsFor'] += cur_homepts
			season_data[cur_homeid]['PointsAgainst'] += cur_awaypts
			season_data[cur_awayid]['PointsFor'] += cur_awaypts
			season_data[cur_awayid]['PointsAgainst'] += cur_homepts
			season_data[cur_homeid]['PointsForHeadToHead'][cur_awayid] += cur_homepts
			season_data[cur_homeid]['PointsAgainstHeadToHead'][cur_awayid] += cur_awaypts
			season_data[cur_awayid]['PointsForHeadToHead'][cur_homeid] += cur_awaypts
			season_data[cur_awayid]['PointsAgainstHeadToHead'][cur_homeid] += cur_homepts
			if cur_isconferencegame:
				season_data[cur_homeid]['PointsForConference'] += cur_homepts
				season_data[cur_homeid]['PointsAgainstConference'] += cur_awaypts
				season_data[cur_awayid]['PointsForConference'] += cur_awaypts
				season_data[cur_awayid]['PointsAgainstConference'] += cur_homepts
			if cur_isdivisiongame:
				season_data[cur_homeid]['PointsForDivision'] += cur_homepts
				season_data[cur_homeid]['PointsAgainstDivision'] += cur_awaypts
				season_data[cur_awayid]['PointsForDivision'] += cur_awaypts
				season_data[cur_awayid]['PointsAgainstDivision'] += cur_homepts
	# Calculate team statistics
	for teamid in teamid_list:
		for team2id in teamid_list:
			season_data[teamid]['PointDifferentialHeadToHead'][team2id] = season_data[teamid]['PointsForHeadToHead'][team2id] - season_data[teamid]['PointsAgainstHeadToHead'][team2id]
			if season_data[teamid]['WinsHeadToHead'][team2id] + season_data[teamid]['LossesHeadToHead'][team2id] + season_data[teamid]['TiesHeadToHead'][team2id] == 0:
				season_data[teamid]['Win%HeadToHead'][team2id] = math.nan
			else:
				season_data[teamid]['Win%HeadToHead'][team2id] = (season_data[teamid]['WinsHeadToHead'][team2id] + (season_data[teamid]['TiesHeadToHead'][team2id] / 2)) / (season_data[teamid]['WinsHeadToHead'][team2id] + season_data[teamid]['LossesHeadToHead'][team2id] + season_data[teamid]['TiesHeadToHead'][team2id])
		season_data[teamid]['Win%Division'] = (season_data[teamid]['WinsDivision'] + (season_data[teamid]['TiesDivision'] / 2)) / (season_data[teamid]['WinsDivision'] + season_data[teamid]['LossesDivision'] + season_data[teamid]['TiesDivision'])
		season_data[teamid]['PointDifferentialDivision'] = season_data[teamid]['PointsForDivision'] - season_data[teamid]['PointsAgainstDivision']
		season_data[teamid]['Win%Conference'] = (season_data[teamid]['WinsConference'] + (season_data[teamid]['TiesConference'] / 2)) / (season_data[teamid]['WinsConference'] + season_data[teamid]['LossesConference'] + season_data[teamid]['TiesConference'])
		season_data[teamid]['PointDifferentialConference'] = season_data[teamid]['PointsForConference'] - season_data[teamid]['PointsAgainstConference']
		season_data[teamid]['Win%'] = (season_data[teamid]['Wins'] + (season_data[teamid]['Ties'] / 2)) / (season_data[teamid]['Wins'] + season_data[teamid]['Losses'] + season_data[teamid]['Ties'])
		season_data[teamid]['PointDifferential'] = season_data[teamid]['PointsFor'] - season_data[teamid]['PointsAgainst']
	# Figure out division winners with basic tiebreakers (not actual NFL tiebreakers)
	for cur_division in divisions_list:
		subset_teams = [x for x in list(season_data.keys()) if cur_division == teamid_divisions[x]]
		win_pct_list = [season_data[x]['Win%'] for x in subset_teams]
		max_win_pct = max(win_pct_list)
		remaining_teams = [x for x in subset_teams if season_data[x]['Win%'] == max_win_pct]
		while len(remaining_teams) >= 2:
			removed_team = None
			if len(remaining_teams) == 2:
				# Head to head winning percentage and point differential
				if season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
					removed_team = remaining_teams[1]
				elif season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
					removed_team = remaining_teams[0]
				elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
					removed_team = remaining_teams[1]
				elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
					removed_team = remaining_teams[0]
			# Loop through a number of options to eliminate teams
			team_removal_stats = ['Win%Division', 'PointDifferentialDivision', 'Win%Conference', 'PointDifferentialConference', 'PointDifferential']
			for removal_stat in team_removal_stats:
				if removed_team is None:
					subset_stat_list = [season_data[x][removal_stat] for x in remaining_teams]
					subset_min_stat = min(subset_stat_list)
					lowest_team_list = [x for x in remaining_teams if season_data[x][removal_stat] == subset_min_stat]
					if len(lowest_team_list) == 1:
						removed_team = lowest_team_list[0]
			# Randomly select an eliminated team
			while removed_team is None:
				subset_stat_list = season_rng.uniform(size = len(remaining_teams)).tolist()
				subset_min_stat = min(subset_stat_list)
				lowest_team_list = [remaining_teams[x] for x in range(0, len(remaining_teams), 1) if subset_stat_list[x] == subset_min_stat]
				if len(lowest_team_list) == 1:
					removed_team = lowest_team_list[0]
			remaining_teams = [x for x in remaining_teams if x != removed_team]
		winner_team = remaining_teams[0]
		season_data[winner_team]['WonDivision'] = True
		season_data[winner_team]['MadePlayoffs'] = True

	# Determine conference winners, wild card teams, and playoff seeds
	for cur_conference in conferences_list:
		# Pick conference winners and seed division winners
		subset_teams = [x for x in list(season_data.keys()) if (cur_conference == teamid_conferences[x]) and (season_data[x]['WonDivision'])]
		playoff_seed = 1
		while len(subset_teams) > 0:
			win_pct_list = [season_data[x]['Win%'] for x in subset_teams]
			max_win_pct = max(win_pct_list)
			remaining_teams = [x for x in subset_teams if season_data[x]['Win%'] == max_win_pct]
			while len(remaining_teams) >= 2:
				removed_team = None
				if len(remaining_teams) == 2:
					# Head to head winning percentage and point differential
					if season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[1]
					elif season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[0]
					elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[1]
					elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[0]
				# Loop through a number of options to eliminate teams
				team_removal_stats = ['Win%Conference', 'PointDifferentialConference', 'PointDifferential']
				for removal_stat in team_removal_stats:
					if removed_team is None:
						subset_stat_list = [season_data[x][removal_stat] for x in remaining_teams]
						subset_min_stat = min(subset_stat_list)
						lowest_team_list = [x for x in remaining_teams if season_data[x][removal_stat] == subset_min_stat]
						if len(lowest_team_list) == 1:
							removed_team = lowest_team_list[0]
				# Randomly select an eliminated team
				while removed_team is None:
					subset_stat_list = season_rng.uniform(size = len(remaining_teams)).tolist()
					subset_min_stat = min(subset_stat_list)
					lowest_team_list = [remaining_teams[x] for x in range(0, len(remaining_teams), 1) if subset_stat_list[x] == subset_min_stat]
					if len(lowest_team_list) == 1:
						removed_team = lowest_team_list[0]
				remaining_teams = [x for x in remaining_teams if x != removed_team]
			winner_team = remaining_teams[0]
			if playoff_seed == 1:
				season_data[winner_team]['WonConference'] = True
			season_data[winner_team]['PlayoffSeed'] = playoff_seed
			subset_teams = [x for x in subset_teams if x != winner_team]
			playoff_seed += 1

		for wild_card_num in range(0, 3, 1):
			subset_teams = [x for x in list(season_data.keys()) if (cur_conference == teamid_conferences[x]) and (not season_data[x]['MadePlayoffs'])]
			win_pct_list = [season_data[x]['Win%'] for x in subset_teams]
			max_win_pct = max(win_pct_list)
			remaining_teams = [x for x in subset_teams if season_data[x]['Win%'] == max_win_pct]
			while len(remaining_teams) >= 2:
				removed_team = None
				if len(remaining_teams) == 2:
					# Head to head winning percentage and point differential
					if season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[1]
					elif season_data[remaining_teams[0]]['WinsHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['WinsHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[0]
					elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] > season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[1]
					elif season_data[remaining_teams[0]]['PointDifferentialHeadToHead'][remaining_teams[1]] < season_data[remaining_teams[1]]['PointDifferentialHeadToHead'][remaining_teams[0]]:
						removed_team = remaining_teams[0]
				# Loop through a number of options to eliminate teams
				team_removal_stats = ['Win%Conference', 'PointDifferentialConference', 'PointDifferential']
				for removal_stat in team_removal_stats:
					if removed_team is None:
						subset_stat_list = [season_data[x][removal_stat] for x in remaining_teams]
						subset_min_stat = min(subset_stat_list)
						lowest_team_list = [x for x in remaining_teams if season_data[x][removal_stat] == subset_min_stat]
						if len(lowest_team_list) == 1:
							removed_team = lowest_team_list[0]
				# Randomly select an eliminated team
				while removed_team is None:
					subset_stat_list = season_rng.uniform(size = len(remaining_teams)).tolist()
					subset_min_stat = min(subset_stat_list)
					lowest_team_list = [remaining_teams[x] for x in range(0, len(remaining_teams), 1) if subset_stat_list[x] == subset_min_stat]
					if len(lowest_team_list) == 1:
						removed_team = lowest_team_list[0]
				remaining_teams = [x for x in remaining_teams if x != removed_team]
			winner_team = remaining_teams[0]
			season_data[winner_team]['WildCard'] = True
			season_data[winner_team]['MadePlayoffs'] = True
			season_data[winner_team]['PlayoffSeed'] = int(round(len(divisions_list) / 2)) + 1 + wild_card_num

	# Simulate the playoffs... uh, don't talk about playoffs... you kiddin' me? ... playoffs?  I just hope we can win a simulated game, another game...
	superbowl_teams = []
	playoff_bracket = {}
	# This must be sorted for repeatability
	for cur_conference in sorted(conferences_list):
		playoff_bracket[cur_conference] = []
		for playoff_seed in range(1, 8, 1):
			for teamid in teamid_list:
				if (teamid_conferences[teamid] == cur_conference) and (season_data[teamid]['PlayoffSeed'] == playoff_seed):
					playoff_bracket[cur_conference].append(teamid)
		# Wild card round
		losing_teams = []
		game_list = [[7, 2], [6, 3], [5, 4]]
		season_data[playoff_bracket[cur_conference][0]]['MadeDivisionRound'] = True
		for game_seeds in game_list:
			away_teamid = playoff_bracket[cur_conference][game_seeds[0] - 1]
			home_teamid = playoff_bracket[cur_conference][game_seeds[1] - 1]
			winning_teamid, losing_teamid = simulate_game(input_data['TeamRatings'][home_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][home_teamid]['DefenseRatingList'][rating_id], home_teamid, input_data['TeamRatings'][away_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][away_teamid]['DefenseRatingList'][rating_id], away_teamid, points_mean, points_stdev, points_df, points_use_norm_dist, home_advantage, False, season_rng)
			season_data[winning_teamid]['MadeDivisionRound'] = True
			losing_teams.append(losing_teamid)
		playoff_bracket[cur_conference] = [x for x in playoff_bracket[cur_conference] if x not in losing_teams]
		# Division round
		losing_teams = []
		game_list = [[4, 1], [3, 2]]
		for game_seeds in game_list:
			away_teamid = playoff_bracket[cur_conference][game_seeds[0] - 1]
			home_teamid = playoff_bracket[cur_conference][game_seeds[1] - 1]
			winning_teamid, losing_teamid = simulate_game(input_data['TeamRatings'][home_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][home_teamid]['DefenseRatingList'][rating_id], home_teamid, input_data['TeamRatings'][away_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][away_teamid]['DefenseRatingList'][rating_id], away_teamid, points_mean, points_stdev, points_df, points_use_norm_dist, home_advantage, False, season_rng)
			season_data[winning_teamid]['MadeConferenceFinal'] = True
			losing_teams.append(losing_teamid)
		playoff_bracket[cur_conference] = [x for x in playoff_bracket[cur_conference] if x not in losing_teams]
		# Conference championship
		away_teamid = playoff_bracket[cur_conference][0]
		home_teamid = playoff_bracket[cur_conference][1]
		winning_teamid, losing_teamid = simulate_game(input_data['TeamRatings'][home_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][home_teamid]['DefenseRatingList'][rating_id], home_teamid, input_data['TeamRatings'][away_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][away_teamid]['DefenseRatingList'][rating_id], away_teamid, points_mean, points_stdev, points_df, points_use_norm_dist, home_advantage, False, season_rng)
		season_data[winning_teamid]['MadeSuperBowl'] = True
		superbowl_teams.append(winning_teamid)
	# Super Bowl
	away_teamid = superbowl_teams[0]
	home_teamid = superbowl_teams[1]
	winning_teamid, losing_teamid = simulate_game(input_data['TeamRatings'][home_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][home_teamid]['DefenseRatingList'][rating_id], home_teamid, input_data['TeamRatings'][away_teamid]['OffenseRatingList'][rating_id], input_data['TeamRatings'][away_teamid]['DefenseRatingList'][rating_id], away_teamid, points_mean, points_stdev, points_df, points_use_norm_dist, home_advantage, True, season_rng)
	season_data[winning_teamid]['WonSuperBowl'] = True
	return season_data

def main ():
	if (len(sys.argv) < 5):
		print('Usage: '+sys.argv[0]+' <input JSON file> <season> <number of parallel jobs> <number of simulations>')
		exit()

	input_file_name = sys.argv[1].strip()
	try:
		prediction_season = int(sys.argv[2].strip())
		parallel_processes = int(sys.argv[3].strip())
		n_simulations = int(sys.argv[4].strip())
	except:
		print('Invalid numerical parameter')
		sys.exit(1)

	# Load the input data
	input_handle = open(input_file_name, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	input_data = json.load(input_handle)
	input_handle.close()

	# Simulate the seasons in parallel
	season_data_array = joblib.Parallel(n_jobs = parallel_processes)(joblib.delayed(simulate_season)(seasonid, prediction_season, input_data) for seasonid in tqdm.tqdm(range(0, n_simulations, 1)))

	# Calculate totals from the simulations
	teamid_list = list(input_data['TeamRatings'].keys())
	team_stats = {}
	for teamid in teamid_list:
		team_stats[teamid] = {'Wins': 0, 'Losses': 0, 'Ties': 0, 'PointsFor': 0, 'PointsAgainst': 0, 'MaxPointsFor': 0, 'MinPointsFor': 0, 'MaxPointsAgainst': 0, 'MinPointsAgainst': 0, 'WinningSeasons': 0, 'MaxWins': 0, 'MaxLosses': 0, 'MaxTies': 0, 'MaxWin%': 0, 'MinWin%': 1, 'ConferenceWins': 0, 'DivisionWins': 0, 'WildCards': 0, 'Playoffs': 0, 'TotalPlayoffSeeds': 0, 'MeanWins': 0, 'MeanLosses': 0, 'MeanTies': 0, 'MeanPointsFor': 0, 'MeanPointsAgainst': 0, 'MeanWin%': 0, 'ConferenceWin%': 0, 'DivisionWin%': 0, 'WildCard%': 0, 'Playoff%': 0, 'MeanPlayoffSeed': None, 'BestPlayoffSeed': None, 'WinningSeason%': 0, 'MaxPointsDifferential': 0, 'MinPointsDifferential': 0, 'Win%List': [], '10%ileWin%': 0, '25%ileWin%': 0, '50%ileWin%': 0, '75%ileWin%': 0, '90%ileWin%': 0, 'MakeDivisionRoundCount': 0, 'MakeConferenceFinalCount': 0, 'MakeSuperBowlCount': 0, 'WinSuperBowlCount': 0, 'MakeDivisionRound%': 0, 'MakeConferenceFinal%': 0, 'MakeSuperBowl%': 0, 'WinSuperBowl%': 0}
	first_season = True
	# Loop through each simulated season
	for season_data in season_data_array:
		for teamid in teamid_list:
			# Calculate wins, losses, ties, scoring differential, and things like that
			team_stats[teamid]['Wins'] += season_data[teamid]['Wins']
			team_stats[teamid]['MaxWins'] = max(season_data[teamid]['Wins'], team_stats[teamid]['MaxWins'])
			team_stats[teamid]['Losses'] += season_data[teamid]['Losses']
			team_stats[teamid]['MaxLosses'] = max(season_data[teamid]['Losses'], team_stats[teamid]['MaxLosses'])
			team_stats[teamid]['Ties'] += season_data[teamid]['Ties']
			team_stats[teamid]['MaxTies'] = max(season_data[teamid]['Ties'], team_stats[teamid]['MaxTies'])
			team_stats[teamid]['PointsFor'] += season_data[teamid]['PointsFor']
			team_stats[teamid]['PointsAgainst'] += season_data[teamid]['PointsAgainst']
			# Track the maximum and minimum values of some other statistics like points and winning percentage, and just store the values if it's the first season
			if first_season:
				team_stats[teamid]['MaxWin%'] = season_data[teamid]['Win%']
				team_stats[teamid]['MinWin%'] = season_data[teamid]['Win%']
				team_stats[teamid]['MaxPointsFor'] = season_data[teamid]['PointsFor']
				team_stats[teamid]['MinPointsFor'] = season_data[teamid]['PointsFor']
				team_stats[teamid]['MaxPointsAgainst'] = season_data[teamid]['PointsAgainst']
				team_stats[teamid]['MinPointsAgainst'] = season_data[teamid]['PointsAgainst']
				team_stats[teamid]['MaxPointsDifferential'] = season_data[teamid]['PointsFor'] - season_data[teamid]['PointsAgainst']
				team_stats[teamid]['MinPointsDifferential'] = season_data[teamid]['PointsFor'] - season_data[teamid]['PointsAgainst']
			# Compare maximum and minimum values to what's already stored, and update if needed
			team_stats[teamid]['MaxWin%'] = max(season_data[teamid]['Win%'], team_stats[teamid]['MaxWin%'])
			team_stats[teamid]['MinWin%'] = min(season_data[teamid]['Win%'], team_stats[teamid]['MinWin%'])
			team_stats[teamid]['MaxPointsFor'] = max(season_data[teamid]['PointsFor'], team_stats[teamid]['MaxPointsFor'])
			team_stats[teamid]['MinPointsFor'] = min(season_data[teamid]['PointsFor'], team_stats[teamid]['MinPointsFor'])
			team_stats[teamid]['MaxPointsAgainst'] = max(season_data[teamid]['PointsAgainst'], team_stats[teamid]['MaxPointsAgainst'])
			team_stats[teamid]['MinPointsAgainst'] = min(season_data[teamid]['PointsAgainst'], team_stats[teamid]['MinPointsAgainst'])
			team_stats[teamid]['MaxPointsDifferential'] = max(season_data[teamid]['PointsFor'] - season_data[teamid]['PointsAgainst'], team_stats[teamid]['MaxPointsDifferential'])
			team_stats[teamid]['MinPointsDifferential'] = min(season_data[teamid]['PointsFor'] - season_data[teamid]['PointsAgainst'], team_stats[teamid]['MinPointsDifferential'])
			team_stats[teamid]['Win%List'].append(season_data[teamid]['Win%'])
			# Track data about playoff status, winning the division, and winning the conference
			if season_data[teamid]['Win%'] > 0.5:
				team_stats[teamid]['WinningSeasons'] += 1
			if season_data[teamid]['WonConference']:
				team_stats[teamid]['ConferenceWins'] += 1
			if season_data[teamid]['WonDivision']:
				team_stats[teamid]['DivisionWins'] += 1
			if season_data[teamid]['WildCard']:
				team_stats[teamid]['WildCards'] += 1
			if season_data[teamid]['MadePlayoffs']:
				if team_stats[teamid]['Playoffs'] == 0:
					team_stats[teamid]['BestPlayoffSeed'] = season_data[teamid]['PlayoffSeed']
				else:
					team_stats[teamid]['BestPlayoffSeed'] = min(season_data[teamid]['PlayoffSeed'], team_stats[teamid]['BestPlayoffSeed'])
				team_stats[teamid]['Playoffs'] += 1
				team_stats[teamid]['TotalPlayoffSeeds'] += season_data[teamid]['PlayoffSeed']
			# Track playoff progress and update arrays accordingly
			if season_data[teamid]['MadeDivisionRound']:
				team_stats[teamid]['MakeDivisionRoundCount'] += 1
			if season_data[teamid]['MadeConferenceFinal']:
				team_stats[teamid]['MakeConferenceFinalCount'] += 1
			if season_data[teamid]['MadeSuperBowl']:
				team_stats[teamid]['MakeSuperBowlCount'] += 1
			if season_data[teamid]['WonSuperBowl']:
				team_stats[teamid]['WinSuperBowlCount'] += 1
		first_season = False

	# Calculate team averages
	for teamid in teamid_list:
		team_stats[teamid]['MeanWins'] = team_stats[teamid]['Wins'] / n_simulations
		team_stats[teamid]['MeanLosses'] = team_stats[teamid]['Losses'] / n_simulations
		team_stats[teamid]['MeanTies'] = team_stats[teamid]['Ties'] / n_simulations
		team_stats[teamid]['MeanPointsFor'] = team_stats[teamid]['PointsFor'] / n_simulations
		team_stats[teamid]['MeanPointsAgainst'] = team_stats[teamid]['PointsAgainst'] / n_simulations
		team_stats[teamid]['MeanWin%'] = (team_stats[teamid]['Wins'] + (team_stats[teamid]['Ties'] / 2)) / (team_stats[teamid]['Wins'] + team_stats[teamid]['Losses'] + team_stats[teamid]['Ties'])
		team_stats[teamid]['ConferenceWin%'] = team_stats[teamid]['ConferenceWins'] / n_simulations
		team_stats[teamid]['DivisionWin%'] = team_stats[teamid]['DivisionWins'] / n_simulations
		team_stats[teamid]['WildCard%'] = team_stats[teamid]['WildCards'] / n_simulations
		team_stats[teamid]['Playoff%'] = team_stats[teamid]['Playoffs'] / n_simulations
		if team_stats[teamid]['Playoffs'] == 0:
			team_stats[teamid]['MeanPlayoffSeed'] = None
		else:
			team_stats[teamid]['MeanPlayoffSeed'] = team_stats[teamid]['TotalPlayoffSeeds'] / team_stats[teamid]['Playoffs']
		team_stats[teamid]['WinningSeason%'] = team_stats[teamid]['WinningSeasons'] / n_simulations
		team_stats[teamid]['10%ileWin%'] = stats.scoreatpercentile(team_stats[teamid]['Win%List'], 10)
		team_stats[teamid]['25%ileWin%'] = stats.scoreatpercentile(team_stats[teamid]['Win%List'], 25)
		team_stats[teamid]['50%ileWin%'] = stats.scoreatpercentile(team_stats[teamid]['Win%List'], 50)
		team_stats[teamid]['75%ileWin%'] = stats.scoreatpercentile(team_stats[teamid]['Win%List'], 75)
		team_stats[teamid]['90%ileWin%'] = stats.scoreatpercentile(team_stats[teamid]['Win%List'], 90)
		team_stats[teamid]['MakeDivisionRound%'] = team_stats[teamid]['MakeDivisionRoundCount'] / n_simulations
		team_stats[teamid]['MakeConferenceFinal%'] = team_stats[teamid]['MakeConferenceFinalCount'] / n_simulations
		team_stats[teamid]['MakeSuperBowl%'] = team_stats[teamid]['MakeSuperBowlCount'] / n_simulations
		team_stats[teamid]['WinSuperBowl%'] = team_stats[teamid]['WinSuperBowlCount'] / n_simulations

	conferences_list = sorted(list(set([input_data['TeamRatings'][x]['Conference'] for x in teamid_list])))
	divisions_list = list(set([input_data['TeamRatings'][x]['Division'] for x in teamid_list]))
	conference_divisions_list = {}
	for cur_conference in conferences_list:
		conference_divisions_list[cur_conference] = sorted(list(set([input_data['TeamRatings'][x]['Division'] for x in teamid_list if input_data['TeamRatings'][x]['Conference'] == cur_conference])))

	max_team_name_len = max([len(input_data['TeamRatings'][x]['Name']) for x in teamid_list])

	# Sort the teams within their divisions
	division_stats_list = {}
	for cur_division in divisions_list:
		rank_stats_list = []
		for teamid in [x for x in teamid_list if input_data['TeamRatings'][x]['Division'] == cur_division]:
			rank_stats_row = [teamid, team_stats[teamid]['MeanWin%'], team_stats[teamid]['MeanWins'], team_stats[teamid]['MeanLosses'], team_stats[teamid]['MeanTies'], team_stats[teamid]['MeanPointsFor'], team_stats[teamid]['MeanPointsAgainst'], team_stats[teamid]['ConferenceWin%'], team_stats[teamid]['DivisionWin%'], team_stats[teamid]['WildCard%'], team_stats[teamid]['Playoff%'], team_stats[teamid]['MaxWin%'], team_stats[teamid]['MinWin%'], team_stats[teamid]['MeanPlayoffSeed'], team_stats[teamid]['BestPlayoffSeed'], team_stats[teamid]['MaxPointsDifferential'], team_stats[teamid]['MinPointsDifferential'], team_stats[teamid]['WinningSeason%'], team_stats[teamid]['10%ileWin%'], team_stats[teamid]['25%ileWin%'], team_stats[teamid]['50%ileWin%'], team_stats[teamid]['75%ileWin%'], team_stats[teamid]['90%ileWin%'], input_data['TeamRatings'][teamid]['Rating'], team_stats[teamid]['MakeDivisionRound%'], team_stats[teamid]['MakeConferenceFinal%'], team_stats[teamid]['MakeSuperBowl%'], team_stats[teamid]['WinSuperBowl%']]
			rank_stats_list.append(rank_stats_row)
		sorted_rank_stats = sorted(rank_stats_list, key = lambda x: x[10], reverse = True)
		division_stats_list[cur_division] = sorted_rank_stats

	# Write simulated standings in a table
	for cur_conference in conferences_list:
		for cur_division in conference_divisions_list[cur_conference]:
			standings_text = []
			standings_header = ['', 'W', 'L', 'T', 'Win%', 'PF', 'PA', 'Rating']
			standings_alignment = ['', '', '', '', '', '', '', '']
			standings_min_width = [max_team_name_len, 5, 5, 5, 5, 7, 7, 6]
			table_title = cur_division
			cur_line = 0
			for team_data in division_stats_list[cur_division]:
				if cur_line == 0:
					standings_text.append(standings_header)
				standings_line = []
				standings_line.append(input_data['TeamRatings'][team_data[0]]['Name'])
				standings_line.append('{0:.2f}'.format(float(team_data[2])))
				standings_line.append('{0:.2f}'.format(float(team_data[3])))
				standings_line.append('{0:.2f}'.format(float(team_data[4])))
				if float(team_data[1]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[1]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[1]), 3))[1:]))
				standings_line.append('{0:.2f}'.format(float(team_data[5])))
				standings_line.append('{0:.2f}'.format(float(team_data[6])))
				standings_line.append('{0:.2f}'.format(float(team_data[23])))
				standings_text.append(standings_line)
				cur_line = cur_line + 1
			standings_column_width = []
			for cur_column in range(0, len(standings_header), 1):
				standings_column_width.append(max(max([len(x[cur_column]) for x in standings_text]), standings_min_width[cur_column]))
			# Print the table
			print(table_title)
			for cur_line in range(0, len(standings_text), 1):
				cur_line_text = ''
				for cur_column in range(0, len(standings_header), 1):
					if cur_column > 0:
						cur_line_text += ' '
					cur_line_text += ('{:' + standings_alignment[cur_column] + str(int(standings_column_width[cur_column])) + 's}').format(standings_text[cur_line][cur_column])
				print(cur_line_text)
			print('')

	print('')
	print('')

	# Write playoff probabilities in a table
	for cur_conference in conferences_list:
		for cur_division in conference_divisions_list[cur_conference]:
			standings_text = []
			standings_header = ['', 'Rating', 'Win%', 'Playoff%', 'Div%', 'Conf%', 'MeanSeed']
			standings_alignment = ['', '', '>', '>', '>', '>', '>']
			standings_min_width = [max_team_name_len, 6, 5, 7, 7, 7, 4]
			table_title = cur_division
			cur_line = 0
			for team_data in division_stats_list[cur_division]:
				if cur_line == 0:
					standings_text.append(standings_header)
				standings_line = []
				standings_line.append(input_data['TeamRatings'][team_data[0]]['Name'])
				standings_line.append('{0:.2f}'.format(float(team_data[23])))
				if float(team_data[1]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[1]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[1]), 3))[1:]))
				standings_line.append('{0:.2f}'.format(float(team_data[10]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[8]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[7]) * 100) + '%')
				if (team_data[13] is None) or (int(team_data[13]) == 0):
					standings_line.append('-.--')
				else:
					standings_line.append('{0:.2f}'.format(float(team_data[13])))
				standings_text.append(standings_line)
				cur_line = cur_line + 1
			standings_column_width = []
			for cur_column in range(0, len(standings_header), 1):
				standings_column_width.append(max(max([len(x[cur_column]) for x in standings_text]), standings_min_width[cur_column]))
			# Print the table
			print(table_title)
			for cur_line in range(0, len(standings_text), 1):
				cur_line_text = ''
				for cur_column in range(0, len(standings_header), 1):
					if cur_column > 0:
						cur_line_text += ' '
					cur_line_text += ('{:' + standings_alignment[cur_column] + str(int(standings_column_width[cur_column])) + 's}').format(standings_text[cur_line][cur_column])
				print(cur_line_text)
			print('')

	print('')
	print('')

	# Write percentiles of each team's winning percentage in a table
	for cur_conference in conferences_list:
		for cur_division in conference_divisions_list[cur_conference]:
			standings_text = []
			standings_header = ['', 'Win%', '>.500%', '10%ile', '25%ile', '50%ile', '75%ile', '90%ile']
			standings_alignment = ['', '', '', '>', '>', '>', '>', '>']
			standings_min_width = [max_team_name_len, 5, 7, 5, 5, 5, 5, 5]
			table_title = cur_division
			cur_line = 0
			for team_data in division_stats_list[cur_division]:
				if cur_line == 0:
					standings_text.append(standings_header)
				standings_line = []
				standings_line.append(input_data['TeamRatings'][team_data[0]]['Name'])
				if float(team_data[1]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[1]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[1]), 3))[1:]))
				standings_line.append('{0:.2f}'.format(float(team_data[17]) * 100) + '%')
				if float(team_data[18]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[18]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[18]), 3))[1:]))
				if float(team_data[19]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[19]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[19]), 3))[1:]))
				if float(team_data[20]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[20]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[20]), 3))[1:]))
				if float(team_data[21]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[21]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[21]), 3))[1:]))
				if float(team_data[22]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[22]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[22]), 3))[1:]))
				standings_text.append(standings_line)
				cur_line = cur_line + 1
			standings_column_width = []
			for cur_column in range(0, len(standings_header), 1):
				standings_column_width.append(max(max([len(x[cur_column]) for x in standings_text]), standings_min_width[cur_column]))
			# Print the table
			print(table_title)
			for cur_line in range(0, len(standings_text), 1):
				cur_line_text = ''
				for cur_column in range(0, len(standings_header), 1):
					if cur_column > 0:
						cur_line_text += ' '
					cur_line_text += ('{:' + standings_alignment[cur_column] + str(int(standings_column_width[cur_column])) + 's}').format(standings_text[cur_line][cur_column])
				print(cur_line_text)
			print('')

	print('')
	print('')

	# Write simulated playoff statistics in a table
	for cur_conference in conferences_list:
		for cur_division in conference_divisions_list[cur_conference]:
			standings_text = []
			standings_header = ['', 'Win%', 'Playoff%', 'MkDivRd%', 'WinDivRd%', 'WinConf%', 'WinSB%']
			standings_alignment = ['', '', '>', '>', '>', '>', '>']
			standings_min_width = [max_team_name_len, 5, 6, 6, 6, 6, 6]
			table_title = cur_division
			cur_line = 0
			for team_data in division_stats_list[cur_division]:
				if cur_line == 0:
					standings_text.append(standings_header)
				standings_line = []
				standings_line.append(input_data['TeamRatings'][team_data[0]]['Name'])
				if float(team_data[1]) == 1.0:
					standings_line.append('1.000')
				elif float(team_data[1]) == 0.0:
					standings_line.append('.000')
				else:
					standings_line.append(('{0:.3f}'.format(round(float(team_data[1]), 3))[1:]))
				standings_line.append('{0:.2f}'.format(float(team_data[10]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[24]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[25]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[26]) * 100) + '%')
				standings_line.append('{0:.2f}'.format(float(team_data[27]) * 100) + '%')
				
				standings_text.append(standings_line)
				cur_line = cur_line + 1
			standings_column_width = []
			for cur_column in range(0, len(standings_header), 1):
				standings_column_width.append(max(max([len(x[cur_column]) for x in standings_text]), standings_min_width[cur_column]))
			# Print the table
			print(table_title)
			for cur_line in range(0, len(standings_text), 1):
				cur_line_text = ''
				for cur_column in range(0, len(standings_header), 1):
					if cur_column > 0:
						cur_line_text += ' '
					cur_line_text += ('{:' + standings_alignment[cur_column] + str(int(standings_column_width[cur_column])) + 's}').format(standings_text[cur_line][cur_column])
				print(cur_line_text)
			print('')

if __name__ == '__main__':
	main()
