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

import datetime
import json
import math
import numpy as np
import re
import scipy.stats as stats
import sys

def main ():
	if (len(sys.argv) < 11):
		print('Usage: '+sys.argv[0]+' <input JSON file> <use division Y or N> <division> <header frequency> <season> <cutoff type, W = week, N = week name, D = date> <week (YYYY-MM-DD) or date cutoff> <points string> <points abbreviation> <rating decimal places> [previous JSON file]')
		exit()

	# This is intended to postprocess both NFL and college football data, but some tables may not be applicable for one or the other; read in the parameters
	input_file_name = sys.argv[1].strip()
	division_id = sys.argv[3].strip()
	if len(sys.argv[2].strip()) == 0:
		print('Invalid division setting')
		sys.exit(1)
	division_setting_str = (sys.argv[2].strip())[0].lower()
	if division_setting_str == 'n':
		division_id = None
	table_header_frequency = int(sys.argv[4].strip())
	prediction_season = int(sys.argv[5].strip())
	if len(sys.argv[6].strip()) > 0:
		cutoff_letter = (sys.argv[6].strip())[0].lower()
		if cutoff_letter == 'w':
			cutoff_type = 0
		elif cutoff_letter == 'n':
			cutoff_type = 1
		elif cutoff_letter == 'd':
			cutoff_type = 2
		else:
			cutoff_type = None
	else:
		cutoff_type = None
	if cutoff_type is None:
		print('Invalid cutoff type parameter')
		sys.exit(1)
	if cutoff_type == 0:
		prediction_week = int(sys.argv[7].strip())
	elif cutoff_type == 1:
		prediction_week_str = sys.argv[7].strip()
	elif cutoff_type == 2:
		cutoff_date_parse = datetime.datetime.strptime(sys.argv[7].strip(), '%Y-%m-%d')
		cutoff_year = cutoff_date_parse.year
		cutoff_month = cutoff_date_parse.month
		cutoff_day = cutoff_date_parse.day
		cutoff_date = datetime.date(cutoff_year, cutoff_month, cutoff_day)

	points_string = sys.argv[8].strip()
	points_abbrev = sys.argv[9].strip()
	rating_decimal_places = max(int(sys.argv[10].strip()), 0)

	if (len(sys.argv) > 11) and (len(sys.argv[11].strip()) > 0):
		previous_file_name = sys.argv[11].strip()
	else:
		previous_file_name = None

	# Load the rating data
	input_handle = open(input_file_name, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	input_data = json.load(input_handle)
	input_handle.close()
	col_prefix_list = ['Current']
	data_list = [input_data]

	# If there's a previous week of data, load it in
	if previous_file_name is not None:
		previous_handle = open(previous_file_name, 'r')
		if previous_handle is None:
			print('Could not open input file')
			exit()
		previous_data = json.load(previous_handle)
		previous_handle.close()
		col_prefix_list = col_prefix_list + ['Prev']
		data_list = data_list + [previous_data]

	# This is structured to potentially handle multiple weeks of data, looping through the data while reading a few extra columns from the initial (most recent) file
	team_rating_list = []
	team_rating_cols = []
	for dset_idx in range(0, len(data_list), 1):
		col_prefix = col_prefix_list[dset_idx]
		dset_data = data_list[dset_idx]

		# Get a list of teams
		teamid_list = list(dset_data['TeamRatings'].keys())

		# Read in a list of each team's rating data, with a few extra columns for the first input file
		if dset_idx == 0:
			team_rating_cols = team_rating_cols + ['TeamID', 'TeamName', 'Division', 'Conference', col_prefix + 'Rating', col_prefix + 'RatingStDev', col_prefix + 'OffenseRating', col_prefix + 'OffenseRatingStDev', col_prefix + 'DefenseRating', col_prefix + 'DefenseRatingStDev', col_prefix + 'RatingRank']
			for teamid in teamid_list:
				if (dset_data['TeamRatings'][teamid]['Division'] == division_id) or (division_id is None):
					team_rating_list.append([teamid, dset_data['TeamRatings'][teamid]['Name'], dset_data['TeamRatings'][teamid]['Division'], dset_data['TeamRatings'][teamid]['Conference'], dset_data['TeamRatings'][teamid]['Rating'], dset_data['TeamRatings'][teamid]['RatingStDev'], dset_data['TeamRatings'][teamid]['OffenseRating'], dset_data['TeamRatings'][teamid]['OffenseRatingStDev'], dset_data['TeamRatings'][teamid]['DefenseRating'], dset_data['TeamRatings'][teamid]['DefenseRatingStDev']])
			# Only get team IDs and conferences if it's the first iteration of the loop
			div_teamid_list = [x[team_rating_cols.index('TeamID')] for x in team_rating_list]
			div_conference_list = [y for y in list(set([x[team_rating_cols.index('Conference')] for x in team_rating_list])) if y is not None]
			conf_rating_cols = ['Conference', 'TeamCount']
			conf_rating_list = []
			for conf_name in sorted(div_conference_list):
				conf_team_count = len([x for x in team_rating_list if x[team_rating_cols.index('Conference')] == conf_name])
				conf_rating_list.append([conf_name, conf_team_count])
		# Otherwise, load in the data, but skip columns like the team ID, name, division, and conference, because we should already know those in nearly every situation
		else:
			new_cols = [col_prefix + 'Rating', col_prefix + 'RatingStDev', col_prefix + 'OffenseRating', col_prefix + 'OffenseRatingStDev', col_prefix + 'DefenseRating', col_prefix + 'DefenseRatingStDev', col_prefix + 'RatingRank']
			team_rating_cols = team_rating_cols + new_cols
			for row_idx in range(0, len(team_rating_list), 1):
				row_data = team_rating_list[row_idx]
				teamid = row_data[team_rating_cols.index('TeamID')]
				if list(dset_data['TeamRatings'].keys()).count(teamid) > 0:
					row_data = row_data + [dset_data['TeamRatings'][teamid]['Rating'], dset_data['TeamRatings'][teamid]['RatingStDev'], dset_data['TeamRatings'][teamid]['OffenseRating'], dset_data['TeamRatings'][teamid]['OffenseRatingStDev'], dset_data['TeamRatings'][teamid]['DefenseRating'], dset_data['TeamRatings'][teamid]['DefenseRatingStDev']]
				else:
					row_null_data = [math.nan] * len(new_cols)
					row_data = row_data + row_null_data
				team_rating_list[row_idx] = row_data

		# Append rating ranks as needed
		rating_ranks = stats.rankdata(np.array([-x[team_rating_cols.index(col_prefix + 'Rating')] for x in team_rating_list]), method = 'min', nan_policy = 'omit').tolist()
		for cur_idx in range(0, len(rating_ranks), 1):
			rating_list_data = team_rating_list[cur_idx]
			rating_list_data = rating_list_data + [rating_ranks[cur_idx]]
			team_rating_list[cur_idx] = rating_list_data

		# Calculate some statistics of the overall ratings for use in calculating schedule strength, specifically strength of record and strength of schedule
		new_cols1 = [col_prefix + 'TeamSOS', col_prefix + 'TeamSOR', col_prefix + 'LowSOS', col_prefix + 'LowSOR', col_prefix + 'MidSOS', col_prefix + 'MidSOR', col_prefix + 'HighSOS', col_prefix + 'HighSOR', col_prefix + 'TeamRatingNorm', col_prefix + 'Win%', col_prefix + 'MeanOpponentRating']
		new_cols2 = [col_prefix + 'FutureTeamSOS', col_prefix + 'FutureLowSOS', col_prefix + 'FutureMidSOS', col_prefix + 'FutureHighSOS', col_prefix + 'FutureMeanOpponentRating']
		team_rating_cols = team_rating_cols + new_cols1 + new_cols2
		all_div_ratings = [x[team_rating_cols.index(col_prefix + 'Rating')] for x in team_rating_list]
		rating_mean = np.mean(np.array(all_div_ratings))
		rating_stdev = np.std(np.array(all_div_ratings))
		for teamid in div_teamid_list:
			# If we don't have data for the team for some reason, and this is possible in unusual circumstances, set the value to NaN; otherwise, get the team's rating and set up the hypothetical opponents for its schedule
			if list(dset_data['TeamRatings'].keys()).count(teamid) > 0:
				team_is_valid = True
			else:
				team_is_valid = False
			if team_is_valid:
				team_past_games = [x for x in dset_data['TeamRatings'][teamid]['PastSchedule'] if (x['Season'] == prediction_season) and (not x['IsPreseason'])]
				team_future_games = [x for x in dset_data['TeamRatings'][teamid]['FutureSchedule'] if (x['Season'] == prediction_season) and (not x['IsPreseason'])]
				rating_opts = [dset_data['TeamRatings'][teamid]['Rating'], rating_mean - (rating_stdev * 1.5), rating_mean, rating_mean + (rating_stdev * 1.5)]
				team_rating_strength = stats.norm.cdf(dset_data['TeamRatings'][teamid]['Rating'], loc = rating_mean, scale = rating_stdev)
			else:
				team_rating_strength = math.nan
			# Calculate strength of schedule and strength of record for past games
			if team_is_valid and (len(team_past_games) > 0):
				cur_season_winprob_sor = []
				sum_wpct = 0
				sum_opp_rating = 0
				for cur_game in team_past_games:
					# Determine the actual outcome of the game to calculate the actual winning percentage
					if cur_game['TeamScore'] > cur_game['OpponentScore']:
						cur_game_wpct = 1.000
					elif cur_game['OpponentScore'] > cur_game['TeamScore']:
						cur_game_wpct = 0.000
					else:
						cur_game_wpct = 0.500
					sum_wpct += cur_game_wpct
					# Calculate the correct home advantage and get the opponent's effective rating (accounting for home advantage)
					cur_game_winprob_sor = []
					if cur_game['IsHomeGame']:
						game_home_advantage = dset_data['HomeAdvantage']
					elif cur_game['IsAwayGame']:
						game_home_advantage = -dset_data['HomeAdvantage']
					else:
						game_home_advantage = 0
					sum_opp_rating += cur_game['OpponentRating'] - game_home_advantage
					# Loop through the hypothetical teams, predict the outcome, and do strength of record calculations
					for cur_rating_opt in rating_opts:
						game_margin = cur_rating_opt + game_home_advantage - cur_game['OpponentRating']
						if dset_data['IsPredictionErrorNormal']:
							game_opp_prob = stats.norm.cdf(-dset_data['TieCDFBound'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
							game_team_prob = 1 - stats.norm.cdf(dset_data['TieCDFBound'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
						else:
							game_opp_prob = stats.t.cdf(-dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
							game_team_prob = 1 - stats.t.cdf(dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
						if dset_data['TieCDFBound'] > 0:
							game_tie_prob = 1 - (game_opp_prob + game_team_prob)
						else:
							game_tie_prob = 0
						cur_game_winprob_sor.append(game_opp_prob + (game_tie_prob * 0.5))
						cur_game_winprob_sor.append(cur_game_wpct - (game_team_prob + (game_tie_prob * 0.5)))
					cur_season_winprob_sor.append(cur_game_winprob_sor)
				if len(cur_season_winprob_sor) > 0:
					cur_season_sos_sor = np.mean(np.array(cur_season_winprob_sor), axis = 0).tolist()
				else:
					cur_season_sos_sor = [math.nan] * (len(new_cols1) - 3)
			# If this is impossible due to an invalid team, store a lot of NaN values instead
			else:
				cur_season_sos_sor = [math.nan] * (len(new_cols1) - 3)
			rating_list_idx = [x for x in range(0, len(team_rating_list), 1) if team_rating_list[x][team_rating_cols.index('TeamID')] == teamid][0]
			# If possible, calculate the winning percentage and mean opponent rating; otherwise, set to NaN
			if len(team_past_games) > 0:
				team_wpct = sum_wpct / float(len(team_past_games))
				mean_opp_rating = sum_opp_rating / float(len(team_past_games))
			else:
				team_wpct = math.nan
				mean_opp_rating = math.nan
			rating_list_data = team_rating_list[rating_list_idx] + cur_season_sos_sor + [team_rating_strength, team_wpct, mean_opp_rating]
			team_rating_list[rating_list_idx] = rating_list_data
			# Calculate schedule strength for future games, mostly repeating the process for past schedule strength
			sum_opp_rating = 0
			num_opp_rating = 0
			if team_is_valid and (len(team_future_games) > 0):
				cur_season_winprob = []
				for cur_game in team_future_games:
					if cur_game['OpponentRating'] is not None:
						cur_game_winprob = []
						# Determine the appropriate home advantage
						if cur_game['IsHomeGame']:
							game_home_advantage = dset_data['HomeAdvantage']
						elif cur_game['IsAwayGame']:
							game_home_advantage = -dset_data['HomeAdvantage']
						else:
							game_home_advantage = 0
						num_opp_rating += 1
						sum_opp_rating += cur_game['OpponentRating'] - game_home_advantage
						# Loop through the hypothetical opponents and calculate the predicted outcome
						for cur_rating_opt in rating_opts:
							game_margin = cur_rating_opt + game_home_advantage - cur_game['OpponentRating']
							if dset_data['IsPredictionErrorNormal']:
								game_opp_prob = stats.norm.cdf(-dset_data['TieCDFBound'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
								game_team_prob = 1 - stats.norm.cdf(dset_data['TieCDFBound'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
							else:
								game_opp_prob = stats.t.cdf(-dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
								game_team_prob = 1 - stats.t.cdf(dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = game_margin, scale = dset_data['PredictionErrorStDev'])
							if dset_data['TieCDFBound'] > 0:
								game_tie_prob = 1 - (game_opp_prob + game_team_prob)
							else:
								game_tie_prob = 0
							cur_game_winprob.append(game_opp_prob + (game_tie_prob * 0.5))
						cur_season_winprob.append(cur_game_winprob)
				# If we have any past games, calculate averages; otherwise, store NaNs
				if len(cur_season_winprob) > 0:
					cur_season_sos = np.mean(np.array(cur_season_winprob), axis = 0).tolist()
				else:
					cur_season_sos = [math.nan] * (len(new_cols2) - 1)
			else:
				cur_season_sos = [math.nan] * (len(new_cols2) - 1)
			rating_list_idx = [x for x in range(0, len(team_rating_list), 1) if team_rating_list[x][team_rating_cols.index('TeamID')] == teamid][0]
			# Calculate averages for the schedule strength, otherwise store NaNs
			if num_opp_rating > 0:
				mean_opp_rating = sum_opp_rating / float(num_opp_rating)
			else:
				mean_opp_rating = math.nan
			rating_list_data = team_rating_list[rating_list_idx] + cur_season_sos + [mean_opp_rating]
			team_rating_list[rating_list_idx] = rating_list_data

		# Assume a normal distribution for each strength of schedule and strength of record column, then calculate each team's CDF
		new_cols = [col_prefix + x + 'Norm' for x in ['TeamSOS', 'TeamSOR', 'LowSOS', 'LowSOR', 'MidSOS', 'MidSOR', 'HighSOS', 'HighSOR']]
		team_rating_cols = team_rating_cols + new_cols
		for col_idx in [team_rating_cols.index(col_prefix + x) for x in ['TeamSOS', 'TeamSOR', 'LowSOS', 'LowSOR', 'MidSOS', 'MidSOR', 'HighSOS', 'HighSOR']]:
			col_data = [x[col_idx] for x in team_rating_list]
			col_notnan_data = [x for x in col_data if not math.isnan(x)]
			if len(col_notnan_data) > 0:
				col_mean = np.mean(np.array(col_notnan_data))
				col_stdev = np.std(np.array(col_notnan_data))
			else:
				col_mean = math.nan
				col_stdev = math.nan
			col_new_vals = stats.norm.cdf(np.array(col_data), loc = col_mean, scale = col_stdev).tolist()
			for row_idx in range(0, len(col_new_vals), 1):
				rating_list_data = team_rating_list[row_idx]
				rating_list_data = rating_list_data + [col_new_vals[row_idx]]
				team_rating_list[row_idx] = rating_list_data

		# Calculate "overall" ratings, which are the sum of the strength of record ratings and the predictive ratings after applying a normal distribution
		new_cols = [col_prefix + x + 'Ovr' for x in ['Team', 'Low', 'Mid', 'High']]
		team_rating_cols = team_rating_cols + new_cols
		col1_idx = team_rating_cols.index(col_prefix + 'TeamRatingNorm')
		for col_type in ['Team', 'Low', 'Mid', 'High']:
			col2_idx = team_rating_cols.index(col_prefix + col_type + 'SOR')
			for row_idx in range(0, len(col_new_vals), 1):
				rating_list_data = team_rating_list[row_idx]
				rating_list_data = rating_list_data + [rating_list_data[col1_idx] + rating_list_data[col2_idx]]
				team_rating_list[row_idx] = rating_list_data

		# Fit a normal distribution to the "overall" ratings, then calculate each team's CDF
		new_cols = [col_prefix + x + 'Norm' for x in ['TeamOvr', 'LowOvr', 'MidOvr', 'HighOvr']]
		team_rating_cols = team_rating_cols + new_cols
		for col_idx in [team_rating_cols.index(col_prefix + x) for x in ['TeamOvr', 'LowOvr', 'MidOvr', 'HighOvr']]:
			col_data = [x[col_idx] for x in team_rating_list]
			col_notnan_data = [x for x in col_data if not math.isnan(x)]
			if len(col_notnan_data) > 0:
				col_mean = np.mean(np.array(col_notnan_data))
				col_stdev = np.std(np.array(col_notnan_data))
			else:
				col_mean = math.nan
				col_stdev = math.nan
			col_new_vals = stats.norm.cdf(np.array(col_data), loc = col_mean, scale = col_stdev).tolist()
			for row_idx in range(0, len(col_new_vals), 1):
				rating_list_data = team_rating_list[row_idx]
				rating_list_data = rating_list_data + [col_new_vals[row_idx]]
				team_rating_list[row_idx] = rating_list_data

		# Use the playoff rating format, but calculate it for the four different strength of record and strength of schedule columns
		new_cols = [col_prefix + x + 'Playoff' for x in ['Team', 'Low', 'Mid', 'High']]
		team_rating_cols = team_rating_cols + new_cols
		col1_idx = team_rating_cols.index(col_prefix + 'TeamRatingNorm')
		col1_data = [x[col1_idx] for x in team_rating_list]
		col2_idx = team_rating_cols.index(col_prefix + 'Win%')
		col2_data = [x[col2_idx] for x in team_rating_list]
		for col_type in ['Team', 'Low', 'Mid', 'High']:
			col3_idx = team_rating_cols.index(col_prefix + col_type + 'SORNorm')
			col3_data = [x[col3_idx] for x in team_rating_list]
			col4_idx = team_rating_cols.index(col_prefix + col_type + 'SOSNorm')
			col4_data = [x[col4_idx] for x in team_rating_list]
			col_new_vals = np.add(np.add(np.multiply(np.array(col1_data), 0.3), np.multiply(np.array(col2_data), 0.1)), np.add(np.multiply(np.array(col3_data), 0.55), np.multiply(np.array(col4_data), 0.05))).tolist()
			for row_idx in range(0, len(col_new_vals), 1):
				rating_list_data = team_rating_list[row_idx]
				rating_list_data = rating_list_data + [col_new_vals[row_idx]]
				team_rating_list[row_idx] = rating_list_data

		# Rank each column and store the ranks in new columns
		old_cols = ['TeamSOS', 'TeamSOR', 'LowSOS', 'LowSOR', 'MidSOS', 'MidSOR', 'HighSOS', 'HighSOR', 'TeamRatingNorm', 'Win%', 'MeanOpponentRating'] + ['FutureTeamSOS', 'FutureLowSOS', 'FutureMidSOS', 'FutureHighSOS', 'FutureMeanOpponentRating'] + [x + 'Norm' for x in ['TeamSOS', 'TeamSOR', 'LowSOS', 'LowSOR', 'MidSOS', 'MidSOR', 'HighSOS', 'HighSOR']] + [x + 'Ovr' for x in ['Team', 'Low', 'Mid', 'High']] + [x + 'OvrNorm' for x in ['Team', 'Low', 'Mid', 'High']] + [x + 'Playoff' for x in ['Team', 'Low', 'Mid', 'High']]
		new_cols = [col_prefix + x + 'Rank' for x in old_cols]
		team_rating_cols = team_rating_cols + new_cols
		for col_idx in [team_rating_cols.index(col_prefix + x) for x in old_cols]:
			col_ranks = stats.rankdata(np.array([-x[col_idx] for x in team_rating_list]), method = 'min', nan_policy = 'omit').tolist()
			for row_idx in range(0, len(col_ranks), 1):
				rating_list_data = team_rating_list[row_idx]
				rating_list_data.append(col_ranks[row_idx])
				team_rating_list[row_idx] = rating_list_data

		# Calculate conference statistics
		new_conf_rating_cols = [col_prefix + 'MeanRating', col_prefix + 'MeanOffenseRating', col_prefix + 'MeanDefenseRating', col_prefix + 'DetWin%', col_prefix + 'ExpWin%', col_prefix + 'ExpMargin', col_prefix + 'MeanOffDefRating', col_prefix + 'LowDetWin%', col_prefix + 'LowExpWin%', col_prefix + 'MidDetWin%', col_prefix + 'MidExpWin%', col_prefix + 'HighDetWin%', col_prefix + 'HighExpWin%']
		rating_opts = [rating_mean - (rating_stdev * 1.5), rating_mean, rating_mean + (rating_stdev * 1.5)]
		# Loop through the conferences
		for conf_data_idx in range(0, len(conf_rating_list), 1):
			conf_data_row = conf_rating_list[conf_data_idx]
			conf_name = conf_data_row[conf_rating_cols.index('Conference')]
			rating_sum = 0
			offense_rating_sum = 0
			defense_rating_sum = 0
			margin_sum = 0
			wins_sum = 0
			ties_sum = 0
			wpct_sum = 0
			games_sum = 0
			teams_count = 0
			conf_opt_wpct = []
			# Go through each team in the conference
			for team_row in [x for x in team_rating_list if ((x[team_rating_cols.index('Conference')] == conf_name) and (x[team_rating_cols.index(col_prefix + 'OffenseRating')] is not None) and (x[team_rating_cols.index(col_prefix + 'DefenseRating')] is not None) and (x[team_rating_cols.index(col_prefix + 'Rating')] is not None))]:
				teams_count += 1
				cur_team_rating = team_row[team_rating_cols.index(col_prefix + 'Rating')]
				cur_team_offense_rating = team_row[team_rating_cols.index(col_prefix + 'OffenseRating')]
				cur_team_defense_rating = team_row[team_rating_cols.index(col_prefix + 'DefenseRating')]
				rating_sum += cur_team_rating
				offense_rating_sum += cur_team_offense_rating
				defense_rating_sum += cur_team_defense_rating
				# Loop through each team in the division and treat them as a hypothetical opponent
				for opponent_row in [x for x in team_rating_list if ((x[team_rating_cols.index(col_prefix + 'OffenseRating')] is not None) and (x[team_rating_cols.index(col_prefix + 'DefenseRating')] is not None) and (x[team_rating_cols.index(col_prefix + 'Rating')] is not None))]:
					cur_opponent_rating = opponent_row[team_rating_cols.index(col_prefix + 'Rating')]
					games_sum += 1
					# Predict the scoring margin
					cur_game_margin = cur_team_rating - cur_opponent_rating
					# Estimate the winner using a deterministic approach
					if abs(cur_game_margin) <= dset_data['TieCDFBound']:
						ties_sum += 1
					elif cur_game_margin > 0:
						wins_sum += 1
					# Calculate the probability of winning
					if dset_data['IsPredictionErrorNormal']:
						game_opp_prob = stats.norm.cdf(-dset_data['TieCDFBound'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
						game_team_prob = 1 - stats.norm.cdf(dset_data['TieCDFBound'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
					else:
						game_opp_prob = stats.t.cdf(-dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
						game_team_prob = 1 - stats.t.cdf(dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
					# Allow for the possibility of ties in the probabilistic outcome
					if dset_data['TieCDFBound'] > 0:
						game_tie_prob = 1 - (game_opp_prob + game_team_prob)
					else:
						game_tie_prob = 0
					wpct_sum += (game_team_prob + (game_tie_prob / 2))
					margin_sum += cur_game_margin
				team_opt_wpct = []
				# Loop through each of the three hypothetical opponents for the conference
				for cur_opponent_rating in rating_opts:
					# Estimate the scoring margin
					cur_game_margin = cur_team_rating - cur_opponent_rating
					# Predict a deterministic outcome for the game
					if abs(cur_game_margin) <= dset_data['TieCDFBound']:
						det_val = 0.5
					elif cur_game_margin > 0:
						det_val = 1.0
					else:
						det_val = 0.0
					# Calculate a probabilistic outcome for the game
					if dset_data['IsPredictionErrorNormal']:
						game_opp_prob = stats.norm.cdf(-dset_data['TieCDFBound'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
						game_team_prob = 1 - stats.norm.cdf(dset_data['TieCDFBound'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
					else:
						game_opp_prob = stats.t.cdf(-dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
						game_team_prob = 1 - stats.t.cdf(dset_data['TieCDFBound'], dset_data['PredictionErrorDF'], loc = cur_game_margin, scale = dset_data['PredictionErrorStDev'])
					# Allow for the possibility of a tie in the probabilities
					if dset_data['TieCDFBound'] > 0:
						game_tie_prob = 1 - (game_opp_prob + game_team_prob)
					else:
						game_tie_prob = 0
					exp_val = (game_team_prob + (game_tie_prob / 2))
					team_opt_wpct.extend([det_val, exp_val])
				conf_opt_wpct.append(team_opt_wpct)
			# If there are any games to include in the summary (which should almost always be the case), average the results across the conference
			if games_sum > 0:
				det_wpct = (wins_sum + (ties_sum / 2)) / games_sum
				mean_margin = margin_sum / games_sum
				exp_wpct = wpct_sum / games_sum
				conf_opt_wpct_mean = np.mean(np.array(conf_opt_wpct), axis = 0).tolist()
			else:
				det_wpct = math.nan
				mean_margin = math.nan
				exp_wpct = math.nan
				conf_opt_wpct_mean = [math.nan] * (len(rating_opts) * 2)
			# Calculate the mean of the team ratings for the conference
			if teams_count > 0:
				mean_rating = rating_sum / teams_count
				mean_offense_rating = offense_rating_sum / teams_count
				mean_defense_rating = defense_rating_sum / teams_count
				mean_offdef = mean_offense_rating - mean_defense_rating
			else:
				mean_rating = math.nan
				mean_offense_rating = math.nan
				mean_defense_rating = math.nan
				mean_offdef = math.nan
			conf_data_row = conf_data_row + [mean_rating, mean_offense_rating, mean_defense_rating, det_wpct, exp_wpct, mean_margin, mean_offdef] + conf_opt_wpct_mean
			conf_rating_list[conf_data_idx] = conf_data_row
		conf_rating_cols = conf_rating_cols + new_conf_rating_cols

		# Calculate ranks for the conference statistics and also store those
		old_cols = ['MeanRating', 'MeanOffenseRating', 'MeanDefenseRating', 'DetWin%', 'ExpWin%', 'ExpMargin', 'MeanOffDefRating', 'LowDetWin%', 'LowExpWin%', 'MidDetWin%', 'MidExpWin%', 'HighDetWin%', 'HighExpWin%']
		new_cols = [col_prefix + x + 'Rank' for x in old_cols]
		conf_rating_cols = conf_rating_cols + new_cols
		for col_idx in [conf_rating_cols.index(col_prefix + x) for x in old_cols]:
			col_ranks = stats.rankdata(np.array([-x[col_idx] for x in conf_rating_list]), method = 'min', nan_policy = 'omit').tolist()
			for row_idx in range(0, len(col_ranks), 1):
				rating_list_data = conf_rating_list[row_idx]
				rating_list_data.append(col_ranks[row_idx])
				conf_rating_list[row_idx] = rating_list_data

	# Print a team rating table with ranks, changes from one rating to the next (if applicable), and the offense and defense ratings
	rating_text = []
	if previous_file_name is not None:
		rating_header = ['Rank', 'Move', 'Rating', 'Change', 'Team', 'Offense', 'Defense']
		rating_alignment = ['>', '>', '', '', '', '', '']
	else:
		rating_header = ['Rank', 'Rating', 'Team', 'Offense', 'Defense']
		rating_alignment = ['>', '', '', '', '']
	table_title = 'Predictive Ratings'
	table_subtitles = ['Home advantage: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['HomeAdvantage'], rating_decimal_places)) + ' ' + points_string, 'Mean score: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['ScoreMean'], rating_decimal_places)) + ' ' + points_string]
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		# Only include rank changes if we have prior week data
		if previous_file_name is not None:
			# This is reversed so that a negative number means a team dropped in the ratings and a positive number means a team rose
			if not (math.isnan(team_data[team_rating_cols.index('PrevRatingRank')]) or math.isnan(team_data[team_rating_cols.index('CurrentRatingRank')]) or math.isnan(team_data[team_rating_cols.index('PrevRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentRating')])):
				change_int = team_data[team_rating_cols.index('PrevRatingRank')] - team_data[team_rating_cols.index('CurrentRatingRank')]
				if change_int == 0:
					change_str = ' '
				elif change_int > 0:
					change_str = '+' + '{:d}'.format(int(change_int))
				elif change_int < 0:
					change_str = '-' + '{:d}'.format(int(abs(change_int)))
			else:
				change_str = '---'
			rating_line.append(change_str)
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentRating')])))
		# Only add trends in ratings if we have prior week data
		if previous_file_name is not None:
			if not (math.isnan(team_data[team_rating_cols.index('PrevRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentRating')])):
				change_rating = team_data[team_rating_cols.index('CurrentRating')] - team_data[team_rating_cols.index('PrevRating')]
				if change_rating == 0:
					rating_line.append(' ')
				elif change_rating < 0:
					rating_line.append('-' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(abs(change_rating))))
				else:
					rating_line.append('+' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(abs(change_rating))))
			else:
				rating_line.append('---')
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentOffenseRating')])))
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentDefenseRating')])))
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with ranks, changes from one rating to the next (if applicable), the offense and defense ratings, and strength of record
	rating_text = []
	if previous_file_name is not None:
		rating_header = ['Rank', 'Move', 'Rating', 'Change', 'Team', 'Offense', 'Defense', 'SOR']
		rating_alignment = ['>', '>', '', '', '', '', '', '>']
	else:
		rating_header = ['Rank', 'Rating', 'Team', 'Offense', 'Defense', 'SOR']
		rating_alignment = ['>', '', '', '', '', '>']
	table_title = 'Predictive Ratings'
	table_subtitles = ['Home advantage: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['HomeAdvantage'], rating_decimal_places)) + ' ' + points_string, 'Mean score: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['ScoreMean'], rating_decimal_places)) + ' ' + points_string]
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		# Only include rank changes if we have prior week data
		if previous_file_name is not None:
			# This is reversed so that a negative number means a team dropped in the ratings and a positive number means a team rose
			if not (math.isnan(team_data[team_rating_cols.index('PrevRatingRank')]) or math.isnan(team_data[team_rating_cols.index('CurrentRatingRank')]) or math.isnan(team_data[team_rating_cols.index('PrevRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentRating')])):
				change_int = team_data[team_rating_cols.index('PrevRatingRank')] - team_data[team_rating_cols.index('CurrentRatingRank')]
				if change_int == 0:
					change_str = ' '
				elif change_int > 0:
					change_str = '+' + '{:d}'.format(int(change_int))
				elif change_int < 0:
					change_str = '-' + '{:d}'.format(int(abs(change_int)))
			else:
				change_str = '---'
			rating_line.append(change_str)
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentRating')])))
		# Only add trends in ratings if we have prior week data
		if previous_file_name is not None:
			if not (math.isnan(team_data[team_rating_cols.index('PrevRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentRating')])):
				change_rating = team_data[team_rating_cols.index('CurrentRating')] - team_data[team_rating_cols.index('PrevRating')]
				if change_rating == 0:
					rating_line.append(' ')
				elif change_rating < 0:
					rating_line.append('-' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(abs(change_rating))))
				else:
					rating_line.append('+' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(abs(change_rating))))
			else:
				rating_line.append('---')
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentOffenseRating')])))
		rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentDefenseRating')])))
		if not math.isnan(team_data[team_rating_cols.index('CurrentHighSOR')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSOR')]))))
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with ranks and the various schedule strength measures
	rating_text = []
	rating_header = ['Rank', 'Team', 'TeamSOS', 'LowSOS', 'MidSOS', 'HighSOS']
	rating_alignment = ['>', '', '', '', '', '']
	table_title = 'Schedule Ratings'
	table_subtitles = []
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentRatingRank')]) and math.inf) or x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not (math.isnan(team_data[team_rating_cols.index('CurrentTeamSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentTeamSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentTeamSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentTeamSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentLowSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentLowSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentLowSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentLowSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMidSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentMidSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentMidSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMidSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentHighSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentHighSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentHighSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with past and future schedule strength for an average team, which is probably more appropriate for NFL data
	rating_text = []
	rating_header = ['Rank', 'Team', 'SOS', 'Future', 'OppRtg', 'Future']
	rating_alignment = ['>', '', '', '', '', '']
	table_title = 'Schedule Strength for an Average Team'
	table_subtitles = ['Home advantage: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['HomeAdvantage'], rating_decimal_places)) + ' ' + points_string, 'Mean score: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['ScoreMean'], rating_decimal_places)) + ' ' + points_string]
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentRatingRank')]) and math.inf) or x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMidSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentMidSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentMidSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMidSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentFutureMidSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentFutureMidSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentFutureMidSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentFutureMidSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMeanOpponentRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentMeanOpponentRatingRank')])):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentMeanOpponentRating')])) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMeanOpponentRatingRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRatingRank')])):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRating')])) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRatingRank')])) + ')' )
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with schedule strength, including the schedule strength for a highly-ranked opponent, which is probably more appropriate for college football teams
	rating_text = []
	rating_header = ['Rank', 'Team', 'SOS', 'Future', 'OppRtg', 'Future']
	rating_alignment = ['>', '', '', '', '', '']
	table_title = 'Past and Future Schedule Strength'
	table_subtitles = ['Home advantage: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['HomeAdvantage'], rating_decimal_places)) + ' ' + points_string, 'Mean score: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(round(input_data['ScoreMean'], rating_decimal_places)) + ' ' + points_string]
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentRatingRank')]) and math.inf) or x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not (math.isnan(team_data[team_rating_cols.index('CurrentHighSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentHighSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentHighSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentFutureHighSOS')]) or math.isnan(team_data[team_rating_cols.index('CurrentFutureHighSOSRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentFutureHighSOS')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentFutureHighSOSRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMeanOpponentRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentMeanOpponentRatingRank')])):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentMeanOpponentRating')])) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMeanOpponentRatingRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRating')]) or math.isnan(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRatingRank')])):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRating')])) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentFutureMeanOpponentRatingRank')])) + ')' )
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with ranks and the various strength of record columns
	rating_text = []
	rating_header = ['Rank', 'Team', 'TeamSOR', 'LowSOR', 'MidSOR', 'HighSOR']
	rating_alignment = ['>', '', '', '', '', '']
	table_title = 'Strength of Record Ratings'
	table_subtitles = []
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentRatingRank')]) and math.inf) or x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not (math.isnan(team_data[team_rating_cols.index('CurrentTeamSOR')]) or math.isnan(team_data[team_rating_cols.index('CurrentTeamSORRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentTeamSOR')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentTeamSORRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentLowSOR')]) or math.isnan(team_data[team_rating_cols.index('CurrentLowSORRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentLowSOR')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentLowSORRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMidSOR')]) or math.isnan(team_data[team_rating_cols.index('CurrentMidSORRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentMidSOR')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMidSORRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentHighSOR')]) or math.isnan(team_data[team_rating_cols.index('CurrentHighSORRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSOR')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentHighSORRank')])) + ')' )
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with the various "overall" ratings that combine predictive and strength of record ratings
	rating_text = []
	rating_header = ['Rank', 'Team', 'TeamOvr', 'LowOvr', 'MidOvr', 'HighOvr']
	rating_alignment = ['>', '', '', '', '', '']
	table_title = 'Overall Ratings'
	table_subtitles = []
	cur_line = 0
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentRatingRank')]) and math.inf) or x[team_rating_cols.index('CurrentRatingRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentRatingRank')])))
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not (math.isnan(team_data[team_rating_cols.index('CurrentTeamOvr')]) or math.isnan(team_data[team_rating_cols.index('CurrentTeamOvrRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentTeamOvr')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentTeamOvrRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentLowOvr')]) or math.isnan(team_data[team_rating_cols.index('CurrentLowOvrRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentLowOvr')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentLowOvrRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentMidOvr')]) or math.isnan(team_data[team_rating_cols.index('CurrentMidOvrRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentMidOvr')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentMidOvrRank')])) + ')' )
		else:
			rating_line.append('---')
		if not (math.isnan(team_data[team_rating_cols.index('CurrentHighOvr')]) or math.isnan(team_data[team_rating_cols.index('CurrentHighOvrRank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighOvr')]))) + ' (' + '{:d}'.format(int(team_data[team_rating_cols.index('CurrentHighOvrRank')])) + ')' )
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a team rating table with the playoff ratings
	rating_text = []
	if previous_file_name is not None:
		rating_header = ['Rank', 'Move', 'Rating', 'Change', 'Team', 'SOR', 'SOS', 'Win%', 'Fwd']
		rating_alignment = ['>', '>', '', '', '', '', '', '', '']
	else:
		rating_header = ['Rank', 'Rating', 'Team', 'SOR', 'SOS', 'Win%', 'Fwd']
		rating_alignment = ['>', '', '', '', '', '', '']
	table_title = 'Playoff Ratings'
	table_subtitles = []
	cur_line = 0
	#for team_data in [y for y in sorted(team_rating_list, key = lambda x: x[2], reverse = True)]:
	for team_data in sorted(team_rating_list, key = lambda x: (math.isnan(x[team_rating_cols.index('CurrentHighPlayoffRank')]) and math.inf) or x[team_rating_cols.index('CurrentHighPlayoffRank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		if not math.isnan(team_data[team_rating_cols.index('CurrentHighPlayoffRank')]):
			rating_line.append('{:d}'.format(int(team_data[team_rating_cols.index('CurrentHighPlayoffRank')])))
		else:
			rating_line.append('---')
		# Only include rank changes if we have prior week data
		if previous_file_name is not None:
			# This is reversed so that a negative number means a team dropped in the ratings and a positive number means a team rose
			change_int = team_data[team_rating_cols.index('PrevHighPlayoffRank')] - team_data[team_rating_cols.index('CurrentHighPlayoffRank')]
			if change_int == 0:
				change_str = ' '
			elif change_int > 0:
				change_str = '+' + '{:d}'.format(int(change_int))
			elif change_int < 0:
				change_str = '-' + '{:d}'.format(int(abs(change_int)))
			else:
				change_str = '---'
			rating_line.append(change_str)
		if not math.isnan(team_data[team_rating_cols.index('CurrentHighPlayoff')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.4f}'.format(float(team_data[team_rating_cols.index('CurrentHighPlayoff')]))))
		else:
			rating_line.append('---')
		# Only add trends in ratings if we have prior week data
		if previous_file_name is not None:
			if not (math.isnan(team_data[team_rating_cols.index('CurrentHighPlayoff')]) or math.isnan(team_data[team_rating_cols.index('PrevHighPlayoff')])):
				change_rating = team_data[team_rating_cols.index('CurrentHighPlayoff')] - team_data[team_rating_cols.index('PrevHighPlayoff')]
				if change_rating == 0:
					rating_line.append(' ')
				elif change_rating < 0:
					rating_line.append('-' + re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.4f}'.format(float(abs(change_rating)))))
				else:
					rating_line.append('+' + re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.4f}'.format(float(abs(change_rating)))))
			else:
				rating_line.append('---')
		rating_line.append(str(team_data[team_rating_cols.index('TeamName')]))
		if not math.isnan(team_data[team_rating_cols.index('CurrentHighSORNorm')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSORNorm')]))) + ' ')
		else:
			rating_line.append('---')
		if not math.isnan(team_data[team_rating_cols.index('CurrentHighSOSNorm')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentHighSOSNorm')]))) + ' ')
		else:
			rating_line.append('---')
		if not math.isnan(team_data[team_rating_cols.index('CurrentWin%')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentWin%')]))) + ' ')
		else:
			rating_line.append('---')
		if not math.isnan(team_data[team_rating_cols.index('CurrentTeamRatingNorm')]):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(team_data[team_rating_cols.index('CurrentTeamRatingNorm')]))))
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Print a conference rating table
	rating_text = []
	rating_header = ['Rank', 'Win%', 'Conference', 'HighWin%', 'Rating', 'Offense', 'Defense', 'OffDef']
	rating_alignment = ['>', '>', '', '', '', '', '', '']
	table_title = 'Conference Ratings'
	table_subtitles = []
	cur_line = 0
	for conf_data in sorted(conf_rating_list, key = lambda x: (math.isnan(x[conf_rating_cols.index('CurrentExpWin%Rank')]) and math.inf) or x[conf_rating_cols.index('CurrentExpWin%Rank')]):
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(conf_data[conf_rating_cols.index('CurrentExpWin%Rank')])))
		if not math.isnan(conf_data[conf_rating_cols.index('CurrentExpWin%')]) :
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(conf_data[conf_rating_cols.index('CurrentExpWin%')]))))
		else:
			rating_line.append('---')
		rating_line.append(str(conf_data[conf_rating_cols.index('Conference')]))
		
		if not (math.isnan(conf_data[conf_rating_cols.index('CurrentHighExpWin%')]) or math.isnan(conf_data[conf_rating_cols.index('CurrentHighExpWin%Rank')])):
			rating_line.append(re.sub(r'^(-?)0(?=\.)', r'\1', '{0:.3f}'.format(float(conf_data[conf_rating_cols.index('CurrentHighExpWin%')]))) + ' (' + '{:d}'.format(int(conf_data[conf_rating_cols.index('CurrentHighExpWin%Rank')])) + ')' )
		else:
			rating_line.append('---')
		if not math.isnan(conf_data[conf_rating_cols.index('CurrentMeanRating')]):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(conf_data[conf_rating_cols.index('CurrentMeanRating')])))
		else:
			rating_line.append('---')
		if not math.isnan(conf_data[conf_rating_cols.index('CurrentMeanOffenseRating')]):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(conf_data[conf_rating_cols.index('CurrentMeanOffenseRating')])))
		else:
			rating_line.append('---')
		if not math.isnan(conf_data[conf_rating_cols.index('CurrentMeanDefenseRating')]):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(conf_data[conf_rating_cols.index('CurrentMeanDefenseRating')])))
		else:
			rating_line.append('---')
		if not (math.isnan(conf_data[conf_rating_cols.index('CurrentMeanOffDefRating')]) or math.isnan(conf_data[conf_rating_cols.index('CurrentMeanOffDefRatingRank')])):
			rating_line.append(('{0:.' + str(rating_decimal_places) + 'f}').format(float(conf_data[conf_rating_cols.index('CurrentMeanOffDefRating')])) + ' (' + '{:d}'.format(int(conf_data[conf_rating_cols.index('CurrentMeanOffDefRatingRank')])) + ')')
		else:
			rating_line.append('---')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	# Calculate the widths of the columns, including the headers, in advance
	rating_column_width = []
	for cur_column in range(0, len(rating_header), 1):
		rating_column_width.append(max([len(x[cur_column]) for x in rating_text]))
	# Print the table
	print(table_title)
	for cur_subtitle in table_subtitles:
		print(cur_subtitle)
	for cur_line in range(0, len(rating_text), 1):
		cur_line_text = ''
		for cur_column in range(0, len(rating_header), 1):
			if cur_column > 0:
				cur_line_text += ' '
			cur_line_text += ('{:' + rating_alignment[cur_column] + str(int(rating_column_width[cur_column])) + 's}').format(rating_text[cur_line][cur_column])
		print(cur_line_text)

	print('')
	print('')
	# Read in some overall data about the statistics of the ratings and past games, then calculate thresholds for things like blowouts, close games, high scoring games, and low scoring games
	team_rating_stdev = np.std(np.array([input_data['TeamRatings'][x]['Rating'] for x in input_data['TeamRatings']]))
	team_rating_mean = np.mean(np.array([input_data['TeamRatings'][x]['Rating'] for x in input_data['TeamRatings']]))
	abs_margin_list = np.array(input_data['ActualMarginList'])
	margin_competitive_threshold = np.median(abs_margin_list)
	blowout_game_threshold = round(stats.scoreatpercentile(abs_margin_list, 75), 0)
	close_game_threshold = round(stats.scoreatpercentile(abs_margin_list, 25), 0)
	if input_data['IsPredictionErrorNormal']:
		max_competitive_prob = stats.norm.cdf(margin_competitive_threshold, loc = 0, scale = input_data['PredictionErrorStDev']) - stats.norm.cdf(-margin_competitive_threshold, loc = 0, scale = input_data['PredictionErrorStDev'])
	else:
		max_competitive_prob = stats.t.cdf(margin_competitive_threshold, input_data['PredictionErrorDF'], loc = 0, scale = input_data['PredictionErrorStDev']) - stats.t.cdf(-margin_competitive_threshold, input_data['PredictionErrorDF'], loc = 0, scale = input_data['PredictionErrorStDev'])
	total_score_baseline = input_data['ScoreMean'] * 2
	total_score_list = np.array(input_data['ActualTotalScoreList'])
	low_score_threshold = round(stats.scoreatpercentile(total_score_list, 20), 0)
	high_score_threshold = round(stats.scoreatpercentile(total_score_list, 80), 0)
	tie_cdf_bound = input_data['TieCDFBound']

	# Only iterate once because it's a single week
	if (cutoff_type == 0) or (cutoff_type == 1):
		iter_count = 1
	# Get the date of the first game that hasn't been played yet
	elif cutoff_type == 2:
		unplayed_days = [datetime.datetime.strptime(x['Date'], '%Y-%m-%d').date() for x in input_data['FutureSchedule'] if (x['Season'] == prediction_season) and (len(x['Date']) > 0)]
		if len(unplayed_days) > 0:
			first_unplayed_day = min(unplayed_days)
			iter_count = (cutoff_date - first_unplayed_day).days + 1
		else:
			iter_count = 0
	# Otherwise, no games, but this shouldn't really happen
	else:
		iter_count = 0

	# Loop through each iteration (which may only be once) and print the games during that period
	for iter_game_period in range(0, iter_count, 1):
		# Specify a week as a number, so we match that
		if cutoff_type == 0:
			game_prediction_list = sorted([x for x in input_data['FutureSchedule'] if (x['Season'] == prediction_season) and (x['Week'] == prediction_week)], key = lambda x: x['Date'])
		# Specify a week as a string, so we match that
		elif cutoff_type == 1:
			game_prediction_list = sorted([x for x in input_data['FutureSchedule'] if (x['Season'] == prediction_season) and (x['WeekString'] == prediction_week_str)], key = lambda x: x['Date'])
		# Get the next date, find all games that day, and print a header
		elif cutoff_type == 2:
			cur_search_date = first_unplayed_day + datetime.timedelta(days = iter_game_period)
			game_prediction_list = [x for x in input_data['FutureSchedule'] if (x['Season'] == prediction_season) and (len(x['Date']) > 0) and (datetime.datetime.strptime(x['Date'], '%Y-%m-%d').date() == cur_search_date)]
			if len(game_prediction_list) > 0:
				print('')
				print('Games on ' + cur_search_date.strftime('%A, %B') + ' ' + re.sub(r'^\D*0*', '', cur_search_date.strftime('%d')) + ', ' + cur_search_date.strftime('%Y'))
				print('')

		# Queue up the data with game predictions in advance so it can be sorted and then printed
		game_rating_data = []

		# Loop through each game for the period where predictions will be printed, only counting games that haven't been played yet
		for cur_game in game_prediction_list:
			# Only do this for games where one of the opponents is in the division
			if ((cur_game['HomeDivision'] == division_id) or (cur_game['AwayDivision'] == division_id) or (division_id is None)) and (cur_game['HomeRating'] is not None) and (cur_game['AwayRating'] is not None):
				# Determine the home advantage
				if cur_game['IsNeutralSite']:
					game_home_advantage = 0
					game_type_str = ' vs. '
				else:
					game_home_advantage = input_data['HomeAdvantage']
					game_type_str = ' at '
				# Predict the margin
				game_margin = cur_game['HomeRating'] + game_home_advantage - cur_game['AwayRating']
				# Estimate the score and total points
				game_away_team = cur_game['AwayName']
				game_away_eff_rating = cur_game['AwayRating'] - (game_home_advantage / 2)
				game_home_team = cur_game['HomeName']
				game_home_eff_rating = cur_game['HomeRating'] + (game_home_advantage / 2)
				game_margin = game_home_eff_rating - game_away_eff_rating
				game_home_est_score = max(cur_game['HomeOffenseRating'] + (game_home_advantage / 2) - cur_game['AwayDefenseRating'] + input_data['ScoreMean'], 0)
				game_away_est_score = max(cur_game['AwayOffenseRating'] - (game_home_advantage / 2) - cur_game['HomeDefenseRating'] + input_data['ScoreMean'], 0)
				game_est_total_pts = game_home_est_score + game_away_est_score
				# Using the predicted margin, calculate the probabilitiy of each team winning, a blowout, a close game, and a "competitive" game
				if input_data['IsPredictionErrorNormal']:
					game_away_prob = stats.norm.cdf(-tie_cdf_bound, loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_home_prob = 1 - stats.norm.cdf(tie_cdf_bound, loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_blowout_home = 1 - stats.norm.cdf(blowout_game_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_blowout_away = stats.norm.cdf(-blowout_game_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_close_prob = stats.norm.cdf(close_game_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev']) - stats.norm.cdf(-close_game_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_quality_competitive = (stats.norm.cdf(margin_competitive_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev']) - stats.norm.cdf(-margin_competitive_threshold, loc = game_margin, scale = input_data['PredictionErrorStDev'])) / max_competitive_prob
				# Or do this with a Student's t distribution if that's appropriate for the data set
				else:
					game_away_prob = stats.t.cdf(-tie_cdf_bound, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_home_prob = 1 - stats.t.cdf(tie_cdf_bound, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_blowout_home = 1 - stats.t.cdf(blowout_game_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_blowout_away = stats.t.cdf(-blowout_game_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_close_prob = stats.t.cdf(close_game_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev']) - stats.t.cdf(-close_game_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])
					game_quality_competitive = (stats.t.cdf(margin_competitive_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev']) - stats.t.cdf(-margin_competitive_threshold, input_data['PredictionErrorDF'], loc = game_margin, scale = input_data['PredictionErrorStDev'])) / max_competitive_prob
				# Allow for the possibility of ties in the probabilities
				if tie_cdf_bound > 0:
					game_tie_prob = 1 - (game_away_prob + game_home_prob)
				else:
					game_tie_prob = 0
				# Calculate the "quality" of each team by applying a normal distribution to all teams in the data set, then finding each team's position on the distribution, and then find the overall quality rating for the game
				game_quality_away = stats.norm.cdf(game_away_eff_rating, loc = team_rating_mean, scale = team_rating_stdev)
				game_quality_home = stats.norm.cdf(game_home_eff_rating, loc = team_rating_mean, scale = team_rating_stdev)
				game_quality_teams = np.sqrt(game_quality_away * game_quality_home)
				game_quality_overall = np.power(game_quality_away * game_quality_home * game_quality_competitive, float(1)/float(3))
				# Calculate the overall blowout probability, because either team could win in a blowout, so this sums the tails of the distribution
				game_blowout_prob = game_blowout_home + game_blowout_away
				# Using the error statistics for total score, calculate the probability of a high or a low scoring game
				if input_data['IsTotalScoreErrorNormal']:
					low_scoring_prob = stats.norm.cdf(low_score_threshold, loc = game_est_total_pts, scale = input_data['TotalScoreErrorStDev'])
					high_scoring_prob = 1 - stats.norm.cdf(high_score_threshold, loc = game_est_total_pts, scale = input_data['TotalScoreErrorStDev'])
				else:
					low_scoring_prob = stats.t.cdf(low_score_threshold, input_data['TotalScoreErrorDF'], loc = game_est_total_pts, scale = input_data['TotalScoreErrorStDev'])
					high_scoring_prob = 1 - stats.t.cdf(high_score_threshold, input_data['TotalScoreErrorDF'], loc = game_est_total_pts, scale = input_data['TotalScoreErrorStDev'])
				# Prepare the first row of the output, which may have a different format depending on whether ties are possible or not
				if tie_cdf_bound > 0:
					row1str = game_away_team + ' (' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(-game_margin)) + ', ' + '{0:.2f}'.format(float(game_away_prob * 100)) + '%)' + game_type_str + game_home_team + ' (' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(game_margin)) + ', ' + '{0:.2f}'.format(float(game_home_prob * 100)) + '%), Tie (' + '{0:.2f}'.format(float(game_tie_prob * 100)) + '%)'
				else:
					row1str = game_away_team + ' (' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(-game_margin)) + ', ' + '{0:.2f}'.format(float(game_away_prob * 100)) + '%)' + game_type_str + game_home_team + ' (' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(game_margin)) + ', ' + '{0:.2f}'.format(float(game_home_prob * 100)) + '%)'
				# Prepare text for the other rows of the game prediction
				row2str = 'Estimated score: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(game_away_est_score)) + ' - ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(game_home_est_score)) + ', Total: ' + ('{0:.' + str(rating_decimal_places) + 'f}').format(float(game_est_total_pts))
				row3str = 'Quality: ' + '{0:.2f}'.format(float(game_quality_overall * 100)) + '%, Team quality: ' + '{0:.2f}'.format(float(game_quality_teams * 100)) + '%, Competitiveness: ' + '{0:.2f}'.format(float(game_quality_competitive * 100)) + '%'
				row4str = 'Blowout probability (margin >= ' + '{0:.1f}'.format(blowout_game_threshold) + ' ' + points_abbrev +'): ' + '{0:.2f}'.format(float(game_blowout_prob * 100)) + '%'
				row5str = 'Close game probability (margin <= ' + '{0:.1f}'.format(close_game_threshold) + ' ' + points_abbrev +'): ' + '{0:.2f}'.format(float(game_close_prob * 100)) + '%'
				row6str = 'High scoring probability (total >= ' + '{0:.1f}'.format(float(high_score_threshold)) + ' ' + points_abbrev +'): ' + '{0:.2f}'.format(float(high_scoring_prob * 100)) + '%'
				row7str = 'Low scoring probability (total <= ' + '{0:.1f}'.format(float(low_score_threshold)) + ' ' + points_abbrev +'): ' + '{0:.2f}'.format(float(low_scoring_prob * 100)) + '%'
				# Store these in a data structure and put it in a list that can be sorted in later to order games from the highest to the lowest quality
				cur_game_rating_data = {'Row1Str': row1str, 'Row2Str': row2str, 'Row3Str': row3str, 'Row4Str': row4str, 'Row5Str': row5str, 'Row6Str': row6str, 'Row7Str': row7str, 'Quality': game_quality_overall}
				game_rating_data.append(cur_game_rating_data)
		# Print the sorted game predictions
		for cur_game_rating_data in [[y[0] + 1, y[1]] for y in enumerate(sorted(game_rating_data, key = lambda x: x['Quality'], reverse = True))]:
			print('#' + str(cur_game_rating_data[0]) + ': ' + cur_game_rating_data[1]['Row1Str'])
			print(cur_game_rating_data[1]['Row2Str'])
			print(cur_game_rating_data[1]['Row3Str'])
			print(cur_game_rating_data[1]['Row4Str'])
			print(cur_game_rating_data[1]['Row5Str'])
			print(cur_game_rating_data[1]['Row6Str'])
			print(cur_game_rating_data[1]['Row7Str'])
			print('')

if __name__ == '__main__':
	main()
