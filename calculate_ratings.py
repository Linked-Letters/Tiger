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
import joblib
import json
import networkx
import numpy as np
import tqdm
import scipy.stats as stats
import sys

# This function is intended to run in parallel and attempt to rate the teams
def rating_attempt (attempt, teams_in_network, network_game_id_list, input_games_home_id, input_games_away_id, input_games_home_listid, input_games_away_listid, input_games_home_score, input_games_away_score, input_games_season, input_games_is_neutral_site, input_games_is_preseason, id_to_index, id_to_all_index, season_to_rank, preseason_weight, prior_season_weight, prior_preseason_weight, use_preseason_var, game_network_edge_centrality, network_game_score_mean, network_game_margin_stdev, iterations_before_reset, iterations_before_stop, rating_adjustment_scale):
	# Set up a random number generator with a seed that guarantees the results are repeatable but different from all other attempts
	update_rng = np.random.default_rng(seed = [attempt])

	# Set the initial team ratings to zero
	best_team_off_ratings = np.zeros(len(teams_in_network)).tolist()
	best_team_def_ratings = np.zeros(len(teams_in_network)).tolist()
	best_team_ratings = np.add(np.array(best_team_off_ratings), np.array(best_team_def_ratings)).tolist()
	best_abs_total_error = None
	best_home_advantage = 0
	best_abs_total_error = None

	# Set up the initial conditions
	terminate_search = False
	first_iteration = True
	cur_team_off_ratings = np.zeros(len(teams_in_network)).tolist()
	cur_team_def_ratings = np.zeros(len(teams_in_network)).tolist()
	cur_team_ratings = np.add(np.array(cur_team_off_ratings), np.array(cur_team_def_ratings)).tolist()
	cur_home_advantage = 0
	iterations_since_last_update = 0

	# Iterate through the ratings until told to stop
	while not terminate_search:
		total_home_margin_error_weight = 0
		home_predicted_margin_list = []
		total_team_off_margin_error_weight = np.zeros(len(teams_in_network)).tolist()
		total_team_def_margin_error_weight = np.zeros(len(teams_in_network)).tolist()
		home_predicted_off_score_list = []
		home_predicted_def_score_list = []

		# If this is the first iteration, we'll need to store some data that only needs to be calculated once, so create those lists
		if first_iteration:
			total_team_weight = np.zeros(len(teams_in_network)).astype(int).tolist()
			total_notneutral_weight = 0
			home_actual_margin_list = []
			gt_game_weight_list = []
			home_actual_off_score_list = []
			home_actual_def_score_list = []

		# Loop through all games in the network
		for cur_game_id in network_game_id_list:
			# Get the references to the teams in the arrays
			cur_game_home_teamid = input_games_home_id[cur_game_id]
			cur_game_away_teamid = input_games_away_id[cur_game_id]
			cur_game_homeidx = id_to_index[cur_game_home_teamid]
			cur_game_awayidx = id_to_index[cur_game_away_teamid]
			cur_game_home_allidx = id_to_all_index[cur_game_home_teamid]
			cur_game_away_allidx = id_to_all_index[cur_game_away_teamid]
			# Get the actual score of the game
			cur_game_home_score = input_games_home_score[cur_game_id]
			cur_game_away_score = input_games_away_score[cur_game_id]
			cur_game_season = input_games_season[cur_game_id]
			# Determine the weight of the game
			if cur_game_season == season_to_rank:
				cur_game_weight = 1.0
				if use_preseason_var:
					cur_game_is_preseason = input_games_is_preseason[cur_game_id]
					if cur_game_is_preseason:
						cur_game_weight = preseason_weight
			elif cur_game_season < season_to_rank:
				cur_game_weight = prior_season_weight
				if use_preseason_var:
					cur_game_is_preseason = input_games_is_preseason[cur_game_id]
					if cur_game_is_preseason:
						cur_game_weight = prior_preseason_weight
			else:
				cur_game_weight = 0.0
			# Adjust accordingly based on its centrality in the network of games
			cur_game_gt_weight = cur_game_weight / np.log(1 / game_network_edge_centrality[(min(cur_game_home_allidx, cur_game_away_allidx), max(cur_game_home_allidx, cur_game_away_allidx), 0)])
			# Predict the scores of the home and away teams, and the margin
			cur_game_is_neutral_site = input_games_is_neutral_site[cur_game_id]
			cur_game_predicted_off_score = max(cur_team_off_ratings[cur_game_homeidx] + (int(not cur_game_is_neutral_site) * cur_home_advantage / 2.0) - cur_team_def_ratings[cur_game_awayidx] + network_game_score_mean, 0)
			cur_game_predicted_def_score = max(cur_team_off_ratings[cur_game_awayidx] - (int(not cur_game_is_neutral_site) * cur_home_advantage / 2.0) - cur_team_def_ratings[cur_game_homeidx] + network_game_score_mean, 0)
			cur_game_predicted_margin = cur_game_predicted_off_score - cur_game_predicted_def_score
			# Calculate the actual margin for the game
			cur_game_actual_margin = cur_game_home_score - cur_game_away_score
			# If this is the first iteration, populate arrays with things like the weight of the game and the actual score and margin
			if first_iteration:
				total_team_weight[cur_game_homeidx] += cur_game_gt_weight
				total_team_weight[cur_game_awayidx] += cur_game_gt_weight
				if not cur_game_is_neutral_site:
					total_notneutral_weight += cur_game_gt_weight
				home_actual_margin_list.append(cur_game_actual_margin)
				home_actual_off_score_list.append(cur_game_home_score)
				home_actual_def_score_list.append(cur_game_away_score)
				gt_game_weight_list.append(cur_game_gt_weight)
			# Populate some lists with the predicted margin, predicted scores, and the errors
			home_predicted_margin_list.append(cur_game_predicted_margin)
			cur_game_margin_error = cur_game_actual_margin - cur_game_predicted_margin
			home_predicted_off_score_list.append(cur_game_predicted_off_score)
			home_predicted_def_score_list.append(cur_game_predicted_def_score)
			cur_game_off_margin_error = cur_game_home_score - cur_game_predicted_off_score
			cur_game_def_margin_error = cur_game_away_score - cur_game_predicted_def_score
			total_team_off_margin_error_weight[cur_game_homeidx] += cur_game_off_margin_error * cur_game_gt_weight
			total_team_def_margin_error_weight[cur_game_homeidx] -= cur_game_def_margin_error * cur_game_gt_weight
			total_team_off_margin_error_weight[cur_game_awayidx] += cur_game_def_margin_error * cur_game_gt_weight
			total_team_def_margin_error_weight[cur_game_awayidx] -= cur_game_off_margin_error * cur_game_gt_weight
			# If the game isn't played at a neutral site, we also want to update the overall error
			if not cur_game_is_neutral_site:
				total_home_margin_error_weight += cur_game_margin_error * cur_game_gt_weight
		# The contribution of the total error from the predicted margin
		abs_total_error = np.sum(np.multiply(np.array(gt_game_weight_list), np.absolute(np.subtract(stats.norm.cdf(np.array(home_actual_margin_list), loc = 0, scale = network_game_margin_stdev), stats.norm.cdf(np.array(home_predicted_margin_list), loc = 0, scale = network_game_margin_stdev)))))
		# Add in the contribution from the predicted score errors
		abs_total_error = abs_total_error + np.sum(np.multiply(np.array(gt_game_weight_list + gt_game_weight_list), np.absolute(np.subtract(stats.norm.cdf(np.array(home_actual_off_score_list + home_actual_def_score_list), loc = network_game_score_mean, scale = network_game_margin_stdev), stats.norm.cdf(np.array(home_predicted_off_score_list + home_predicted_def_score_list), loc = network_game_score_mean, scale = network_game_margin_stdev)))))
		# Determine if this rating attempt should be stored as the "best" rating attempt
		update_best_stats = False
		# If it's the first iteration, then it's automatically the best
		if first_iteration:
			update_best_stats = True
		# Or if the error is lower, it's the best rating attempt
		elif abs_total_error < best_abs_total_error:
			update_best_stats = True
		# If it's the best rating attempt, store these ratings and the home advantage, and then reset the counter of iterations since we last improved the ratings
		if update_best_stats:
			best_team_off_ratings = cur_team_off_ratings.copy()
			best_team_def_ratings = cur_team_def_ratings.copy()
			best_team_ratings = cur_team_ratings.copy()
			best_home_advantage = cur_home_advantage
			best_abs_total_error = abs_total_error
			iterations_since_last_update = 0
		# Otherwise, it's not the best rating attempt, so iterate our counter since we last found the best rating attempt
		else:
			iterations_since_last_update += 1
			# Because the ratings are updated with random numbers on each iteration to search for the best ratings, this can cause some drift, so every so often we reset back to the previous best rating attempt
			if (iterations_since_last_update % iterations_before_reset) == 0:
				cur_team_off_ratings = best_team_off_ratings.copy()
				cur_team_def_ratings = best_team_def_ratings.copy()
				cur_team_ratings = best_team_ratings.copy()
				cur_home_advantage = best_home_advantage
		# If we haven't gone enough iterations without improvement to stop, we need to update the ratings
		if iterations_since_last_update < iterations_before_stop:
			# Update the offense and defense ratings for the teams
			for cur_teamidx in range(0, len(cur_team_ratings), 1):
				cur_team_off_ratings[cur_teamidx] += update_rng.normal(loc = (total_team_off_margin_error_weight[cur_teamidx] / total_team_weight[cur_teamidx]) / rating_adjustment_scale, scale = (1 / rating_adjustment_scale))
				cur_team_def_ratings[cur_teamidx] += update_rng.normal(loc = (total_team_def_margin_error_weight[cur_teamidx] / total_team_weight[cur_teamidx]) / rating_adjustment_scale, scale = (1 / rating_adjustment_scale))
			# Update the home advantage
			cur_home_advantage += update_rng.normal(loc = (total_home_margin_error_weight / total_notneutral_weight) / rating_adjustment_scale, scale = (1 / rating_adjustment_scale))
			# Center the offense and defense ratings on zero
			cur_team_off_ratings = np.subtract(np.array(cur_team_off_ratings), np.mean(np.array(cur_team_off_ratings))).tolist()
			cur_team_def_ratings = np.subtract(np.array(cur_team_def_ratings), np.mean(np.array(cur_team_def_ratings))).tolist()
			# Update the overall ratings to be the sum of the offense and defense ratings
			cur_team_ratings = np.add(np.array(cur_team_off_ratings), np.array(cur_team_def_ratings)).tolist()
		# If we have gone too many iterations without an improvement, halt the rating process
		else:
			terminate_search = True
		# We've iterated at least once, so we no longer want to do the first iteration tasks
		first_iteration = False
	# Return the best set of ratings that were found
	return [best_team_off_ratings, best_team_def_ratings, best_team_ratings, best_home_advantage]

def main ():
	# This is the number of iterations without a rating improvement before a reset; add this as a command line parameter at some point, though it probably doesn't need to be changed often
	iterations_before_reset = 200
	# This is the number of iterations without a rating improvement before stopping; add this as a command line parameter at some point, though it probably doesn't need to be changed often
	iterations_before_stop = 1000
	# What is the scale for adjusting the ratings, and a factor for scaling it related to the standard deviation of the scoring margin; add these as command line parameters at some point, though they probably don't need to be changed often
	rating_adjustment_scale_baseline = 20
	rating_adjustment_scale_factor = 27
	# Set this to none if there should be no cap on the rating adjustment scale, though this can make runtimes excessively long in some instances ; add this as a command line parameter in the future, though it probably doesn't need to be changed often
	rating_adjustment_scale_cap = 100

	if (len(sys.argv) < 14):
		print('Usage: '+sys.argv[0]+' <input JSON file> <output JSON file> <processes to use> <season to rate> <earliest season to load> <preseason weight 0-1> <prior season weight 0-1> <prior preseason weight 0-1> <match teams with ID instead of name, y or n> <final date to rank YYYY-MM-DD> <tie probability> <tie probability search interval, suggest 0.001 for football points, adjust appropriately> <number of ratings attempts to average>')
		exit()

	input_file_name = sys.argv[1].strip()
	output_file_name = sys.argv[2].strip()
	try:
		parallel_processes = int(sys.argv[3].strip())
		season_to_rank = int(sys.argv[4].strip())
		earliest_season = int(sys.argv[5].strip())
		preseason_weight = np.clip(float(sys.argv[6].strip()), 0, 1)
		prior_season_weight = np.clip(float(sys.argv[7].strip()), 0, 1)
		prior_preseason_weight = np.clip(float(sys.argv[8].strip()), 0, 1)
		final_date_str = sys.argv[10].strip()
		final_rank_year = int(final_date_str[0:4])
		final_rank_month = int(final_date_str[5:7])
		final_rank_day = int(final_date_str[8:10])
		final_date_to_rank = datetime.date(final_rank_year, final_rank_month, final_rank_day)
		tie_probability = float(sys.argv[11].strip())
		tie_cdf_search_interval = float(sys.argv[12].strip())
		num_rating_attempts = int(sys.argv[13].strip())
	except:
		print('Error in seasons, weights, dates, or probabilities; please check for correctness')
		exit()
	try:
		str_var = sys.argv[9].strip()
		if (str_var[0] == 'y') or (str_var[0] == 'Y'):
			use_team_id = True
		elif (str_var[0] == 'n') or (str_var[0] == 'N'):
			use_team_id = False
		else:
			raise Exception('Invalid option')
	except:
		print('Error in team match parameter')
		exit()

	# Load the input JSON data
	input_handle = open(input_file_name, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	input_data = json.load(input_handle)
	input_handle.close()

	# Get a list of all data variables
	data_keys = list(set([y for z in [list(input_data[x].keys()) for x in list(input_data.keys())] for y in z]))

	# Set some basic filters to be selective about which games are and are not read in
	game_filter = [[x, all([input_data[x]['IsCompleted'], (input_data[x]['HomeScore'] is not None), (input_data[x]['AwayScore'] is not None), (input_data[x]['IsNeutralSite'] is not None), (input_data[x]['Season'] is not None), ((input_data[x]['Year'] is not None) and (input_data[x]['Month'] is not None) and (input_data[x]['Day'] is not None) and (datetime.date(int(str(input_data[x]['Year']).strip()), int(str(input_data[x]['Month']).strip()), int(str(input_data[x]['Day']).strip())) <= final_date_to_rank)), ((input_data[x]['Season'] is not None) and (int(input_data[x]['Season']) >= earliest_season) and (int(input_data[x]['Season']) <= season_to_rank))])] for x in input_data.keys()]
	unplayed_game_filter = [[x, all([not input_data[x]['IsCompleted'], (input_data[x]['IsNeutralSite'] is not None), (input_data[x]['Season'] is not None), ((input_data[x]['Year'] is not None) and (input_data[x]['Month'] is not None) and (input_data[x]['Day'] is not None)), ((input_data[x]['Season'] is not None) and (int(input_data[x]['Season']) >= season_to_rank))])] for x in input_data.keys()]
	if data_keys.count('IsPreseason'):
		use_preseason_var = True
		game_filter = [[x[0], ((input_data[x[0]]['IsPreseason'] is not None) and x[1])] for x in game_filter]
		unplayed_game_filter = [[x[0], ((input_data[x[0]]['IsPreseason'] is not None) and x[1])] for x in unplayed_game_filter]
	if data_keys.count('IsPostponed'):
		game_filter = [[x[0], ((not input_data[x[0]]['IsPostponed']) and x[1])] for x in game_filter]
		unplayed_game_filter = [[x[0], ((not input_data[x[0]]['IsPostponed']) and x[1])] for x in unplayed_game_filter]
	if use_team_id:
		game_filter = [[x[0], all([input_data[x[0]]['HomeID'] is not None, input_data[x[0]]['AwayID'] is not None, x[1]])] for x in game_filter]
		unplayed_game_filter = [[x[0], all([input_data[x[0]]['HomeID'] is not None, input_data[x[0]]['AwayID'] is not None, x[1]])] for x in unplayed_game_filter]

	# Load in data for games that satisfy the filters, including a bunch of stuff that we want to pass through from the input file to the ratings like division and conference
	input_games_home_score = [input_data[x[0]]['HomeScore'] for x in game_filter if x[1]]
	input_games_away_score = [input_data[x[0]]['AwayScore'] for x in game_filter if x[1]]
	input_games_date = [datetime.date(int(str(input_data[x[0]]['Year']).strip()), int(str(input_data[x[0]]['Month']).strip()), int(str(input_data[x[0]]['Day']).strip())) for x in game_filter if x[1]]
	input_games_is_neutral_site = [input_data[x[0]]['IsNeutralSite'] for x in game_filter if x[1]]
	input_games_season = [input_data[x[0]]['Season'] for x in game_filter if x[1]]
	if use_preseason_var:
		input_games_is_preseason = [input_data[x[0]]['IsPreseason'] for x in game_filter if x[1]]
	else:
		input_games_is_preseason = [False for x in game_filter if x[1]]
	if use_team_id:
		input_games_home_id = [input_data[x[0]]['HomeID'] for x in game_filter if x[1]]
		input_games_away_id = [input_data[x[0]]['AwayID'] for x in game_filter if x[1]]
		input_games_home_name = [input_data[x[0]]['HomeName'] for x in game_filter if x[1]]
		input_games_away_name = [input_data[x[0]]['AwayName'] for x in game_filter if x[1]]
	else:
		input_games_home_id = [input_data[x[0]]['HomeName'] for x in game_filter if x[1]]
		input_games_away_id = [input_data[x[0]]['AwayName'] for x in game_filter if x[1]]
	if data_keys.count('HomeConference'):
		input_games_home_conference = [input_data[x[0]]['HomeConference'] for x in game_filter if x[1]]
	else:
		input_games_home_conference = [None] * len(game_filter)
	if data_keys.count('AwayConference'):
		input_games_away_conference = [input_data[x[0]]['AwayConference'] for x in game_filter if x[1]]
	else:
		input_games_away_conference = [None] * len(game_filter)
	if data_keys.count('HomeDivision'):
		input_games_home_division = [input_data[x[0]]['HomeDivision'] for x in game_filter if x[1]]
	else:
		input_games_home_division = [None] * len(game_filter)
	if data_keys.count('AwayDivision'):
		input_games_away_division = [input_data[x[0]]['AwayDivision'] for x in game_filter if x[1]]
	else:
		input_games_away_division = [None] * len(game_filter)
	if data_keys.count('IsConferenceGame'):
		input_games_conference_game = [input_data[x[0]]['IsConferenceGame'] for x in game_filter if x[1]]
	else:
		input_games_conference_game = [None] * len(game_filter)
	if data_keys.count('Week'):
		input_games_week = [input_data[x[0]]['Week'] for x in game_filter if x[1]]
	else:
		input_games_week = [None] * len(game_filter)
	if data_keys.count('WeekString'):
		input_games_weekstring = [input_data[x[0]]['WeekString'] for x in game_filter if x[1]]
	else:
		input_games_weekstring = [None] * len(game_filter)

	# Load in data for unplayed games, which is mostly the same as the games that have already been played minus the scores
	unplayed_games_date = [datetime.date(int(str(input_data[x[0]]['Year']).strip()), int(str(input_data[x[0]]['Month']).strip()), int(str(input_data[x[0]]['Day']).strip())) for x in unplayed_game_filter if x[1]]
	unplayed_games_is_neutral_site = [input_data[x[0]]['IsNeutralSite'] for x in unplayed_game_filter if x[1]]
	unplayed_games_season = [input_data[x[0]]['Season'] for x in unplayed_game_filter if x[1]]
	if use_preseason_var:
		unplayed_games_is_preseason = [input_data[x[0]]['IsPreseason'] for x in unplayed_game_filter if x[1]]
	if use_team_id:
		unplayed_games_home_id = [input_data[x[0]]['HomeID'] for x in unplayed_game_filter if x[1]]
		unplayed_games_away_id = [input_data[x[0]]['AwayID'] for x in unplayed_game_filter if x[1]]
		unplayed_games_home_name = [input_data[x[0]]['HomeName'] for x in unplayed_game_filter if x[1]]
		unplayed_games_away_name = [input_data[x[0]]['AwayName'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_home_id = [input_data[x[0]]['HomeName'] for x in unplayed_game_filter if x[1]]
		unplayed_games_away_id = [input_data[x[0]]['AwayName'] for x in unplayed_game_filter if x[1]]
		unplayed_games_home_name = [input_data[x[0]]['HomeName'] for x in unplayed_game_filter if x[1]]
		unplayed_games_away_name = [input_data[x[0]]['AwayName'] for x in unplayed_game_filter if x[1]]
	if data_keys.count('HomeConference'):
		unplayed_games_home_conference = [input_data[x[0]]['HomeConference'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_home_conference = [None] * len(unplayed_game_filter)
	if data_keys.count('AwayConference'):
		unplayed_games_away_conference = [input_data[x[0]]['AwayConference'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_away_conference = [None] * len(unplayed_game_filter)
	if data_keys.count('HomeDivision'):
		unplayed_games_home_division = [input_data[x[0]]['HomeDivision'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_home_division = [None] * len(unplayed_game_filter)
	if data_keys.count('AwayDivision'):
		unplayed_games_away_division = [input_data[x[0]]['AwayDivision'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_away_division = [None] * len(unplayed_game_filter)
	if data_keys.count('IsConferenceGame'):
		unplayed_games_conference_game = [input_data[x[0]]['IsConferenceGame'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_conference_game = [None] * len(unplayed_game_filter)
	if data_keys.count('Week'):
		unplayed_games_week = [input_data[x[0]]['Week'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_week = [None] * len(unplayed_game_filter)
	if data_keys.count('WeekString'):
		unplayed_games_weekstring = [input_data[x[0]]['WeekString'] for x in unplayed_game_filter if x[1]]
	else:
		unplayed_games_weekstring = [None] * len(unplayed_game_filter)

	# Get a list of all teams
	if use_team_id:
		team_id_name_links = {}
		for cur_season in range(season_to_rank, earliest_season - 1, -1):
			for cur_game_id in range(0, len(input_games_home_id), 1):
				cur_game_homeid = input_games_home_id[cur_game_id]
				cur_game_awayid = input_games_away_id[cur_game_id]
				cur_game_homename = input_games_home_name[cur_game_id]
				cur_game_awayname = input_games_away_name[cur_game_id]
				cur_game_season = input_games_season[cur_game_id]
				if (cur_season == cur_game_season):
					if list(team_id_name_links.keys()).count(cur_game_homeid) == 0:
						team_id_name_links[cur_game_homeid] = cur_game_homename
					if list(team_id_name_links.keys()).count(cur_game_awayid) == 0:
						team_id_name_links[cur_game_awayid] = cur_game_awayname
			for cur_game_id in range(0, len(unplayed_games_home_id), 1):
				cur_game_homeid = unplayed_games_home_id[cur_game_id]
				cur_game_awayid = unplayed_games_away_id[cur_game_id]
				cur_game_homename = unplayed_games_home_name[cur_game_id]
				cur_game_awayname = unplayed_games_away_name[cur_game_id]
				cur_game_season = unplayed_games_season[cur_game_id]
				if (cur_season == cur_game_season):
					if list(team_id_name_links.keys()).count(cur_game_homeid) == 0:
						team_id_name_links[cur_game_homeid] = cur_game_homename
					if list(team_id_name_links.keys()).count(cur_game_awayid) == 0:
						team_id_name_links[cur_game_awayid] = cur_game_awayname
		# Sort the team IDs in the order of their names so that the rest of the rating system works identically regardless of whether IDs or names are used internally
		all_team_list = [y[1] for y in sorted([[team_id_name_links[x], x] for x in list(team_id_name_links.keys())], key = lambda x: x[0])]
	else:
		all_team_list = sorted(list(set(input_games_home_id + input_games_away_id)))
		team_id_name_links = {}
		for team_id in all_team_list:
			team_id_name_links[team_id] = team_id

	# Get a list of divisions and conferences, and then affiliate the teams appropriately for their current season (allowing for realignment from past seasons)
	team_id_division_links = {}
	team_id_conference_links = {}
	for team_id in list(set(input_games_home_id + input_games_away_id + unplayed_games_home_id + unplayed_games_away_id)):
		team_id_division_links[team_id] = None
		team_id_conference_links[team_id] = None
	for cur_game_id in range(0, len(input_games_home_id), 1):
		cur_game_season = input_games_season[cur_game_id]
		if cur_game_season == season_to_rank:
			cur_game_homeid = input_games_home_id[cur_game_id]
			cur_game_awayid = input_games_away_id[cur_game_id]
			if team_id_division_links[cur_game_homeid] is None:
				team_id_division_links[cur_game_homeid] = input_games_home_division[cur_game_id]
			if team_id_division_links[cur_game_awayid] is None:
				team_id_division_links[cur_game_awayid] = input_games_away_division[cur_game_id]
			if team_id_conference_links[cur_game_homeid] is None:
				team_id_conference_links[cur_game_homeid] = input_games_home_conference[cur_game_id]
			if team_id_conference_links[cur_game_awayid] is None:
				team_id_conference_links[cur_game_awayid] = input_games_away_conference[cur_game_id]
	for cur_game_id in range(0, len(unplayed_games_home_id), 1):
		cur_game_season = unplayed_games_season[cur_game_id]
		if cur_game_season == season_to_rank:
			cur_game_homeid = unplayed_games_home_id[cur_game_id]
			cur_game_awayid = unplayed_games_away_id[cur_game_id]
			if team_id_division_links[cur_game_homeid] is None:
				team_id_division_links[cur_game_homeid] = unplayed_games_home_division[cur_game_id]
			if team_id_division_links[cur_game_awayid] is None:
				team_id_division_links[cur_game_awayid] = unplayed_games_away_division[cur_game_id]
			if team_id_conference_links[cur_game_homeid] is None:
				team_id_conference_links[cur_game_homeid] = unplayed_games_home_conference[cur_game_id]
			if team_id_conference_links[cur_game_awayid] is None:
				team_id_conference_links[cur_game_awayid] = unplayed_games_away_conference[cur_game_id]

	# Get the team indexes for each game
	input_games_home_listid = [all_team_list.index(x) for x in input_games_home_id]
	input_games_away_listid = [all_team_list.index(x) for x in input_games_away_id]

	# And then check for connected components
	game_network = networkx.MultiGraph()
	game_network.add_nodes_from([x for x in list(range(0, len(all_team_list), 1))])
	game_network.add_edges_from([(input_games_home_listid[x], input_games_away_listid[x]) for x in range(0, len(input_games_home_id), 1)])
	game_network_edge_centrality = networkx.edge_betweenness_centrality(game_network)
	game_network_groups = list(networkx.connected_components(game_network))
	if len(game_network_groups) > 1:
		print('Unconnected teams in the primary network:')
		for x in range(1, len(game_network_groups), 1):
			for teamid in game_network_groups[x]:
				print(team_id_name_links[all_team_list[teamid]] + ': ' + str(all_team_list[teamid]))
		print(('Proceeding with the largest network, size %d' % len(game_network_groups[0])))
	# Get the teams in the primary network
	teams_in_network = sorted(list(game_network_groups[0]))
	id_to_all_index = dict([(all_team_list[x], x) for x in teams_in_network])
	id_to_index = dict([[y[1], y[0]] for y in enumerate([all_team_list[x] for x in teams_in_network])])
	all_index_to_id = {x: y for y, x in id_to_all_index.items()}
	index_to_id = {x: y for y, x in id_to_index.items()}

	# Filter games so that they only include data when both teams are in the network, and do this only once
	network_game_id_list = [x for x in range(0, len(input_games_home_score), 1) if (teams_in_network.count(input_games_home_listid[x]) > 0) and (teams_in_network.count(input_games_away_listid[x]) > 0)]

	# Then calculate some statistics about the games in the network to tune the rating adjustment process
	network_game_margin_list = [input_games_home_score[x] - input_games_away_score[x] for x in network_game_id_list]
	network_game_margin_stdev = np.std(np.array(network_game_margin_list + np.multiply(np.array(network_game_margin_list), -1).tolist()))
	network_game_margin_abs_list = np.absolute(np.array(network_game_margin_list)).tolist()
	network_game_score_list = [input_games_home_score[x] for x in network_game_id_list] + [input_games_away_score[x] for x in network_game_id_list]
	network_game_total_score_list = [input_games_home_score[x] + input_games_away_score[x] for x in network_game_id_list]
	network_game_score_stdev = np.std(np.array(network_game_score_list))
	network_game_score_mean = np.mean(np.array(network_game_score_list))
	network_total_games = len(network_game_id_list)
	rating_adjustment_scale = rating_adjustment_scale_baseline * (rating_adjustment_scale_factor / network_game_margin_stdev)
	if rating_adjustment_scale_cap is not None:
		rating_adjustment_scale = min(rating_adjustment_scale, rating_adjustment_scale_cap)

	# Set up a link between team indexes (in the arrays) and the IDs
	teams_in_network_reverse = [None] * len(all_team_list)
	for cur_team in teams_in_network:
		teams_in_network_reverse[cur_team] = teams_in_network.index(cur_team)

	# Create empty lists to store the results of the rating attempts
	best_team_ratings_list = []
	best_home_advantage_list = []
	best_team_off_ratings_list = []
	best_team_def_ratings_list = []

	# Run parallel jobs to compute the ratings, each with different random number seeds so that the results from each attempt are unique but repeatable (by predictably assigning the random seeds)
	attempt_results = joblib.Parallel(n_jobs = parallel_processes)(joblib.delayed(rating_attempt)(cur_rating_attempt, teams_in_network, network_game_id_list, input_games_home_id, input_games_away_id, input_games_home_listid, input_games_away_listid, input_games_home_score, input_games_away_score, input_games_season, input_games_is_neutral_site, input_games_is_preseason, id_to_index, id_to_all_index, season_to_rank, preseason_weight, prior_season_weight, prior_preseason_weight, use_preseason_var, game_network_edge_centrality, network_game_score_mean, network_game_margin_stdev, iterations_before_reset, iterations_before_stop, rating_adjustment_scale) for cur_rating_attempt in tqdm.tqdm(range(num_rating_attempts)))

	# Reorder the arrays with the output from the rating attempts
	attempt_results_zip = list(zip(*attempt_results))
	best_team_off_ratings_list = list(attempt_results_zip[0])
	best_team_def_ratings_list = list(attempt_results_zip[1])
	best_team_ratings_list = list(attempt_results_zip[2])
	best_home_advantage_list = list(attempt_results_zip[3])

	# Then obtain the median of the ratings, which will be used as the final ratings
	median_team_off_ratings = np.median(np.array(best_team_off_ratings_list), axis = 0)
	stdev_team_off_ratings = np.std(np.array(best_team_off_ratings_list), axis = 0)
	median_team_def_ratings = np.median(np.array(best_team_def_ratings_list), axis = 0)
	stdev_team_def_ratings = np.std(np.array(best_team_def_ratings_list), axis = 0)
	median_team_ratings = np.median(np.array(best_team_ratings_list), axis = 0)
	stdev_team_ratings = np.std(np.array(best_team_ratings_list), axis = 0)
	median_home_advantage = np.median(np.array(best_home_advantage_list))

	# Estimate the distribution of prediction errors for the margin of victory, the scores of each team, and the total score
	prediction_errors = []
	score_prediction_errors = []
	total_score_prediction_errors = []
	for cur_game_id in network_game_id_list:
		include_game = True
		if use_preseason_var:
			cur_game_is_preseason = input_games_is_preseason[cur_game_id]
			if cur_game_is_preseason and (preseason_weight == 0):
				include_game = False
			if (cur_game_season < season_to_rank) and (prior_preseason_weight == 0) and cur_game_is_preseason:
					include_game = False
		if (cur_game_season < season_to_rank) and (prior_season_weight == 0):
			include_game = False
		if include_game:
			cur_game_homeid = input_games_home_listid[cur_game_id]
			cur_game_awayid = input_games_away_listid[cur_game_id]
			cur_game_homeidx = teams_in_network_reverse[cur_game_homeid]
			cur_game_awayidx = teams_in_network_reverse[cur_game_awayid]
			cur_game_home_score = input_games_home_score[cur_game_id]
			cur_game_away_score = input_games_away_score[cur_game_id]
			cur_game_is_neutral_site = input_games_is_neutral_site[cur_game_id]
			cur_game_predicted_margin = median_team_ratings[cur_game_homeidx] + (int(not cur_game_is_neutral_site) * median_home_advantage) - median_team_ratings[cur_game_awayidx]
			cur_game_actual_margin = cur_game_home_score - cur_game_away_score
			cur_prediction_error = cur_game_predicted_margin - cur_game_actual_margin
			cur_game_predicted_home_score = median_team_off_ratings[cur_game_homeidx] + (int(not cur_game_is_neutral_site) * median_home_advantage / 2) - median_team_def_ratings[cur_game_awayidx] + network_game_score_mean
			cur_game_predicted_away_score = median_team_off_ratings[cur_game_awayidx] - (int(not cur_game_is_neutral_site) * median_home_advantage / 2) + median_team_def_ratings[cur_game_homeidx] + network_game_score_mean
			cur_home_prediction_error = cur_game_predicted_home_score - cur_game_home_score
			cur_away_prediction_error = cur_game_predicted_away_score - cur_game_away_score
			cur_total_prediction_error = (cur_game_predicted_home_score + cur_game_predicted_away_score) - (cur_game_home_score + cur_game_away_score)
			prediction_errors.append(cur_prediction_error)
			score_prediction_errors.append(cur_home_prediction_error)
			score_prediction_errors.append(cur_away_prediction_error)
			total_score_prediction_errors.append(cur_total_prediction_error)
	prediction_error_stdev = np.std(np.array(prediction_errors))
	prediction_error_fit = stats.t.fit(np.array(prediction_errors), floc = 0, fscale = prediction_error_stdev)
	prediction_error_mean = 0
	prediction_error_df = prediction_error_fit[0]
	score_prediction_error_stdev = np.std(score_prediction_errors)
	score_prediction_error_fit = stats.t.fit(np.array(score_prediction_errors), floc = 0, fscale = score_prediction_error_stdev)
	score_prediction_error_df = score_prediction_error_fit[0]
	total_score_prediction_error_stdev = np.std(total_score_prediction_errors)
	total_score_prediction_error_fit = stats.t.fit(np.array(total_score_prediction_errors), floc = 0, fscale = total_score_prediction_error_stdev)
	total_score_prediction_error_df = total_score_prediction_error_fit[0]

	# Determine if we should use a normal distribution (df > 50) or stick with the t distribution
	use_normal = bool(prediction_error_df > 50)
	score_use_normal = bool(score_prediction_error_df > 50)
	total_score_use_normal = bool(total_score_prediction_error_df > 50)

	# Calculate the portion of the distribution to predict a tie
	terminate_search = False
	cur_cdf_tie_probability = 0
	cur_cdf_bound = 0
	while not terminate_search:
		prev_cdf_tie_probability = cur_cdf_tie_probability
		prev_cdf_bound = cur_cdf_bound
		cur_cdf_bound = cur_cdf_bound + tie_cdf_search_interval
		# Estimate the distribution of prediction errors, steadily increasing the proportion of the distribution that constitutes a tie until the estimated probability of a tie matches the observed probability of a tie
		cur_tie_probability_sum = 0.0
		cur_tie_probability_n = 0
		for cur_game_id in network_game_id_list:
			include_game = True
			if use_preseason_var:
				cur_game_is_preseason = input_games_is_preseason[cur_game_id]
				if cur_game_is_preseason and (preseason_weight == 0):
					include_game = False
			if (cur_game_season < season_to_rank) and (prior_season_weight == 0):
				include_game = False
			if include_game:
				cur_game_homeid = input_games_home_listid[cur_game_id]
				cur_game_awayid = input_games_away_listid[cur_game_id]
				cur_game_homeidx = teams_in_network_reverse[cur_game_homeid]
				cur_game_awayidx = teams_in_network_reverse[cur_game_awayid]
				cur_game_home_score = input_games_home_score[cur_game_id]
				cur_game_away_score = input_games_away_score[cur_game_id]
				cur_game_is_neutral_site = input_games_is_neutral_site[cur_game_id]
				cur_game_predicted_margin = median_team_ratings[cur_game_homeidx] + (int(not cur_game_is_neutral_site) * median_home_advantage) - median_team_ratings[cur_game_awayidx]
				if use_normal:
					cur_tie_probability_sum = cur_tie_probability_sum + stats.norm.cdf(cur_cdf_bound, loc = cur_game_predicted_margin, scale = prediction_error_stdev) - stats.norm.cdf(-cur_cdf_bound, loc = cur_game_predicted_margin, scale = prediction_error_stdev)
				else:
					cur_tie_probability_sum = cur_tie_probability_sum + stats.t.cdf(cur_cdf_bound, prediction_error_df, loc = cur_game_predicted_margin, scale = prediction_error_stdev) - stats.t.cdf(-cur_cdf_bound, prediction_error_df, loc = cur_game_predicted_margin, scale = prediction_error_stdev)
				cur_tie_probability_n = cur_tie_probability_n + 1
		cur_cdf_tie_probability = cur_tie_probability_sum / cur_tie_probability_n
		if cur_cdf_tie_probability >= tie_probability:
			terminate_search = True
	tie_cdf_bound = ((tie_probability - prev_cdf_tie_probability) / (cur_cdf_tie_probability - prev_cdf_tie_probability)) * tie_cdf_search_interval + prev_cdf_bound

	# Store some general information about the results of the ratings, which are used for things like predicting the outcomes of future games
	output_data = {}
	output_data['TieCDFBound'] = tie_cdf_bound
	output_data['IsPredictionErrorNormal'] = use_normal
	output_data['PredictionErrorStDev'] = prediction_error_stdev
	output_data['PredictionErrorDF'] = prediction_error_df
	output_data['NumberOfGames'] = len(network_game_id_list)
	output_data['Season'] = season_to_rank
	output_data['EarliestSeasonUsed'] = earliest_season
	output_data['PreseasonWeight'] = preseason_weight
	output_data['PriorPreseasonWeight'] = prior_preseason_weight
	output_data['PriorSeasonWeight'] = prior_season_weight
	output_data['HomeAdvantage'] = median_home_advantage
	output_data['NumberOfRatingAttempts'] = num_rating_attempts
	output_data['FinalRatingDate'] = final_date_to_rank.strftime('%Y-%m-%d')
	output_data['RequestedTieProbability'] = tie_probability
	output_data['HomeAdvantageList'] = best_home_advantage_list
	output_data['IsScoreErrorNormal'] = score_use_normal
	output_data['ScoreErrorStDev'] = score_prediction_error_stdev
	output_data['ScoreErrorDF'] = score_prediction_error_df
	output_data['IsTotalScoreErrorNormal'] = total_score_use_normal
	output_data['TotalScoreErrorStDev'] = total_score_prediction_error_stdev
	output_data['TotalScoreErrorDF'] = total_score_prediction_error_df
	output_data['ScoreMean'] = network_game_score_mean
	output_data['ScoreStDev'] = network_game_score_stdev
	output_data['ActualMarginList'] = network_game_margin_abs_list
	output_data['ActualScoreList'] = network_game_score_list
	output_data['ActualTotalScoreList'] = network_game_total_score_list
	output_data['PastSchedule'] = []
	output_data['FutureSchedule'] = []

	# Sort the ratings so that teams with higher ratings get output first
	team_rating_list = [[x[0] + 1] + [x[1][0]] for x in list(enumerate(sorted(zip(teams_in_network, median_team_ratings), key = lambda x: x[1], reverse = True)))]

	# Look through and store information about each team's rating
	output_data['TeamRatings'] = {}
	for team_rank_row in team_rating_list:
		team_rank = team_rank_row[0]
		team_idx = teams_in_network_reverse[team_rank_row[1]]
		team_id = index_to_id[team_idx]
		team_name = team_id_name_links[all_team_list[team_rank_row[1]]]
		team_division = team_id_division_links[team_id]
		team_conference = team_id_conference_links[team_id]
		output_data['TeamRatings'][team_id] = {}
		output_data['TeamRatings'][team_id]['Rank'] = team_rank
		output_data['TeamRatings'][team_id]['Name'] = team_name
		output_data['TeamRatings'][team_id]['Rating'] = median_team_ratings[team_idx]
		output_data['TeamRatings'][team_id]['OffenseRating'] = median_team_off_ratings[team_idx]
		output_data['TeamRatings'][team_id]['DefenseRating'] = median_team_def_ratings[team_idx]
		output_data['TeamRatings'][team_id]['RatingStDev'] = stdev_team_ratings[team_idx]
		output_data['TeamRatings'][team_id]['OffenseRatingStDev'] = stdev_team_off_ratings[team_idx]
		output_data['TeamRatings'][team_id]['DefenseRatingStDev'] = stdev_team_def_ratings[team_idx]
		output_data['TeamRatings'][team_id]['RatingList'] = np.array(best_team_ratings_list)[:, team_idx].tolist()
		output_data['TeamRatings'][team_id]['OffenseRatingList'] = np.array(best_team_off_ratings_list)[:, team_idx].tolist()
		output_data['TeamRatings'][team_id]['DefenseRatingList'] = np.array(best_team_def_ratings_list)[:, team_idx].tolist()
		output_data['TeamRatings'][team_id]['Division'] = team_division
		output_data['TeamRatings'][team_id]['Conference'] = team_conference
		output_data['TeamRatings'][team_id]['PastSchedule'] = []
		output_data['TeamRatings'][team_id]['FutureSchedule'] = []

		# Store data about the future games so that, while redundant, makes it simpler to postprocess the data
		for cur_game_id in range(0, len(unplayed_games_season), 1):
			include_game = True
			cur_game_homeid = unplayed_games_home_id[cur_game_id]
			cur_game_awayid = unplayed_games_away_id[cur_game_id]
			if (cur_game_homeid != team_id) and (cur_game_awayid != team_id):
				include_game = False
			cur_game_is_preseason = False
			if use_preseason_var:
				cur_game_is_preseason = unplayed_games_is_preseason[cur_game_id]
				if cur_game_is_preseason and (preseason_weight == 0):
					include_game = False
				if (cur_game_season < season_to_rank) and (prior_preseason_weight == 0) and cur_game_is_preseason:
					include_game = False
			cur_game_season = unplayed_games_season[cur_game_id]
			if (cur_game_season < season_to_rank) and (prior_season_weight == 0):
				include_game = False
			if include_game:
				# Get information about the site of the game and if the teams are in the same division and conference
				cur_game_is_neutral_site = unplayed_games_is_neutral_site[cur_game_id]
				cur_game_conference_game = unplayed_games_conference_game[cur_game_id]
				cur_game_is_home_game = False
				cur_game_is_away_game = False
				cur_game_week = unplayed_games_week[cur_game_id]
				cur_game_weekstring = unplayed_games_weekstring[cur_game_id]
				if cur_game_homeid == team_id:
					cur_game_oppid = cur_game_awayid
					if not cur_game_is_neutral_site:
						cur_game_is_home_game = True
					cur_game_oppname = unplayed_games_away_name[cur_game_id]
					cur_game_team_conference = unplayed_games_home_conference[cur_game_id]
					cur_game_team_division = unplayed_games_home_division[cur_game_id]
					cur_game_opp_conference = unplayed_games_away_conference[cur_game_id]
					cur_game_opp_division = unplayed_games_away_division[cur_game_id]
				if cur_game_awayid == team_id:
					cur_game_oppid = cur_game_homeid
					if not cur_game_is_neutral_site:
						cur_game_is_away_game = True
					cur_game_oppname = unplayed_games_home_name[cur_game_id]
					cur_game_team_conference = unplayed_games_away_conference[cur_game_id]
					cur_game_team_division = unplayed_games_away_division[cur_game_id]
					cur_game_opp_conference = unplayed_games_home_conference[cur_game_id]
					cur_game_opp_division = unplayed_games_home_division[cur_game_id]
				# Still include the game if the team isn't in the ratings yet, because it is part of the schedule
				if list(id_to_index.keys()).count(cur_game_oppid) > 0:
					cur_game_oppidx = id_to_index[cur_game_oppid]
				else:
					cur_game_oppidx = None
				# Get the opponent rating, or set it to None if the opponent hasn't been rated yet
				if cur_game_oppidx is not None:
					cur_game_opprating = median_team_ratings[cur_game_oppidx]
					cur_game_off_opprating = median_team_off_ratings[cur_game_oppidx]
					cur_game_def_opprating = median_team_def_ratings[cur_game_oppidx]
				else:
					cur_game_opprating = None
					cur_game_off_opprating = None
					cur_game_def_opprating = None
				cur_game_date = unplayed_games_date[cur_game_id]
				# Store data in the team data structure
				cur_game_output_data = {'Season': cur_game_season, 'Team': team_id, 'TeamName': team_name, 'TeamRating': median_team_ratings[team_idx], 'TeamOffenseRating': median_team_off_ratings[team_idx], 'TeamDefenseRating': median_team_def_ratings[team_idx], 'Opponent': cur_game_oppid, 'OpponentName': cur_game_oppname, 'OpponentRating': cur_game_opprating, 'OpponentOffenseRating': cur_game_off_opprating, 'OpponentDefenseRating': cur_game_def_opprating, 'IsPreseason': cur_game_is_preseason, 'IsNeutralSite': cur_game_is_neutral_site, 'IsHomeGame': cur_game_is_home_game, 'IsAwayGame': cur_game_is_away_game, 'Date': cur_game_date.strftime('%Y-%m-%d'), 'IsConferenceGame': cur_game_conference_game, 'TeamConference': cur_game_team_conference, 'TeamDivision': cur_game_team_division, 'OpponentConference': cur_game_opp_conference, 'OpponentDivision': cur_game_opp_division, 'Week': cur_game_week, 'WeekString': cur_game_weekstring}
				output_data['TeamRatings'][team_id]['FutureSchedule'].append(cur_game_output_data)
				# Only add the game to the (redundant) list of future games if the team is the home team, so we don't add duplicate games
				if cur_game_homeid == team_id:
					output_data['FutureSchedule'].append({'Season': cur_game_season, 'HomeID': team_id, 'HomeName': team_name, 'HomeRating': median_team_ratings[team_idx], 'HomeOffenseRating': median_team_off_ratings[team_idx], 'HomeDefenseRating': median_team_def_ratings[team_idx], 'AwayID': cur_game_oppid, 'AwayName': cur_game_oppname, 'AwayRating': cur_game_opprating, 'AwayOffenseRating': cur_game_off_opprating, 'AwayDefenseRating': cur_game_def_opprating, 'IsPreseason': cur_game_is_preseason, 'IsNeutralSite': cur_game_is_neutral_site, 'Date': cur_game_date.strftime('%Y-%m-%d'), 'IsConferenceGame': cur_game_conference_game, 'HomeConference': cur_game_team_conference, 'HomeDivision': cur_game_team_division, 'AwayConference': cur_game_opp_conference, 'AwayDivision': cur_game_opp_division, 'Week': cur_game_week, 'WeekString': cur_game_weekstring})

		# Store data about the past games so that, while redundant, makes it simpler to postprocess the data
		for cur_game_id in network_game_id_list:
			include_game = True
			cur_game_homeid = all_team_list[input_games_home_listid[cur_game_id]]
			cur_game_awayid = all_team_list[input_games_away_listid[cur_game_id]]
			if (cur_game_homeid != team_id) and (cur_game_awayid != team_id):
				include_game = False
			cur_game_is_preseason = False
			if use_preseason_var:
				cur_game_is_preseason = input_games_is_preseason[cur_game_id]
				if cur_game_is_preseason and (preseason_weight == 0):
					include_game = False
				if (cur_game_season < season_to_rank) and (prior_preseason_weight == 0) and cur_game_is_preseason:
					include_game = False
			cur_game_season = input_games_season[cur_game_id]
			# Only include games from the current season or, if the prior season has any weight, games from the prior season
			if (cur_game_season < season_to_rank) and (prior_season_weight == 0):
				include_game = False
			if include_game:
				# Check about the site of the game and if it's within the conference or division
				cur_game_is_neutral_site = input_games_is_neutral_site[cur_game_id]
				cur_game_conference_game = input_games_conference_game[cur_game_id]
				cur_game_is_home_game = False
				cur_game_is_away_game = False
				cur_game_week = input_games_week[cur_game_id]
				cur_game_weekstring = input_games_weekstring[cur_game_id]
				# Make sure home and away are ordered correctly
				if cur_game_homeid == team_id:
					cur_game_oppid = cur_game_awayid
					cur_game_opplistid = input_games_away_listid[cur_game_id]
					if not cur_game_is_neutral_site:
						cur_game_is_home_game = True
					cur_game_team_score = input_games_home_score[cur_game_id]
					cur_game_opp_score = input_games_away_score[cur_game_id]
					cur_game_team_conference = input_games_home_conference[cur_game_id]
					cur_game_team_division = input_games_home_division[cur_game_id]
					cur_game_opp_conference = input_games_away_conference[cur_game_id]
					cur_game_opp_division = input_games_away_division[cur_game_id]
				if cur_game_awayid == team_id:
					cur_game_oppid = cur_game_homeid
					cur_game_opplistid = input_games_home_listid[cur_game_id]
					if not cur_game_is_neutral_site:
						cur_game_is_away_game = True
					cur_game_team_score = input_games_away_score[cur_game_id]
					cur_game_opp_score = input_games_home_score[cur_game_id]
					cur_game_team_conference = input_games_away_conference[cur_game_id]
					cur_game_team_division = input_games_away_division[cur_game_id]
					cur_game_opp_conference = input_games_home_conference[cur_game_id]
					cur_game_opp_division = input_games_home_division[cur_game_id]
				# Get data about the opponent
				cur_game_oppname = team_id_name_links[cur_game_oppid]
				cur_game_opprating = median_team_ratings[teams_in_network_reverse[cur_game_opplistid]]
				cur_game_off_opprating = median_team_off_ratings[teams_in_network_reverse[cur_game_opplistid]]
				cur_game_def_opprating = median_team_def_ratings[teams_in_network_reverse[cur_game_opplistid]]
				cur_game_date = input_games_date[cur_game_id]
				cur_game_output_data = {'Season': cur_game_season, 'Team': team_id, 'TeamName': team_name, 'TeamRating': median_team_ratings[team_idx], 'TeamOffenseRating': median_team_off_ratings[team_idx], 'TeamDefenseRating': median_team_def_ratings[team_idx], 'Opponent': cur_game_oppid, 'OpponentName': cur_game_oppname, 'OpponentRating': cur_game_opprating, 'OpponentOffenseRating': cur_game_off_opprating, 'OpponentDefenseRating': cur_game_def_opprating, 'IsPreseason': cur_game_is_preseason, 'IsNeutralSite': cur_game_is_neutral_site, 'IsHomeGame': cur_game_is_home_game, 'IsAwayGame': cur_game_is_away_game, 'TeamScore': cur_game_team_score, 'OpponentScore': cur_game_opp_score, 'Date': cur_game_date.strftime('%Y-%m-%d'), 'IsConferenceGame': cur_game_conference_game, 'TeamConference': cur_game_team_conference, 'TeamDivision': cur_game_team_division, 'OpponentConference': cur_game_opp_conference, 'OpponentDivision': cur_game_opp_division, 'Week': cur_game_week, 'WeekString': cur_game_weekstring}
				output_data['TeamRatings'][team_id]['PastSchedule'].append(cur_game_output_data)
				# Only do this if the team is the home team, so we don't add duplicate games to the (redundant) master list of completed games
				if cur_game_homeid == team_id:
					output_data['PastSchedule'].append({'Season': cur_game_season, 'HomeID': team_id, 'HomeName': team_name, 'HomeRating': median_team_ratings[team_idx], 'HomeOffenseRating': median_team_off_ratings[team_idx], 'HomeDefenseRating': median_team_def_ratings[team_idx], 'AwayID': cur_game_oppid, 'AwayName': cur_game_oppname, 'AwayRating': cur_game_opprating, 'AwayOffenseRating': cur_game_off_opprating, 'AwayDefenseRating': cur_game_def_opprating, 'IsPreseason': cur_game_is_preseason, 'IsNeutralSite': cur_game_is_neutral_site, 'HomeScore': cur_game_team_score, 'AwayScore': cur_game_opp_score, 'Date': cur_game_date.strftime('%Y-%m-%d'), 'IsConferenceGame': cur_game_conference_game, 'HomeConference': cur_game_team_conference, 'HomeDivision': cur_game_team_division, 'AwayConference': cur_game_opp_conference, 'AwayDivision': cur_game_opp_division, 'Week': cur_game_week, 'WeekString': cur_game_weekstring})

	# Output the data
	file_handle = open(output_file_name, 'w')
	if file_handle is not None:
		json.dump(output_data, file_handle)
		file_handle.close()

if __name__ == '__main__':
	main()

