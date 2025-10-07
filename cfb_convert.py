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

import csv
import datetime
import json
import sys

def main ():
	# Get the parameters from the command line
	if len(sys.argv) < 3:
		print('Usage: '+sys.argv[0]+' <output file> <input file> [additional input files]')
		sys.exit(1)

	output_file = sys.argv[1].strip()
	game_data = {}
	game_count = 0

	# Read in each CSV file with games and load the desired columns
	for param_index in range(2, len(sys.argv), 1):
		input_file = sys.argv[param_index].strip()

		# Read the list of games from the CSV file
		input_handle = open(input_file, newline = '', encoding = 'utf-8-sig')
		reader = csv.reader(input_handle)
		input_data = list(reader)
		input_handle.close()
		input_data_header = input_data.pop(0)

		# Get a list of column IDs for useful variables
		input_data_gameid_col = input_data_header.index('Id')
		input_data_iscompleted_col = input_data_header.index('Completed')
		input_data_isneutralsite_col = input_data_header.index('NeutralSite')
		input_data_isconferencegame_col = input_data_header.index('ConferenceGame')
		input_data_homeid_col = input_data_header.index('HomeId')
		input_data_awayid_col = input_data_header.index('AwayId')
		input_data_hometeam_col = input_data_header.index('HomeTeam')
		input_data_awayteam_col = input_data_header.index('AwayTeam')
		input_data_homepoints_col = input_data_header.index('HomePoints')
		input_data_awaypoints_col = input_data_header.index('AwayPoints')
		input_data_homediv_col = input_data_header.index('HomeClassification')
		input_data_awaydiv_col = input_data_header.index('AwayClassification')
		input_data_homeconf_col = input_data_header.index('HomeConference')
		input_data_awayconf_col = input_data_header.index('AwayConference')
		input_data_week_col = input_data_header.index('Week')
		input_data_season_col = input_data_header.index('Season')
		input_data_utcdate_col = input_data_header.index('StartDate')
		input_data_seasontype_col = input_data_header.index('SeasonType')

		# Loop through each row and then populate the output data structure with the data on that row
		for game_row in input_data:
			cur_game_ispreseason = False
			cur_game_league = 'NCAA'
			cur_game_gameid = game_row[input_data_gameid_col]
			try:
				cur_game_season = int(game_row[input_data_season_col].strip())
			except:
				cur_game_season = None
			if game_row[input_data_iscompleted_col] == 'true':
				cur_game_iscompleted = True
			else:
				cur_game_iscompleted = False
			if game_row[input_data_isconferencegame_col] == 'true':
				cur_game_isconferencegame = True
			else:
				cur_game_isconferencegame = False
			if game_row[input_data_isneutralsite_col] == 'true':
				cur_game_isneutralsite = True
			else:
				cur_game_isneutralsite = False
			if game_row[input_data_seasontype_col] == 'postseason':
				cur_game_ispostseason = True
			else:
				cur_game_ispostseason = False
			cur_game_homeid = game_row[input_data_homeid_col]
			cur_game_awayid = game_row[input_data_awayid_col]
			cur_game_homename = game_row[input_data_hometeam_col]
			cur_game_awayname = game_row[input_data_awayteam_col]
			cur_game_homescorestr = game_row[input_data_homepoints_col]
			cur_game_awayscorestr = game_row[input_data_awaypoints_col]
			# Use the score as an extra verification to determine if the game is completed, otherwise set the scores to None
			if cur_game_iscompleted:
				try:
					cur_game_homescore = int(cur_game_homescorestr.strip())
					cur_game_awayscore = int(cur_game_awayscorestr.strip())
				except:
					cur_game_homescore = None
					cur_game_awayscore = None
			else:
				cur_game_homescore = None
				cur_game_awayscore = None
			cur_game_homediv = game_row[input_data_homediv_col]
			cur_game_awaydiv = game_row[input_data_awaydiv_col]
			cur_game_homeconf = game_row[input_data_homeconf_col]
			cur_game_awayconf = game_row[input_data_awayconf_col]
			cur_game_week = game_row[input_data_week_col]
			# Try to convert the week to a number
			try:
				cur_game_weeknumber = int(cur_game_week)
			except:
				cur_game_weeknumber = None
			# Parse the date of the game
			cur_game_datetimestr = game_row[input_data_utcdate_col]
			cur_game_datestr = cur_game_datetimestr.split('T')[0]
			cur_game_year = int(cur_game_datestr.split('-')[0])
			cur_game_month = int(cur_game_datestr.split('-')[1])
			cur_game_day = int(cur_game_datestr.split('-')[2])
			cur_game_datevar = datetime.date(cur_game_year, cur_game_month, cur_game_day)
			cur_game_epoch_day = (cur_game_datevar - datetime.date(1970, 1, 1)).days
			# Store the data in a data structure and append it to the game data list
			output_row = {'GameID': cur_game_gameid, 'IsPreseason': cur_game_ispreseason, 'IsPostseason': cur_game_ispostseason, 'IsNeutralSite': cur_game_isneutralsite, 'League': cur_game_league, 'Season': cur_game_season, 'IsCompleted': cur_game_iscompleted, 'IsConferenceGame': cur_game_isconferencegame, 'HomeID': cur_game_homeid, 'AwayID': cur_game_awayid, 'HomeName': cur_game_homename, 'AwayName': cur_game_awayname, 'HomeScore': cur_game_homescore, 'AwayScore': cur_game_awayscore, 'HomeDivision': cur_game_homediv, 'AwayDivision': cur_game_awaydiv, 'HomeConference': cur_game_homeconf, 'AwayConference': cur_game_awayconf, 'Year': cur_game_year, 'Month': cur_game_month, 'Day': cur_game_day, 'EpochDay': cur_game_epoch_day, 'DateTimeString': cur_game_datetimestr, 'WeekString': cur_game_week, 'Week': cur_game_weeknumber}
			game_count = game_count + 1
			game_data[game_count] = output_row

	# Output a JSON with the data for all the games
	file_handle = open(output_file, 'w')
	if file_handle is not None:
		json.dump(game_data, file_handle)
		file_handle.close()

if __name__ == '__main__':
	main()

