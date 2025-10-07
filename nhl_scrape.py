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

import bs4
import collections
import datetime
import json
import io
import pandas as pd
import re
import requests
import sys
import time
import warnings

# Suppress irrelevant warnings when parsing
def suppress_bs4_warnings ():
	bs4.warnings.filterwarnings('ignore', category = bs4.MarkupResemblesLocatorWarning)

# Attempt to download a page in a robust manner
def retrieve_page (request_url, request_delay_time = 5, max_requests = 10):
	downloaded = False
	end_requests = False
	request_count = 0
	while end_requests == False:
		server_response = requests.get(request_url)
		request_count = request_count + 1
		# If the page is downloaded successfully
		if 200 <= server_response.status_code <= 299:
			end_requests = True
			downloaded = True
		# The maximum number of requests has been reached
		elif request_count >= max_requests:
			end_requests = True
		# There's an error, so wait and retry
		else:
			time.sleep(request_delay_time)
	if not downloaded:
		warnings.warn(('Error downloading %s, code %d') % (request_url, server_response.status_code))
		return None

	return server_response

# Get a list of parsed Sports Reference tables from HTML text
def get_parsed_sref_tables (htmltext, delete_headers = True):
	# Parse the HTML text
	soup = bs4.BeautifulSoup(htmltext.replace('&nbsp;', ' '), features='html.parser')

	# Get a list of tables that match the required classes
	table_list = soup.find_all('table', class_ = ['sortable', 'stats_table'])
	if table_list is None:
		table_list = []

	# Scan through the comments and convert comments that contain HTML to actual HTML
	for comment_match in soup.find_all(string = lambda string:isinstance(string, bs4.Comment)):
		parent = comment_match.parent
		reject_comment = False
		while parent:
			if parent.name in ['script', 'style']:
				reject_comment = True
			parent = parent.parent
		if not reject_comment:
			comment_match_soup = bs4.BeautifulSoup(str(comment_match), 'html.parser')
			if len(comment_match_soup.find_all()) > 0:
				table_list_result = comment_match_soup.find_all('table', class_ = ['sortable', 'stats_table'])
				if table_list_result is not None:
					table_list.extend(table_list_result)

	# Loop through each table in the list and delete all extra headers
	if delete_headers:
		if table_list is not None:
			for this_table in table_list:
				header_list = this_table.find_all('tr', class_ = ['thead'])
				if header_list is not None:
					for cur_header in header_list:
						cur_header.extract()
				header_list = this_table.find_all('tr', class_ = ['over_header'])
				if header_list is not None:
					for cur_header in header_list:
						cur_header.extract()

	# Return the list of tables
	return table_list

# Parse a Sports Reference table as if it's a schedule
def parse_sref_schedule_table (this_table, this_season = None, season_new_column = 'Season'):
	# If we can get the table ID, then retrieve it
	if this_table.has_attr('id'):
		this_table_id = this_table['id'].strip()
	else:
		this_table_id = None
	# If we can retrieve the table ID, then we should parse it
	if this_table_id is not None:
		this_table_data = pd.read_html(io.StringIO(str(this_table)), extract_links = 'body')[0]
		# We need to iterate through the dataframe columns and rows to extract player names and team IDs, remove tuples, and put an asterisk prior to teams and players that aren't real
		for column_name in list(set(this_table_data.columns).intersection(['Visitor', 'Home'])):
			this_table_data[column_name + '.Name'] = pd.Series(dtype = 'string')
		nrows = len(this_table_data)
		ncols = len(this_table_data.columns)
		for i in range(0, nrows, 1):
			for j in range(0, ncols, 1):
				cdata = this_table_data.iat[i, j]
				# Test if the data in the cell is iterable and not a string
				if (isinstance(cdata, collections.abc.Iterable)) and (not isinstance(cdata, str)):
					cdata0 = cdata[0]
					cdata1 = cdata[1]
					isiter = True
				else:
					cdata0 = cdata
					cdata1 = cdata
					isiter = False
				# Handle if it's a column for teams, try to extract the team ID from the URL
				if ['Visitor', 'Home'].count(this_table_data.columns[j]) > 0:
					if isiter:
						if cdata1 is not None:
							try:
								this_table_data.iat[i, j] = cdata1.split('/')[-2].split('.')[0]
							except:
								this_table_data.iat[i, j] = '*' + str(cdata0)
							this_table_data.loc[this_table_data.index[i], this_table_data.columns[j] + '.Name'] = str(cdata0)
						else:
							this_table_data.iat[i, j] = '*' + str(cdata0)
							this_table_data.loc[this_table_data.index[i], this_table_data.columns[j] + '.Name'] = str(cdata0)
					else:
						this_table_data.iat[i, j] = '*' + str(cdata0)
						this_table_data.loc[this_table_data.index[i], this_table_data.columns[j] + '.Name'] = str(cdata0)
				# Otherwise, just remove the tuple
				else:
					this_table_data.iat[i, j] = str(cdata0)

		# Next, go through and remove extra rows like league averages and totals, and clear unwanted rows (multiple rows with stats from different teams, league averages/totals, etc...)
		drop_rows = []
		if list(this_table_data.columns).count('Home') > 0:
			home_column = list(this_table_data.columns).index('Home')
		else:
			home_column = None
		if list(this_table_data.columns).count('Visitor') > 0:
			visitor_column = list(this_table_data.columns).index('Visitor')
		else:
			visitor_column = None
		for i in range(0, nrows, 1):
			row_index = list(this_table_data.index).index(i)
			del_row = False
			if this_season is not None:
				this_table_data.at[row_index, season_new_column] = this_season
			if (home_column is None) or (visitor_column is None) or (len(str(this_table_data.iat[i, home_column]).strip()) == 0) or (len(str(this_table_data.iat[i, visitor_column]).strip()) == 0):
				del_row = True
			# Not a schedule table, or a blank row in a schedule table, then delete
			if del_row:
				drop_rows.append(row_index)

		# Set the season column type to integer
		if this_season is not None:
			if nrows > 0:
				this_table_data[season_new_column] = this_table_data[season_new_column].astype(int)
		# Drop rows that should be dropped
		if len(drop_rows) > 0:
			this_table_data = this_table_data.drop(index = drop_rows)
	# We don't really need this, but set the table data to None if there's no identifier
	else:
		this_table_data = None

	return this_table_id, this_table_data

# Extract data from the API schedule table
def parse_api_schedule_row(schedule_row, season, franchise_lookup, league = 'NHL'):
	offset_hours = int(schedule_row['venueUTCOffset'].strip().split(':')[0])
	offset_minutes = int(schedule_row['venueUTCOffset'].strip().split(':')[1])
	if schedule_row['venueUTCOffset'].strip()[0] == '-':
		offset_minutes = -offset_minutes
	game_date_parse = datetime.datetime.fromisoformat(schedule_row['startTimeUTC'][:-1]).date() + datetime.timedelta(hours = offset_hours, minutes = offset_minutes)
	game_year = game_date_parse.year
	game_month = game_date_parse.month
	game_day = game_date_parse.day
	epoch_day = (datetime.date(game_year, game_month, game_day) - datetime.date(1970, 1, 1)).days
	row_data = {}
	# Check if the game is finished and get information about its status
	if schedule_row['gameState'] == 'FINAL':
		is_finished = True
		home_score = schedule_row['homeTeam']['score']
		away_score = schedule_row['awayTeam']['score']
		row_data['OvertimeStatus'] = schedule_row['gameOutcome']['lastPeriodType']
		if row_data['OvertimeStatus'] == 'REG':
			row_data['Overtime'] = False
			row_data['Shootout'] = False
		elif re.match('.*OT', row_data['OvertimeStatus']):
			row_data['Overtime'] = True
			row_data['Shootout'] = False
		elif row_data['OvertimeStatus'] == 'SO':
			row_data['Overtime'] = True
			row_data['Shootout'] = True
		else:
			row_data['Overtime'] = None
			row_data['Shootout'] = None
	else:
		is_finished = False
		home_score = None
		away_score = None
		row_data['OvertimeStatus'] = ''
		row_data['Overtime'] = False
		row_data['Shootout'] = False
	# Set the game type appropriately
	if schedule_row['gameType'] == 1:
		row_data['IsPreseason'] = True
		row_data['IsPostseason'] = False
	elif schedule_row['gameType'] == 2:
		row_data['IsPreseason'] = False
		row_data['IsPostseason'] = False
	elif schedule_row['gameType'] == 3:
		row_data['IsPreseason'] = False
		row_data['IsPostseason'] = True
	else:
		row_data['IsPreseason'] = None
		row_data['IsPostseason'] = None
	# Set venue information
	row_data['IsNeutralSite'] = schedule_row['neutralSite']
	row_data['Venue'] = schedule_row['venue']['default']
	row_data['Attendance'] = ''
	row_data['Notes'] = ''
	row_data['Year'] = game_year
	row_data['Month'] = game_month
	row_data['Day'] = game_day
	row_data['EpochDay'] = epoch_day
	row_data['Season'] = season

	row_data['AwayScore'] = away_score
	row_data['HomeScore'] = home_score
	row_data['IsCompleted'] = is_finished

	# Set some default information that is required
	row_data['Week'] = None
	row_data['WeekString'] = None
	row_data['League'] = league

	row_data['HomeName'] = schedule_row['homeTeam']['placeName']['default'].strip() + ' ' + schedule_row['homeTeam']['commonName']['default'].strip()
	row_data['AwayName'] = schedule_row['awayTeam']['placeName']['default'].strip() + ' ' + schedule_row['awayTeam']['commonName']['default'].strip()

	# Try to match the home team with the franchise table
	input_id = schedule_row['homeTeam']['abbrev']
	input_name = row_data['HomeName']
	cur_franchise = None
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == input_id))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['FranchiseID'] == input_id))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamName'] == input_name))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['FranchiseName'] == input_name))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is not None:
		row_data['HomeID'] = cur_franchise['FranchiseID']
		row_data['HomeTeamID'] = cur_franchise['TeamID']
		row_data['HomeFranchiseName'] = cur_franchise['FranchiseName']
		row_data['HomeName'] = cur_franchise['TeamName']
		row_data['HomeConference'] = cur_franchise['Conference']
		row_data['HomeDivision'] = cur_franchise['Division']
	else:
		row_data['HomeID'] = None
		row_data['HomeTeamID'] = input_id
		row_data['HomeFranchiseName'] = input_name
		row_data['HomeName'] = None
		row_data['HomeConference'] = None
		row_data['HomeDivision'] = None

	# Try to match the away team with the franchise table
	input_id = schedule_row['awayTeam']['abbrev']
	input_name = row_data['AwayName']
	cur_franchise = None
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == input_id))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['FranchiseID'] == input_id))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamName'] == input_name))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is None:
		franchise_data = [x for x in franchise_lookup if ((x['Season'] == season) and (x['FranchiseName'] == input_name))]
		if len(franchise_data) > 0:
			cur_franchise = franchise_data[0]
	if cur_franchise is not None:
		row_data['AwayID'] = cur_franchise['FranchiseID']
		row_data['AwayTeamID'] = cur_franchise['TeamID']
		row_data['AwayFranchiseName'] = cur_franchise['FranchiseName']
		row_data['AwayName'] = cur_franchise['TeamName']
		row_data['AwayConference'] = cur_franchise['Conference']
		row_data['AwayDivision'] = cur_franchise['Division']
	else:
		row_data['AwayID'] = None
		row_data['AwayTeamID'] = input_id
		row_data['AwayFranchiseName'] = input_name
		row_data['AwayName'] = None
		row_data['AwayConference'] = None
		row_data['AwayDivision'] = None
	return(row_data)

# Extract data from a row of the schedule table
def parse_schedule_row (row, season, franchise_lookup, is_postseason = False, league = 'NHL'):
	game_date_parse = datetime.datetime.strptime(row['Date'].strip(), '%Y-%m-%d')
	game_year = game_date_parse.year
	game_month = game_date_parse.month
	game_day = game_date_parse.day
	# Use the presence or absence of score data to determine if the game is finished or not
	if (len(row['G'].strip()) == 0) or (len(row['G.1'].strip()) == 0):
		is_finished = False
	else:
		is_finished = True
	epoch_day = (datetime.date(game_year, game_month, game_day) - datetime.date(1970, 1, 1)).days
	# Store the data about the participants in a data structure, much of which is pulled from the franchise data
	row_data = {}
	row_data['Season'] = season
	row_data['HomeTeamID'] = row['Home'].strip()
	row_data['AwayTeamID'] = row['Visitor'].strip()
	home_franchise = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == row_data['HomeTeamID']))][0]
	away_franchise = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == row_data['AwayTeamID']))][0]
	row_data['HomeID'] = home_franchise['FranchiseID']
	row_data['AwayID'] = away_franchise['FranchiseID']
	row_data['HomeFranchiseName'] = home_franchise['FranchiseName']
	row_data['AwayFranchiseName'] = away_franchise['FranchiseName']
	row_data['HomeConference'] = home_franchise['Conference']
	row_data['HomeDivision'] = home_franchise['Division']
	row_data['AwayConference'] = away_franchise['Conference']
	row_data['AwayDivision'] = away_franchise['Division']
	row_data['HomeName'] = row['Home.Name'].strip()
	row_data['AwayName'] = row['Visitor.Name'].strip()
	# Get information specific to the game like the score and overtime status
	row_data['IsCompleted'] = is_finished
	if is_finished:
		row_data['AwayScore'] = int(row['G'].strip())
		row_data['HomeScore'] = int(row['G.1'].strip())
	else:
		row_data['AwayScore'] = None
		row_data['HomeScore'] = None
	row_data['OvertimeStatus'] = row['Unnamed: 6'].strip()
	row_data['GameLengthString'] = row['LOG'].strip()
	if re.match('.*OT', row_data['OvertimeStatus']):
		row_data['Overtime'] = True
		row_data['Shootout'] = False
	elif row_data['OvertimeStatus'] == 'SO':
		row_data['Overtime'] = True
		row_data['Shootout'] = True
	else:
		row_data['Overtime'] = False
		row_data['Shootout'] = False
	row_data['Attendance'] = row['Att.'].strip()
	row_data['Notes'] = row['Notes'].strip()
	row_data['Year'] = game_year
	row_data['Month'] = game_month
	row_data['Day'] = game_day
	row_data['EpochDay'] = epoch_day
	# If the notes column of the table begins with "at" and contains more than one word, assume it means that the game is really a neutral site game
	notes_split = row_data['Notes'].split()
	if (len(notes_split) > 1) and (notes_split[0] == 'at'):
		row_data['IsNeutralSite'] = True
	else:
		row_data['IsNeutralSite'] = False
	row_data['IsPreseason'] = False
	row_data['IsPostseason'] = is_postseason
	row_data['Week'] = None
	row_data['WeekString'] = None
	row_data['League'] = league
	return row_data

def get_nhlapi_schedule(preseason_start, season_start, request_delay = 5):
	schedule_rows = []
	current_date = preseason_start
	while current_date < season_start:
		date_string = current_date.strftime('%Y-%m-%d')
		nhlapi_url = 'https://api-web.nhle.com/v1/schedule/' + date_string
		nhlapi_page = retrieve_page(nhlapi_url)
		if nhlapi_page is None:
			warnings.warn('Cannot retrieve API data for date ' + date_string)
		else:
			try:
				nhlapi_data = json.loads(nhlapi_page.text)
				for cur_day in nhlapi_data['gameWeek']:
					for cur_game in cur_day['games']:
						if cur_game['gameType'] == 1:
							schedule_rows.append(cur_game)
				current_date = datetime.datetime.strptime(nhlapi_data['nextStartDate'].strip(), '%Y-%m-%d').date()
			except:
				warnings.warn('Error parsing game data for date beginning ' + date_string)
				nhlapi_data = None
				current_date = current_date + datetime.timedelta(days = 7)
		time.sleep(request_delay)
	return schedule_rows

def main ():
	# Get the parameters from the command line
	if len(sys.argv) < 5:
		print('Usage: '+sys.argv[0]+' <franchise file> <start season> <end season> <output file>')
		sys.exit(1)
	try:
		start_season = int(sys.argv[2].strip())
		end_season = int(sys.argv[3].strip())
	except:
		print('Invalid season')
		sys.exit(1)
	franchise_file = sys.argv[1].strip()
	input_handle = open(franchise_file, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	franchise_data = json.load(input_handle)
	input_handle.close()
	output_file = sys.argv[4].strip()
	request_delay = 5
	suppress_bs4_warnings()
	game_data = {}
	game_count = 0
	# Loop through the seasons
	for current_season in range(start_season, end_season + 1, 1):
		franchise_lookup = [x for x in franchise_data if x['Season'] == current_season]
		season_fail = False
		season_table = None
		postseason_table = None
		wha_season_table = None
		wha_postseason_table = None
		# Load the page with NHL games for the season, then extract the regular season and postseason tables
		if current_season >= 1918:
			season_url = (('https://www.hockey-reference.com/leagues/NHL_%d_games.html') % (current_season))
			season_page = retrieve_page(season_url)
			if season_page is None:
				season_fail = True
			else:
				season_data = get_parsed_sref_tables(season_page.text)
				for html_table in season_data:
					parsed_table = parse_sref_schedule_table(html_table)
					if parsed_table[0] == 'games':
						season_table = parsed_table[1]
					elif parsed_table[0] == 'games_playoffs':
						postseason_table = parsed_table[1]
				if season_table is None:
					season_fail = True
			time.sleep(request_delay)
		else:
			season_page = None
		# If there was a WHA season, extract the regular season and postseason tables
		if (current_season >= 1973) and (current_season <= 1979):
			wha_season_url = (('https://www.hockey-reference.com/leagues/WHA_%d_games.html') % (current_season))
			wha_season_page = retrieve_page(wha_season_url)
			if wha_season_page is None:
				season_fail = True
			else:
				wha_season_data = get_parsed_sref_tables(wha_season_page.text)
				for html_table in afl_season_data:
					parsed_table = parse_sref_schedule_table(html_table)
					if parsed_table[0] == 'games':
						wha_season_table = parsed_table[1]
					elif parsed_table[0] == 'games_playoffs':
						wha_postseason_table = parsed_table[1]
				if wha_season_table is None:
					season_fail = True
			time.sleep(request_delay)
		# If we can't download the season table, issue a warning, but still try to parse the season
		if season_fail:
			warnings.warn(('Error downloading data for season %d') % (current_season))
		# Find the earliest date of the regular season, if possible
		if season_table is not None:
			game_dates = []
			for row_idx, row in season_table.iterrows():
				game_dates.append(datetime.datetime.strptime(row['Date'].strip(), '%Y-%m-%d').date())
			if len(game_dates) > 0:
				season_start = min(game_dates)
				# Assume the preseason could start as early as 60 days before the regular season
				preseason_start = season_start - datetime.timedelta(days = 60)
				preseason_schedule = get_nhlapi_schedule(preseason_start, season_start)
				# Go through the data, parse it, and add it to the JSON
				for schedule_row in preseason_schedule:
					game_count = game_count + 1
					game_data[game_count] = parse_api_schedule_row(schedule_row, current_season, franchise_lookup)
		# Parse the NHL regular season table (if it exists)
		if season_table is not None:
			for row_idx, row in season_table.iterrows():
				game_count = game_count + 1
				game_data[game_count] = parse_schedule_row(row, current_season, franchise_lookup, is_postseason = False)
		# Parse the NHL postseason table (if it exists)
		if postseason_table is not None:
			for row_idx, row in postseason_table.iterrows():
				game_count = game_count + 1
				game_data[game_count] = parse_schedule_row(row, current_season, franchise_lookup, is_postseason = True)
		# Parse the WHA regular season table (if it exists)
		if wha_season_table is not None:
			for row_idx, row in wha_season_table.iterrows():
				game_count = game_count + 1
				game_data[game_count] = parse_schedule_row(row, current_season, franchise_lookup, is_postseason = False, league = 'WHA')
		# Parse the WHA postseason table (if it exists)
		if wha_postseason_table is not None:
			for row_idx, row in wha_postseason_table.iterrows():
				game_count = game_count + 1
				game_data[game_count] = parse_schedule_row(row, current_season, franchise_lookup, is_postseason = True, league = 'WHA')
		# Store the results in a JSON file
		file_handle = open(output_file, 'w')
		if file_handle is not None:
			json.dump(game_data, file_handle)
			file_handle.close()

if __name__ == '__main__':
	main()

