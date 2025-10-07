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
import urllib
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
def get_parsed_sref_month_urls (htmltext, baseurl):
	# Parse the HTML text
	soup = bs4.BeautifulSoup(htmltext.replace('&nbsp;', ' '), features='html.parser')

	# Get a list of tables that match the required classes
	div_list = soup.find_all('div', class_ = ['filter'])
	if div_list is None:
		div_list = []

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
				div_list_result = comment_match_soup.find_all('div', class_ = ['filter'])
				if div_list_result is not None:
					div_list.extend(div_list_result)

	# Now get a list of months from the links
	schedule_url_list = []
	for div_node in div_list:
		for a_node in div_node.find_all('a'):
			if a_node.has_attr('href'):
				new_url = urllib.parse.urljoin(baseurl, a_node.get('href'))
				schedule_url_list.append(new_url)
	return schedule_url_list

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

# Parse a Sports Reference table as if it's standings
def parse_sref_standings_table (this_table, split_division_name = False):
	# If we can get the table ID, then retrieve it
	if this_table.has_attr('id'):
		this_table_id = this_table['id'].strip()
	else:
		this_table_id = None
	standings_data = {}
	# If we can retrieve the table ID, then we should parse it
	if this_table_id is not None:
		table_caption = this_table.find('caption')
		if (table_caption.text.strip() == 'Division Standings Table') or (table_caption.text.strip() == 'Table'):
			this_table_data = pd.read_html(io.StringIO(str(this_table)), extract_links = 'body')[0]
			column0 = this_table_data.columns.tolist()[0]
			valid_table = False
			if (column0 == 'Team'):
				standings_conference = None
				valid_table = True
			elif re.match('.* Conference$', column0):
				conference_split = column0.split()
				if (len(conference_split) > 0) and (conference_split[-1] == 'Conference'):
					standings_conference = ' '.join(conference_split[:-1])
					valid_table = True
			standings_division = None
			if valid_table:
				for idx, row in this_table_data.iterrows():
					if row[column0][1] is None:
						standings_division = row[column0][0].strip()
						if split_division_name and (standings_division.split()[-1] == 'Division') and (len(standings_division.split()) > 1):
							standings_division = ' '.join(standings_division.split()[:-1])
						if len(standings_division.strip()) == 0:
							standings_division = None
					else:
						try:
							team_id = row[column0][1].split('/')[-2].split('.')[0]
							if team_id is not None:
								standings_data[team_id] = {'Conference': standings_conference, 'Division': standings_division}
						except:
							pass
	return standings_data

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
		for column_name in list(set(this_table_data.columns).intersection(['Visitor/Neutral', 'Home/Neutral'])):
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
				if ['Visitor/Neutral', 'Home/Neutral'].count(this_table_data.columns[j]) > 0:
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
		if list(this_table_data.columns).count('Home/Neutral') > 0:
			home_column = list(this_table_data.columns).index('Home/Neutral')
		else:
			home_column = None
		if list(this_table_data.columns).count('Visitor/Neutral') > 0:
			visitor_column = list(this_table_data.columns).index('Visitor/Neutral')
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

def parse_schedule_row (row, season, division_lookup, franchise_lookup, postseason_start, league = 'NBA'):
	is_postseason = False
	game_date_parse = datetime.datetime.strptime(row['Date'].strip(), '%a, %b %d, %Y')
	# It might be possible for a postseason game to be played on the same day as a tiebreaker or a play-in game, so check for this and use those entries in the notes column to determine if the same is a regular season or a postseason game (though there's probably a more robust way to do this with box score IDs and parsing URLs)
	if (postseason_start is not None) and (game_date_parse.date() >= postseason_start) and (row['Notes'].strip() != 'Tiebreaker') and (row['Notes'].strip() != 'Play-In Game'):
		is_postseason = True
	game_year = game_date_parse.year
	game_month = game_date_parse.month
	game_day = game_date_parse.day
	# Use the points columns to determine if there's a score, and therefore if the game has finished or instead will be played in the future
	if (len(row['PTS'].strip()) == 0) or (len(row['PTS.1'].strip()) == 0):
		is_finished = False
	else:
		is_finished = True
	epoch_day = (datetime.date(game_year, game_month, game_day) - datetime.date(1970, 1, 1)).days
	# Store data about the franchises and teams in the data structure about the game, mostly using data from the division data and the franchise table
	row_data = {}
	row_data['Season'] = season
	home_team_id = row['Home/Neutral'].strip()
	away_team_id = row['Visitor/Neutral'].strip()
	if len(home_team_id) > 0:
		row_data['HomeTeamID'] = home_team_id
	else:
		row_data['HomeTeamID'] = None
	if len(away_team_id) > 0:
		row_data['AwayTeamID'] = away_team_id
	else:
		row_data['AwayTeamID'] = None
	home_franchise_list = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == row_data['HomeTeamID']))]
	if len(home_franchise_list) > 0:
		home_franchise = home_franchise_list[0]
		row_data['HomeID'] = home_franchise['FranchiseID']
		row_data['HomeFranchiseName'] = home_franchise['FranchiseName']
	else:
		row_data['HomeID'] = None
		row_data['HomeFranchiseName'] = None
	away_franchise_list = [x for x in franchise_lookup if ((x['Season'] == season) and (x['TeamID'] == row_data['AwayTeamID']))]
	if len(away_franchise_list) > 0:
		away_franchise = away_franchise_list[0]
		row_data['AwayID'] = away_franchise['FranchiseID']
		row_data['AwayFranchiseName'] = away_franchise['FranchiseName']
	else:
		row_data['AwayID'] = None
		row_data['AwayFranchiseName'] = None
	if list(division_lookup.keys()).count(row_data['HomeTeamID']) > 0:
		row_data['HomeConference'] = division_lookup[row_data['HomeTeamID']]['Conference']
		row_data['HomeDivision'] = division_lookup[row_data['HomeTeamID']]['Division']
	else:
		row_data['HomeConference'] = None
		row_data['HomeDivision'] = None
	if list(division_lookup.keys()).count(row_data['AwayTeamID']) > 0:
		row_data['AwayConference'] = division_lookup[row_data['AwayTeamID']]['Conference']
		row_data['AwayDivision'] = division_lookup[row_data['AwayTeamID']]['Division']
	else:
		row_data['AwayConference'] = None
		row_data['AwayDivision'] = None
	row_data['HomeName'] = row['Home/Neutral.Name'].strip()
	row_data['AwayName'] = row['Visitor/Neutral.Name'].strip()
	# Get information about overtimes and scores, and parse some of the other columns
	row_data['IsCompleted'] = is_finished
	if is_finished:
		row_data['AwayScore'] = int(row['PTS'].strip())
		row_data['HomeScore'] = int(row['PTS.1'].strip())
	else:
		row_data['AwayScore'] = None
		row_data['HomeScore'] = None
	row_data['OvertimeStatus'] = row['Unnamed: 7'].strip()
	row_data['GameLengthString'] = row['LOG'].strip()
	if re.match('.*OT', row_data['OvertimeStatus']):
		row_data['Overtime'] = True
	else:
		row_data['Overtime'] = False
	row_data['Venue'] = row['Arena'].strip()
	row_data['Attendance'] = row['Attend.'].strip()
	row_data['Notes'] = row['Notes'].strip()
	row_data['Year'] = game_year
	row_data['Month'] = game_month
	row_data['Day'] = game_day
	row_data['EpochDay'] = epoch_day
	notes_split = row_data['Notes'].split()
	# If the notes column begins with "at" and has multiple words, assume that the game is played at a neutral site
	if (len(notes_split) > 1) and (notes_split[0] == 'at'):
		row_data['IsNeutralSite'] = True
	else:
		row_data['IsNeutralSite'] = False
	row_data['IsPreseason'] = False
	row_data['IsPostseason'] = is_postseason
	row_data['Week'] = None
	row_data['WeekString'] = None
	row_data['League'] = league
	# Return the data structure with the row
	return row_data

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
		season_tables = []
		aba_season_tables = []
		# Distinguish between the NBA and BAA name when requesting standings
		if current_season >= 1950:
			standings_url = (('https://www.basketball-reference.com/leagues/NBA_%d_standings.html') % (current_season))
		else:
			standings_url = (('https://www.basketball-reference.com/leagues/BAA_%d_standings.html') % (current_season))
		# Load and parse the NBA/BAA standings table
		standings_page = retrieve_page(standings_url)
		division_lookup = {}
		if standings_page is None:
			season_fail = True
		else:
			standings_data = get_parsed_sref_tables(standings_page.text, delete_headers = False)
			for html_table in standings_data:
				table_lookup_data = parse_sref_standings_table(html_table, split_division_name = True)
				division_lookup = {**division_lookup, **table_lookup_data}
		time.sleep(request_delay)
		# If there should be NBA/BAA data, load the page
		if current_season >= 1947:
			if current_season >= 1950:
				season_url = (('https://www.basketball-reference.com/leagues/NBA_%d_games.html') % (current_season))
			else:
				season_url = (('https://www.basketball-reference.com/leagues/BAA_%d_games.html') % (current_season))
			season_page = retrieve_page(season_url)
			time.sleep(request_delay)
			# Schedules on Basketball Reference are split into monthly pages, so get all the URLs that need to be loaded to obtain a full schedule
			if season_page is None:
				season_fail = True
			else:
				schedule_urls = get_parsed_sref_month_urls(season_page.text, 'https://www.basketball-reference.com/leagues/')
				# Extract the schedule tables
				for schedule_url in schedule_urls:
					schedule_page = retrieve_page(schedule_url)
					if schedule_page is None:
						season_fail = True
					else:
						schedule_data = get_parsed_sref_tables(schedule_page.text)
						for html_table in schedule_data:
							parsed_table = parse_sref_schedule_table(html_table)
							if parsed_table[0] == 'schedule':
								season_tables.append(parsed_table[1])
					time.sleep(request_delay)
				if len(season_tables) == 0:
					season_fail = True
				# There's a separate page that also has a playoff game schedule (but it's separate)
				if current_season >= 1950:
					playoff_url = (('https://www.basketball-reference.com/playoffs/NBA_%d_games.html') % (current_season))
				else:
					playoff_url = (('https://www.basketball-reference.com/playoffs/BAA_%d_games.html') % (current_season))
				# Retrive the playoff data, and obtain the first date when there was a playoff game as a cutoff for the regular season and the playoffs, though there might be a better way to distinguish between playoff and regular season games with box score URLs (possible future improvements)
				playoff_page = retrieve_page(playoff_url)
				playoff_start = None
				if playoff_page is not None:
					playoff_data = get_parsed_sref_tables(playoff_page.text, delete_headers = False)
					for html_table in playoff_data:
						parsed_table = parse_sref_schedule_table(html_table)
						if parsed_table[0] == 'schedule':
							playoff_dates = []
							for idx, row in parsed_table[1].iterrows():
								playoff_dates.append(datetime.datetime.strptime(row['Date'], '%a, %b %d, %Y').date())
							if len(playoff_dates) > 0:
								playoff_start = min(playoff_dates)
			time.sleep(request_delay)
		else:
			season_page = None
		# Everything mostly works the same with ABA data if that is available
		if (current_season >= 1968) and (current_season <= 1976):
			aba_season_url = (('https://www.basketball-reference.com/leagues/ABA_%d_games.html') % (current_season))
			aba_season_page = retrieve_page(aba_season_url)
			time.sleep(request_delay)
			if aba_season_page is None:
				aba_season_fail = True
			else:
				# Schedules on Basketball Reference are split into monthly pages, so get all the URLs that need to be loaded to obtain a full schedule
				aba_schedule_urls = get_parsed_sref_month_urls(aba_season_page.text, 'https://www.basketball-reference.com/leagues/')
				# Retrieve the schedule pages, then extract the tables with game data
				for aba_schedule_url in aba_schedule_urls:
					aba_schedule_page = retrieve_page(aba_schedule_url)
					if aba_schedule_page is None:
						season_fail = True
					else:
						aba_schedule_data = get_parsed_sref_tables(aba_schedule_page.text)
						for html_table in aba_schedule_data:
							parsed_table = parse_sref_schedule_table(html_table)
							if parsed_table[0] == 'schedule':
								aba_season_tables.append(parsed_table[1])
					time.sleep(request_delay)
				if len(aba_season_tables) == 0:
					season_fail = True
				# Also, try to retrieve a playoff page and get the start date of the playoffs because Basketball Reference does not otherwise easily distinguish between playoff and regular season games in the tables (though this might be possible with game IDs pulled from box score URLs)
				aba_playoff_url = (('https://www.basketball-reference.com/playoffs/ABA_%d_games.html') % (current_season))
				aba_playoff_page = retrieve_page(aba_playoff_url)
				aba_playoff_start = None
				if aba_playoff_page is not None:
					aba_playoff_data = get_parsed_sref_tables(aba_playoff_page.text, delete_headers = False)
					for html_table in aba_playoff_data:
						parsed_table = parse_sref_schedule_table(html_table)
						if parsed_table[0] == 'schedule':
							aba_playoff_dates = []
							for idx, row in parsed_table[1].iterrows():
								aba_playoff_dates.append(datetime.datetime.strptime(row['Date'], '%a, %b %d, %Y').date())
							if len(aba_playoff_dates) > 0:
								aba_playoff_start = min(aba_playoff_dates)
			time.sleep(request_delay)
			# Request ABA standings and parse the table to get divisions
			aba_standings_url = (('https://www.basketball-reference.com/leagues/ABA_%d_standings.html') % (current_season))
			aba_standings_page = retrieve_page(aba_standings_url)
			if aba_standings_page is None:
				season_fail = True
			else:
				aba_standings_data = get_parsed_sref_tables(aba_standings_page.text, delete_headers = False)
				for html_table in aba_standings_data:
					table_lookup_data = parse_sref_standings_table(html_table, split_division_name = True)
					division_lookup = {**division_lookup, **table_lookup_data}
			time.sleep(request_delay)
		# If we can't download the season table, issue a warning, but still try to parse the season
		if season_fail:
			warnings.warn(('Error downloading data for season %d') % (current_season))
		# Try to parse NBA/BAA data
		if len(season_tables) > 0:
			for season_table in season_tables:
				for row_idx, row in season_table.iterrows():
					game_count = game_count + 1
					game_data[game_count] = parse_schedule_row(row, current_season, division_lookup, franchise_lookup, postseason_start = playoff_start)
		# Try to parse ABA data
		if len(aba_season_tables) > 0:
			for aba_season_table in aba_season_tables:
				for row_idx, row in aba_season_table.iterrows():
					game_count = game_count + 1
					game_data[game_count] = parse_schedule_row(row, current_season, division_lookup, franchise_lookup, postseason_start = aba_playoff_start, league = 'ABA')
		# Store the results in a JSON file
		file_handle = open(output_file, 'w')
		if file_handle is not None:
			json.dump(game_data, file_handle)
			file_handle.close()

if __name__ == '__main__':
	main()

