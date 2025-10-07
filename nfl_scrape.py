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

# Parse a Sports Reference table as if it's standings
def parse_sref_standings_table (this_table, split_division_name = False):
	# Create some static regular expressions
	if 'standings_re' not in parse_sref_standings_table.__dict__:
		parse_sref_standings_table.standings_re = re.compile('^(?!.*\\ Playoff\\ ).*Standings\\ Table.*$')
	# If we can get the table ID, then retrieve it
	if this_table.has_attr('id'):
		this_table_id = this_table['id'].strip()
	else:
		this_table_id = None
	standings_data = {}
	# If we can retrieve the table ID, then we should parse it
	if this_table_id is not None:
		table_caption = this_table.find('caption')
		if (table_caption is not None) and (parse_sref_standings_table.standings_re.match(table_caption.text.strip()) is not None):
			caption_split = table_caption.text.strip().split()
			if (caption_split.count('Standings') > 0) and (caption_split.index('Standings') > 0):
				standings_conference = caption_split[caption_split.index('Standings') - 1].strip()
			else:
				standings_conference = None
			this_table_data = pd.read_html(io.StringIO(str(this_table)), extract_links = 'body')[0]
			standings_division = None
			if list(this_table_data.columns).count('Tm') > 0:
				for idx, row in this_table_data.iterrows():
					if row['Tm'][1] is None:
						standings_division = row['Tm'][0].strip()
						if split_division_name and (standings_division.split()[0] == standings_conference) and (len(standings_division.split()) > 1):
							standings_division = ' '.join(standings_division.split()[1:])
						if len(standings_division.strip()) == 0:
							standings_division = None
					else:
						try:
							team_id = row['Tm'][1].split('/')[-2].split('.')[0]
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
		for column_name in list(set(this_table_data.columns).intersection(['Winner/tie', 'Loser/tie', 'VisTm', 'HomeTm'])):
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
				if ['Winner/tie', 'Loser/tie', 'VisTm', 'HomeTm'].count(this_table_data.columns[j]) > 0:
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
		if list(this_table_data.columns).count('Week') > 0:
			week_column = list(this_table_data.columns).index('Week')
		else:
			week_column = None
		for i in range(0, nrows, 1):
			row_index = list(this_table_data.index).index(i)
			del_row = False
			if this_season is not None:
				this_table_data.at[row_index, season_new_column] = this_season
			if (week_column is not None):
				wdata = str(this_table_data.iat[i, week_column])
			else:
				wdata = None
			# Not a schedule table, or a blank row in a schedule table, then delete
			if (wdata is None):
				del_row = True
			elif len(wdata.strip()) == 0:
				del_row = True
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

# Extract data from a row of the schedule table
def parse_schedule_row (row, season, division_lookup, preseason_only = False, league = 'NFL'):
	# Create some static regular expressions
	if 'preseason_re' not in parse_schedule_row.__dict__:
		parse_schedule_row.preseason_re = re.compile('^Pre[0-9][0-9]*$')
	if 'weeknumber_re' not in parse_schedule_row.__dict__:
		parse_schedule_row.weeknumber_re = re.compile('^[0-9][0-9]*$')
	week_str = row['Week'].strip()
	is_neutral = (row['Unnamed: 5'].strip() == 'N')
	is_preseason = False
	is_postseason = False
	# Check the format of the table rows
	if list(row.keys()).count('Winner/tie') > 0:
		if parse_schedule_row.weeknumber_re.match(week_str) is not None:
			week_number = int(week_str)
		else:
			is_postseason = True
			week_number = None
		if row['Unnamed: 5'].strip() == '@':
			vis_col_str = 'Winner/tie'
			home_col_str = 'Loser/tie'
			vis_col_name_str = 'Winner/tie.Name'
			home_col_name_str = 'Loser/tie.Name'
			if list(row.keys()).count('Pts') > 0:
				vis_pts_col_str = 'Pts'
				home_pts_col_str = 'Pts.1'
			else:
				vis_pts_col_str = 'PtsW'
				home_pts_col_str = 'PtsL'
		else:
			vis_col_str = 'Loser/tie'
			home_col_str = 'Winner/tie'
			vis_col_name_str = 'Loser/tie.Name'
			home_col_name_str = 'Winner/tie.Name'
			if list(row.keys()).count('Pts') > 0:
				vis_pts_col_str = 'Pts.1'
				home_pts_col_str = 'Pts'
			else:
				vis_pts_col_str = 'PtsL'
				home_pts_col_str = 'PtsW'
		game_date_parse = datetime.datetime.strptime(row['Date'].strip(), '%Y-%m-%d')
		game_year = game_date_parse.year
		game_month = game_date_parse.month
		game_day = game_date_parse.day
		if (len(row[vis_pts_col_str].strip()) == 0) or (len(row[home_pts_col_str].strip()) == 0):
			is_finished = False
		else:
			is_finished = True
	# If there are different column names as is sometimes the case with NFL tables, make sure the data gets read from the correct column names
	else:
		if parse_schedule_row.preseason_re.match(week_str) is not None:
			is_preseason = True
			week_number = int(week_str[3:])
		elif parse_schedule_row.weeknumber_re.match(week_str) is not None:
			week_number = int(week_str)
			if preseason_only:
				is_preseason = True
		else:
			is_postseason = True
			week_number = None
		vis_col_str = 'VisTm'
		home_col_str = 'HomeTm'
		vis_col_name_str = 'VisTm.Name'
		home_col_name_str = 'HomeTm.Name'
		if list(row.keys()).count('PF') > 0:
			vis_pts_col_str = 'PF'
			home_pts_col_str = 'Pts'
		else:
			vis_pts_col_str = 'Pts'
			home_pts_col_str = 'Pts.1'
		game_date_parse = datetime.datetime.strptime(row['Unnamed: 2'].strip(), '%B %d')
		game_month = game_date_parse.month
		game_day = game_date_parse.day
		if game_month >= 7:
			game_year = season
		else:
			game_year = season + 1
		if (len(row[vis_pts_col_str].strip()) == 0) or (len(row[home_pts_col_str].strip()) == 0):
			is_finished = False
		elif (int(row[vis_pts_col_str].strip()) == 0) and (int(row[home_pts_col_str].strip()) == 0):
			is_finished = False
		else:
			is_finished = True
	# This is used potentially for sorting games by the day on which the game is played
	epoch_day = (datetime.date(game_year, game_month, game_day) - datetime.date(1970, 1, 1)).days
	# Store the data for this game in a data structure
	row_data = {}
	row_data['Season'] = season
	row_data['HomeID'] = row[home_col_str].strip()
	row_data['AwayID'] = row[vis_col_str].strip()
	if list(division_lookup.keys()).count(row_data['HomeID']) > 0:
		row_data['HomeConference'] = division_lookup[row_data['HomeID']]['Conference']
		row_data['HomeDivision'] = division_lookup[row_data['HomeID']]['Division']
	else:
		row_data['HomeConference'] = None
		row_data['HomeDivision'] = None
	if list(division_lookup.keys()).count(row_data['AwayID']) > 0:
		row_data['AwayConference'] = division_lookup[row_data['AwayID']]['Conference']
		row_data['AwayDivision'] = division_lookup[row_data['AwayID']]['Division']
	else:
		row_data['AwayConference'] = None
		row_data['AwayDivision'] = None
	row_data['HomeName'] = row[home_col_name_str].strip()
	row_data['AwayName'] = row[vis_col_name_str].strip()
	row_data['IsCompleted'] = is_finished
	if is_finished:
		row_data['AwayScore'] = int(row[vis_pts_col_str].strip())
		row_data['HomeScore'] = int(row[home_pts_col_str].strip())
	else:
		row_data['AwayScore'] = None
		row_data['HomeScore'] = None
	row_data['Year'] = game_year
	row_data['Month'] = game_month
	row_data['Day'] = game_day
	row_data['EpochDay'] = epoch_day
	row_data['IsNeutralSite'] = is_neutral
	row_data['IsPreseason'] = is_preseason
	row_data['IsPostseason'] = is_postseason
	row_data['Week'] = week_number
	row_data['WeekString'] = week_str
	row_data['League'] = league
	# Return the data structure
	return row_data

def main ():
	# Get the parameters from the command line
	if len(sys.argv) < 4:
		print('Usage: '+sys.argv[0]+' <start season> <end season> <output file>')
		sys.exit(1)
	try:
		start_season = int(sys.argv[1].strip())
		end_season = int(sys.argv[2].strip())
	except:
		print('Invalid season')
		sys.exit(1)
	output_file = sys.argv[3].strip()
	request_delay = 5
	suppress_bs4_warnings()
	preseason_re = re.compile('^Pre[0-9][0-9]*$')
	game_data = {}
	game_count = 0
	# Loop through the seasons
	for current_season in range(start_season, end_season + 1, 1):
		season_fail = False
		preseason_table = None
		season_table = None
		afl_season_table = None
		standings_url = (('https://www.pro-football-reference.com/years/%d/') % (current_season))
		standings_page = retrieve_page(standings_url)
		division_lookup = {}
		# Load in a standings table to get NFL divisions
		if standings_page is None:
			season_fail = True
		else:
			standings_data = get_parsed_sref_tables(standings_page.text, delete_headers = False)
			for html_table in standings_data:
				table_lookup_data = parse_sref_standings_table(html_table)
				division_lookup = {**division_lookup, **table_lookup_data}
		time.sleep(request_delay)
		# If there should be preseason data, attempt to load the schedule and results
		if current_season >= 1983:
			preseason_url = (('https://www.pro-football-reference.com/years/%d/preseason.htm') % (current_season))
			preseason_page = retrieve_page(preseason_url)
			if preseason_page is None:
				season_fail = True
			else:
				preseason_data = get_parsed_sref_tables(preseason_page.text)
				for html_table in preseason_data:
					parsed_table = parse_sref_schedule_table(html_table)
					if parsed_table[0] == 'preseason':
						preseason_table = parsed_table[1]
				if preseason_table is None:
					season_fail = True
			time.sleep(request_delay)
		else:
			preseason_page = None
		# Load results for the NFL season if the page should exist
		if current_season >= 1922:
			season_url = (('https://www.pro-football-reference.com/years/%d/games.htm') % (current_season))
			season_page = retrieve_page(season_url)
			if season_page is None:
				season_fail = True
			else:
				season_data = get_parsed_sref_tables(season_page.text)
				for html_table in season_data:
					parsed_table = parse_sref_schedule_table(html_table)
					if parsed_table[0] == 'games':
						season_table = parsed_table[1]
				if season_table is None:
					season_fail = True
			time.sleep(request_delay)
		else:
			season_page = None
		# Read AFL games if appropriate
		if (current_season >= 1960) and (current_season <= 1969):
			# Load in the AFL schedule
			afl_season_url = (('https://www.pro-football-reference.com/years/%d_AFL/games.htm') % (current_season))
			afl_season_page = retrieve_page(afl_season_url)
			if afl_season_page is None:
				season_fail = True
			else:
				afl_season_data = get_parsed_sref_tables(afl_season_page.text)
				for html_table in afl_season_data:
					parsed_table = parse_sref_schedule_table(html_table)
					if parsed_table[0] == 'games':
						afl_season_table = parsed_table[1]
				if afl_season_table is None:
					season_fail = True
			time.sleep(request_delay)
			# Load an AFL standings table to get divisions
			afl_standings_url = (('https://www.pro-football-reference.com/years/%d_AFL/') % (current_season))
			afl_standings_page = retrieve_page(afl_standings_url)
			if afl_standings_page is None:
				season_fail = True
			else:
				afl_standings_data = get_parsed_sref_tables(afl_standings_page.text, delete_headers = False)
				for html_table in afl_standings_data:
					table_lookup_data = parse_sref_standings_table(html_table)
					division_lookup = {**division_lookup, **table_lookup_data}
			time.sleep(request_delay)
		# If we can't download the season table, issue a warning, but still try to parse the season
		if season_fail:
			warnings.warn(('Error downloading data for season %d') % (current_season))
		# If we have season data and it includes preseason games, that's going to work best
		if (season_table is not None) and (any([preseason_re.search(x) is not None for x in list(season_table['Week'])])):
			for row_idx, row in season_table.iterrows():
				if preseason_re.search(row['Week'].strip()) is not None:
					game_count = game_count + 1
					game_data[game_count] = parse_schedule_row(row, current_season, division_lookup)
		# Otherwise, parse the preseason data page
		elif preseason_table is not None:
			for row_idx, row in preseason_table.iterrows():
				game_count = game_count + 1
				game_data[game_count] = parse_schedule_row(row, current_season, division_lookup, preseason_only = True)
		# Parse regular season NFL data
		if season_table is not None:
			for row_idx, row in season_table.iterrows():
				if preseason_re.search(row['Week'].strip()) is None:
					game_count = game_count + 1
					game_data[game_count] = parse_schedule_row(row, current_season, division_lookup)
		# If there's AFL data, parse it
		if afl_season_table is not None:
			for row_idx, row in afl_season_table.iterrows():
				# Exclude the Superbowl because it'll be already included for NFL games
				if row['Week'].strip() != 'SuperBowl':
					game_count = game_count + 1
					game_data[game_count] = parse_schedule_row(row, current_season, division_lookup, league = 'AFL')
		# Write the game data to a JSON file
		file_handle = open(output_file, 'w')
		if file_handle is not None:
			json.dump(game_data, file_handle)
			file_handle.close()

if __name__ == '__main__':
	main()

