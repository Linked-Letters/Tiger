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
import json
import io
import pandas as pd
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

# Parse a Sports Reference table as if it's franchises
def parse_sref_franchises_table (this_table):
	# If we can get the table ID, then retrieve it
	if this_table.has_attr('id'):
		this_table_id = this_table['id'].strip()
	else:
		this_table_id = None
	franchises_data = {}
	# If we can retrieve the table ID, then we should parse it and try to extract teams
	if this_table_id is not None:
		this_table_data = pd.read_html(io.StringIO(str(this_table)), extract_links = 'body')[0]
		if list(this_table_data.columns).count('Franchise') > 0:
			for idx, row in this_table_data.iterrows():
				if row['Franchise'][1] is not None:
					franchise_id = row['Franchise'][1].split('/')[-2].split('.')[0]
					franchises_data[franchise_id] = row['Franchise'][0]
	return franchises_data

# Parse a Sports Reference table as if it's a franchise history
def parse_sref_franchise_history_table (this_table, franchise_id, franchise_name):
	# If we can get the table ID, then retrieve it
	if this_table.has_attr('id'):
		this_table_id = this_table['id'].strip()
	else:
		this_table_id = None
	seasons_data = []
	# If we can retrieve the table ID, then we should parse it and try to extract teams
	if this_table_id == franchise_id:
		this_table_data = pd.read_html(io.StringIO(str(this_table)), extract_links = 'body')[0]
		if list(this_table_data.columns).count('Season') > 0:
			# For every valid entry in the table, extract data about the team for that season, especially its identifier (which will be linked to the franchise ID)
			for idx, row in this_table_data.iterrows():
				if row['Season'][1] is not None:
					team_id = row['Season'][1].split('/')[-2].split('.')[0]
				else:
					team_id = None
				season_name = row['Season'][0].strip()
				league_id = row['Lg'][0]
				try:
					season_data = row['Lg'][1].split('/')[-1].split('.')[0].strip()
					season_year = int(season_data.strip().split('_')[1])
				except:
					season_year = None
				team_name = row['Team'][0].strip()
				if team_name[-1] == '*':
					playoff_bid = True
					team_name = team_name[:-1]
				else:
					playoff_bid = False
				# Store the data in a list
				seasons_data.append({'FranchiseID': franchise_id, 'FranchiseName': franchise_name, 'LeagueID': league_id, 'TeamName': team_name, 'Season': season_year, 'SeasonName': season_name, 'TeamID': team_id, 'InPlayoffs': playoff_bid})
	# Return the list with franchise history
	return seasons_data

def main ():
	# Get the parameters from the command line
	if len(sys.argv) < 2:
		print('Usage: '+sys.argv[0]+' <output file>')
		sys.exit(1)
	output_file = sys.argv[1].strip()
	request_delay = 5
	suppress_bs4_warnings()
	# Retrieve data about all the franchises in the data set
	franchises_url = 'https://www.basketball-reference.com/teams/'
	franchises_page = retrieve_page(franchises_url)
	if franchises_page is None:
		print('Cannot retrieve franchises')
		sys.exit(1)
	else:
		# Parse the page to get the tables of franchises (and there is generally more than one table)
		franchises_data_tables = get_parsed_sref_tables(franchises_page.text, delete_headers = False)
		franchise_data = []
		for html_table in franchises_data_tables:
			franchises_list = parse_sref_franchises_table(html_table)
			# Loop through each entry in each table, then load the franchise history
			for franchise_id in list(franchises_list.keys()):
				print(franchise_id)
				franchise_name = franchises_list[franchise_id]
				cur_franchise_url = (('https://www.basketball-reference.com/teams/%s/') % (franchise_id))
				time.sleep(request_delay)
				cur_franchise_page = retrieve_page(cur_franchise_url)
				if cur_franchise_page is None:
					warnings.warn('Cannot retrieve franchise ' + franchise_id)
				else:
					cur_franchise_data_tables = get_parsed_sref_tables(cur_franchise_page.text, delete_headers = False)
					for franchise_html_table in cur_franchise_data_tables:
						franchise_data.extend(parse_sref_franchise_history_table(franchise_html_table, franchise_id, franchises_list[franchise_id]))

	# Store the result in a JSON file
	file_handle = open(output_file, 'w')
	if file_handle is not None:
		json.dump(franchise_data, file_handle)
		file_handle.close()

if __name__ == '__main__':
	main()

