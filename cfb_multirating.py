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

import json
import numpy as np
import sys

def main ():
	if (len(sys.argv) < 5):
		print('Usage: '+sys.argv[0]+' <input JSON file> <division> <header frequency> <season> [<column title> <extra JSON file> ...]')
		exit()

	input_file_name = sys.argv[1].strip()
	division_id = sys.argv[2].strip()
	table_header_frequency = int(sys.argv[3].strip())
	prediction_season = int(sys.argv[4].strip())

	input_handle = open(input_file_name, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	input_data = json.load(input_handle)
	input_handle.close()

	# Get a list of teams
	teamid_list = list(input_data['TeamRatings'].keys())

	# Read in a list of each team's rating data
	team_rating_list = []
	for teamid in teamid_list:
		if (input_data['TeamRatings'][teamid]['Division'] == division_id):
			team_rating_list.append([teamid, input_data['TeamRatings'][teamid]['Name'], input_data['TeamRatings'][teamid]['Rating']])

	# Loop through the remaining parameters on the command line and treat them as pairs of column headers and data files with ratings
	extra_header = []
	for param_idx in range(5, len(sys.argv) - 1, 2):
		input_handle = open(sys.argv[param_idx + 1], 'r')
		if input_handle is None:
			print('Could not open input file')
			exit()
		extra_data = json.load(input_handle)
		input_handle.close()
		extra_rating_list = []
		extra_teamid_list = list(extra_data['TeamRatings'].keys())
		for teamid in extra_teamid_list:
			if (extra_data['TeamRatings'][teamid]['Division'] == division_id):
				extra_rating_list.append([teamid, extra_data['TeamRatings'][teamid]['Rating']])
		extra_data_list = [[y[0] + 1] + y[1] for y in enumerate(sorted(extra_rating_list, key = lambda x: x[1], reverse = True))]
		extra_data_dict = dict([(x[1], [x[0], x[2]]) for x in extra_data_list])
		for idx in range(0, len(team_rating_list), 1):
			teamid = team_rating_list[idx][0]
			team_rating_list[idx].extend([extra_data_dict[teamid][1], extra_data_dict[teamid][0]])
		extra_header.append(sys.argv[param_idx])

	# Now create a table that will output the data, adding extra headers and columns as necessary
	rating_text = []
	rating_header = ['Rank', 'Rating', 'Team'] + extra_header
	rating_extra_cols = len(extra_header)
	rating_alignment = ['>', '', ''] + ([''] * len(extra_header))
	table_title = 'Predictive Ratings'
	table_subtitles = []
	cur_line = 0
	for team_data in [[y[0] + 1] + y[1] for y in enumerate(sorted(team_rating_list, key = lambda x: x[2], reverse = True))]:
		if (cur_line % table_header_frequency) == 0:
			rating_text.append(rating_header)
		rating_line = []
		rating_line.append('{:d}'.format(int(team_data[0])))
		rating_line.append('{0:.2f}'.format(float(team_data[3])))
		rating_line.append(str(team_data[2]))
		for extra_idx in range(0, rating_extra_cols, 1):
			rating_line.append('{0:.2f}'.format(float(team_data[4 + extra_idx * 2])) + ' (' + '{:d}'.format(int(team_data[5 + extra_idx * 2])) + ') ')
		rating_text.append(rating_line)
		cur_line = cur_line + 1
	print(rating_text)
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

if __name__ == '__main__':
	main()
