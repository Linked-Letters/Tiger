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
import math
import networkx
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def main ():
	if (len(sys.argv) < 6):
		print('Usage: '+sys.argv[0]+' <input JSON file> <output image file> <season to plot> <number of clusters> <league name>')
		exit()

	input_file_name = sys.argv[1].strip()
	output_file_name = sys.argv[2].strip()
	league_name = sys.argv[5].strip()
	try:
		season_to_plot = int(sys.argv[3].strip())
		nclusters = int(sys.argv[4].strip())
	except:
		print('Error in seasons or cluster number')
		exit()

	input_handle = open(input_file_name, 'r')
	if input_handle is None:
		print('Could not open input file')
		exit()
	input_data = json.load(input_handle)
	input_handle.close()

	# Get a list of all data variables
	data_keys = list(set([y for z in [list(input_data[x].keys()) for x in list(input_data.keys())] for y in z]))

	# Set some basic filters
	game_filter = [[x, all([input_data[x]['IsCompleted'], (input_data[x]['HomeScore'] is not None), (input_data[x]['AwayScore'] is not None), (input_data[x]['Season'] is not None) and (int(input_data[x]['Season']) == season_to_plot), ((input_data[x]['Year'] is not None) and (input_data[x]['Month'] is not None) and (input_data[x]['Day'] is not None))])] for x in input_data.keys()]

	# Load in data for games that satisfy the filters
	input_games_home_id = [input_data[x[0]]['HomeName'] for x in game_filter if x[1]]
	input_games_away_id = [input_data[x[0]]['AwayName'] for x in game_filter if x[1]]
	if (data_keys.count('HomeDivision') > 0) and (data_keys.count('AwayDivision') > 0):
		input_games_home_division = [input_data[x[0]]['HomeDivision'] for x in game_filter if x[1]]
		input_games_away_division = [input_data[x[0]]['AwayDivision'] for x in game_filter if x[1]]
		draw_legend = True
	else:
		input_games_home_division = [None] * len(game_filter)
		input_games_away_division = [None] * len(game_filter)
		draw_legend = False
	all_team_list = sorted(list(set(input_games_home_id + input_games_away_id)))
	all_team_divisions = {}
	all_team_divcolor = {}
	# Color code teams according to divisions
	for teamid, div in zip(input_games_home_id + input_games_away_id, input_games_home_division + input_games_away_division):
		all_team_divisions[teamid] = div
		if div == 'fbs':
			all_team_divcolor[teamid] = '#FFCC33'
		elif div == 'fcs':
			all_team_divcolor[teamid] = '#00FF00'
		elif div == 'ii':
			all_team_divcolor[teamid] = '#FF33CC'
		elif div == 'iii':
			all_team_divcolor[teamid] = '#00FFCC'
		elif div == 'NFC North':
			all_team_divcolor[teamid] = '#CC0000'
		elif div == 'NFC South':
			all_team_divcolor[teamid] = '#00CC00'
		elif div == 'NFC East':
			all_team_divcolor[teamid] = '#0000CC'
		elif div == 'NFC West':
			all_team_divcolor[teamid] = '#660066'
		elif div == 'AFC North':
			all_team_divcolor[teamid] = '#FF3333'
		elif div == 'AFC South':
			all_team_divcolor[teamid] = '#33FF33'
		elif div == 'AFC East':
			all_team_divcolor[teamid] = '#3333FF'
		elif div == 'AFC West':
			all_team_divcolor[teamid] = '#993399'
		else:
			all_team_divcolor[teamid] = '#000000'

	# Get the team indexes for each game
	input_games_home_listid = [all_team_list.index(x) for x in input_games_home_id]
	input_games_away_listid = [all_team_list.index(x) for x in input_games_away_id]

	# Get a graph of all the teams in the data set and games played against them
	game_network = networkx.MultiGraph()
	game_network.add_nodes_from([x for x in list(range(0, len(all_team_list), 1))])
	game_network.add_edges_from([(input_games_home_listid[x], input_games_away_listid[x]) for x in range(0, len(input_games_home_id), 1)])
	game_network_groups = list(networkx.connected_components(game_network))

	# Get the teams and games from the largest network and make a sub-network of those teams
	graphid = 0
	node_id_map = dict([[x, list(game_network_groups[graphid])[x]] for x in range(0, len(list(game_network_groups[graphid])), 1)])
	id_node_map = {x: y for y, x in node_id_map.items()}
	sub_network = networkx.MultiGraph()
	sub_network.add_nodes_from([id_node_map[x] for x in list(game_network_groups[graphid])])

	# Dynamically determine the size of the nodes
	node_size = max(32 - math.ceil(math.log2(len(list(game_network_groups[graphid]))) * 3), 3)

	# Prepare the legend if one is desired
	if draw_legend:
		legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color = '#FFCC33', label = 'FBS', lw = 0, markerfacecolor = '#FFCC33', markersize = node_size - 1), mpl.lines.Line2D([0], [0], marker='o', color = '#00FF00', label = 'FCS', lw = 0, markerfacecolor = '#00FF00', markersize = node_size - 1), mpl.lines.Line2D([0], [0], marker='o', color = '#FF33CC', label = 'Division II', lw = 0, markerfacecolor = '#FF33CC', markersize = node_size - 1), mpl.lines.Line2D([0], [0], marker='o', color = '#00FFCC', label = 'Division III', lw = 0, markerfacecolor = '#00FFCC', markersize = node_size - 1), mpl.lines.Line2D([0], [0], marker='o', color = '#000000', label = 'Other', lw = 0, markerfacecolor = '#000000', markersize = node_size - 1)]

	# Add edges corresponding to each game in the sub-network and calculate edge betweenness centrality
	sub_network.add_edges_from([(id_node_map[input_games_home_listid[x]], id_node_map[input_games_away_listid[x]]) for x in range(0, len(input_games_home_id), 1) if ((list(game_network_groups[graphid]).count(input_games_home_listid[x]) > 0) and (list(game_network_groups[graphid]).count(input_games_away_listid[x]) > 0))])
	sub_network_edge_centrality = networkx.edge_betweenness_centrality(sub_network)

	# This is an attempt to cluster the nodes in the graph into clumps where they are better connected, then display them in a way that is easy to understand
	communities = networkx.community.greedy_modularity_communities(sub_network, best_n = nclusters)
	supergraph = networkx.cycle_graph(len(communities))
	superpos = networkx.spring_layout(supergraph, scale = 1.2, seed = 1)
	centers = list(superpos.values())
	pos = {}
	for center, comm in zip(centers, communities):
		pos.update(networkx.spring_layout(networkx.subgraph(sub_network, comm), center = center, seed = 1))
	for nodes in communities:
		node_color_list = [all_team_divcolor[all_team_list[node_id_map[x]]] for x in nodes]
		networkx.draw_networkx_nodes(sub_network, pos = pos, nodelist = nodes, node_color = node_color_list, node_size = node_size)

	# This is an attempt at trying to find a good color and size scale for the edges to show which ones have higher edge betweenness centrality
	maxweight = 1.25 ** math.log10(max([sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())]))
	minweight = 1.25 ** math.log10(min([sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())]))
	magweightdiff = max([sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())]) / min([sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())])

	edgecolorlist = [mpl.colormaps['seismic'](mpl.colors.Normalize(vmin = minweight, vmax = maxweight)(1.25 ** math.log10(y))) for y in [sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())]]
	# With a white background, it's preferable not to have a node actually be white if that's part of the color scale, so darken the edges as needed
	edgecolorlist = [tuple(list(mpl.colors.hsv_to_rgb([y[0], y[1], min(0.85, y[2])])) + [1.0]) for y in [mpl.colors.rgb_to_hsv(x[0:3]).tolist() for x in edgecolorlist]]
	edgewidthlist = [mpl.colors.Normalize(vmin = minweight, vmax = maxweight)(1.25 ** math.log10(y)) ** 2 + 0.1 for y in [sub_network_edge_centrality[x] for x in list(sub_network_edge_centrality.keys())]]

	# Draw the edges between the nodes, which were previously drawn on the graph
	networkx.draw_networkx_edges(sub_network, pos = pos, width = edgewidthlist, edge_color = edgecolorlist)

	# Add a legend and remove a bounding box around the plot, then save the image
	ax = plt.gca()
	if draw_legend:
		ax.legend(handles = legend_elements, loc = 'center right', fontsize = 8)
	plt.text(0.02, 0.98, 'Weight magnitude difference: ' + str(round(magweightdiff , 3)), horizontalalignment = 'left', verticalalignment = 'center', transform = ax.transAxes)
	plt.text(0.98, 0.98, 'Clusters: ' + str(len(communities)), horizontalalignment = 'right', verticalalignment = 'center', transform = ax.transAxes)
	plt.box(False)
	plt.title(str(season_to_plot) + ' ' + league_name + ' Game Map')
	plt.tight_layout()
	plt.savefig(output_file_name, bbox_inches = 'tight', dpi = 300)
	plt.close()
	plt.clf()

if __name__ == '__main__':
	main()
