import pandas as pd
import argparse
import os
import pulp
import json
import glob
from math import ceil
import pathlib

# get enhancer-promoter distance
# this is referred by dmax and dmin
def get_ep_distance(e_start, e_end, p_start, p_end):
	e_pos = (e_start + e_end) / 2
	p_pos = (p_start + p_end) / 2
	return abs(e_pos - p_pos)

def CBMF(input, outdir, cell_type, dmin, dmax, alpha, concat):
	if dmax != "INF": # if dmax is not INF, convert to int
		dmax = int(dmax)

	os.makedirs(os.path.join(outdir), exist_ok=True)

	cboep_epi = pd.DataFrame(
		columns=[
			"label", "enhancer_distance_to_promoter",
			"enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name",
			"promoter_chrom", "promoter_start", "promoter_end", "promoter_name",
		]
	)

	# load positive only
	input_path = os.path.join(input)
	input_epi = pd.read_csv(input_path)
	positive_epi = input_epi[input_epi["label"]==1]

	# chromosome wise
	for chrom, sub_df in positive_epi.groupby("enhancer_chrom"):

		G_from = []
		G_to = []
		G_cap = []

		enhDict_pos = {}
		prmDict_pos = {}
		name2range = {}


		for _, pair_data in sub_df.iterrows():
			enhName = pair_data["enhancer_name"]
			prmName = pair_data["promoter_name"]

			name2range[enhName] = (pair_data["enhancer_start"], pair_data["enhancer_end"])
			name2range[prmName] = (pair_data["promoter_start"], pair_data["promoter_end"])

			if enhDict_pos.get(enhName) == None:
				enhDict_pos[enhName] = {}
			if prmDict_pos.get(prmName) == None:
				prmDict_pos[prmName] = {}

			if enhDict_pos[enhName].get(prmName) == None:
				enhDict_pos[enhName][prmName] = 1
			if prmDict_pos[prmName].get(enhName) == None:
				prmDict_pos[prmName][enhName] = 1

		# source => each enhancer
		for enhName in enhDict_pos.keys():
			cap = len(enhDict_pos[enhName])
			cap = ceil(cap * alpha)
			G_from.append("source")
			G_to.append(enhName)
			G_cap.append(cap)

		# each promoter => sink
		for prmName in prmDict_pos.keys():
			cap = len(prmDict_pos[prmName])
			cap = ceil(cap * alpha)
			G_from.append(prmName)
			G_to.append("sink")
			G_cap.append(cap)

		# each enhancer => each promoter
		enhList = set(sub_df["enhancer_name"].tolist())
		prmList = set(sub_df["promoter_name"].tolist())
		for enhName in enhList:
			for prmName in prmList:
				assert enhDict_pos.get(enhName) != None
				assert prmDict_pos.get(prmName) != None

				if enhDict_pos[enhName].get(prmName) != None: # exist positive pair
					assert prmDict_pos[prmName].get(enhName) != None
					continue

				enh_start, enh_end = name2range[enhName]
				prm_start, prm_end = name2range[prmName]
				distance = get_ep_distance(enh_start, enh_end, prm_start, prm_end)

				# extract pairs in consideration of dmax

				if dmax == "INF":
					if distance >= dmin:
						G_from.append(enhName)
						G_to.append(prmName)
						G_cap.append(1)
				else:
					if distance <= dmax and distance >= dmin:
						G_from.append(enhName)
						G_to.append(prmName)
						G_cap.append(1)

		
		bipartiteGraph = pd.DataFrame(
			{
				"from": G_from,
				"to": G_to,
				"cap": G_cap
			},
			index=None
		)

		assert bipartiteGraph.duplicated().sum() == 0

		
		from_list = bipartiteGraph["from"].tolist()
		to_list = bipartiteGraph["to"].tolist()
		cap_list = bipartiteGraph["cap"].tolist()

		# "z" is sum of flow from "source"
		# Maximizing "z" is our goal
		z = pulp.LpVariable("z", lowBound=0)
		problem = pulp.LpProblem("maxflow", pulp.LpMaximize)
		problem += z

		# create variables
		bipartiteGraph["Var"] = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=cap_list[i],cat=pulp.LpInteger) for i in bipartiteGraph.index]

		# Added constraints on all vertices (flow conservation law)
		for node in set(from_list)|set(to_list):
			if node == "source":
				# sum of flow from "source" == "z"
				fromSource_df = bipartiteGraph[bipartiteGraph["from"] == node]
				sumFlowFromSource = pulp.lpSum(fromSource_df["Var"])
				problem += sumFlowFromSource == z
			elif node == "sink":
				# sum of flow to "sink" == "z"
				toSink_df = bipartiteGraph[bipartiteGraph["to"] == node]
				sumFlowToSink = pulp.lpSum(toSink_df["Var"])
				problem += sumFlowToSink == z
			else:
				# sum of flow into a vertex == sum of flow out
				fromNowNode = bipartiteGraph[bipartiteGraph["from"] == node]
				toNowNode = bipartiteGraph[bipartiteGraph["to"] == node]
				sumFlowFromNode = pulp.lpSum(fromNowNode["Var"])
				sumFlowToNode = pulp.lpSum(toNowNode["Var"])
				problem += sumFlowFromNode == sumFlowToNode


		# solve
		problem.solve()
		bipartiteGraph['Val'] = bipartiteGraph.Var.apply(pulp.value)

		assert bipartiteGraph.duplicated().sum() == 0

		bipartiteGraph = bipartiteGraph[["from", "to", "Val"]]

		# drop "source" and "sink"
		bipartiteGraph = bipartiteGraph[bipartiteGraph["from"] != "source"]
		bipartiteGraph = bipartiteGraph[bipartiteGraph["to"] != "sink"]
		bipartiteGraph = bipartiteGraph[bipartiteGraph["Val"] == 1]
		if len(bipartiteGraph) == 0:
			continue

		bipartiteGraph.drop("Val", axis=1, inplace=True)
		bipartiteGraph["label"] = 0
		bipartiteGraph["enhancer_chrom"] = chrom
		bipartiteGraph["promoter_chrom"] = chrom
		bipartiteGraph.rename(columns={'from': 'enhancer_name', 'to': 'promoter_name'}, inplace=True)
		bipartiteGraph["enhancer_start"] = bipartiteGraph["enhancer_name"].apply(lambda x: name2range[x][0])
		bipartiteGraph["enhancer_end"] =  bipartiteGraph["enhancer_name"].apply(lambda x: name2range[x][1])
		bipartiteGraph["promoter_start"] = bipartiteGraph["promoter_name"].apply(lambda x: name2range[x][0])
		bipartiteGraph["promoter_end"] = bipartiteGraph["promoter_name"].apply(lambda x: name2range[x][1])
		print(len(bipartiteGraph))
		print(bipartiteGraph.columns)
		bipartiteGraph["enhancer_distance_to_promoter"] = bipartiteGraph.apply(
			lambda row:
			get_ep_distance(
				row["enhancer_start"], row["enhancer_end"],
				row["promoter_start"], row["promoter_end"]
			),
			axis=1
		)
		
		cboep_epi = pd.concat([cboep_epi, bipartiteGraph], axis=0, ignore_index=True)
	

	if concat:
		cboep_epi = pd.concat([positive_epi, cboep_epi], axis=0, ignore_index=True)

	cboep_epi["label"] = cboep_epi["label"].astype(int)
	cboep_epi["enhancer_distance_to_promoter"] = cboep_epi["enhancer_distance_to_promoter"].astype(int)
	cboep_epi["enhancer_start"] = cboep_epi["enhancer_start"].astype(int)
	cboep_epi["enhancer_end"] = cboep_epi["enhancer_end"].astype(int)
	cboep_epi["promoter_start"] = cboep_epi["promoter_start"].astype(int)
	cboep_epi["promoter_end"] = cboep_epi["promoter_end"].astype(int)
		
	cboep_epi = cboep_epi[
		["label", "enhancer_distance_to_promoter",
		"enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name",
		"promoter_chrom", "promoter_start", "promoter_end", "promoter_name"]
	]
	# cboep_epi.to_csv(args_outfile, index=False)
	cboep_epi.to_csv(os.path.join(outdir, cell_type + ".csv"), index=False)



def parse_arguments():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("-i", "--input", help="input file path")
	p.add_argument("-o", "--outdir", help="output directory path")
	p.add_argument("-c", "--cell_type", help="cell type name like GM12878 (used as a prefix of output file name)")
	p.add_argument("--dmin", type=int, default=0, help="minimum distance between enhancer and promoter")
	p.add_argument("--dmax", default=2500000, help="maximum distance between enhancer and promoter")
	p.add_argument("--alpha", type=float, default=1.0, help="")
	p.add_argument("--concat", action="store_true", default=False, help="concat CBMF negative and input positive")
	args = p.parse_args()
	return args

if __name__ == "__main__":
	args = parse_arguments()
	CBMF(
		input=args.input,
		outdir=args.outdir,
		cell_type=args.cell_type,
		dmin=args.dmin,
		dmax=args.dmax,
		alpha=args.alpha,
		concat=args.concat
	)






