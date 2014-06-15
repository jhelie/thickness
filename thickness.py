################################################################################################################################################
# IMPORT MODULES
################################################################################################################################################

#import general python tools
import argparse
import operator
from operator import itemgetter
import sys, os, shutil
import os.path
import math

#import python extensions/packages to manipulate arrays
import numpy 				#to manipulate arrays
import scipy 				#mathematical tools and recipesimport MDAnalysis

#import graph building module
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm			#colours library
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
fontP=FontProperties()

#import MDAnalysis
import MDAnalysis
from MDAnalysis import *
import MDAnalysis.analysis
import MDAnalysis.analysis.leaflet
import MDAnalysis.analysis.distances

#set MDAnalysis to use periodic boundary conditions
MDAnalysis.core.flags['use_periodic_selections'] = True
MDAnalysis.core.flags['use_KDTree_routines'] = False
MDAnalysis.core.flags['use_KDTree_routines'] = False

################################################################################################################################################
# RETRIEVE USER INPUTS
################################################################################################################################################

#create parser
#=============
version_nb="2.0.0"
parser = argparse.ArgumentParser(prog='thickness', usage='', add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter, description=\
'''
********************************************
v''' + version_nb + '''
author: Jean Helie
git: https://github.com/jhelie/thickness.git
********************************************

[ Description ]

This script calculates bilayer the evolution of bilayer thickness. It outputs:
 - statistics by species
 - graphs for each specie
 - files for visualisation in VMD

[ Requirements ]

The following python module(s) are needed:
 - MDAnalysis

[ Notes ]

1. It's a good idea to pre-process the xtc first:
    - use trjconv with the -pbc mol option
    - only output the relevant lipids (e.g. no water but no cholesterol either)

2. In case lipids flipflop during the trajectory, a file listing them can be supplied via the -i flag.
   This file can be the output of the ff_detect script and should follow the format:
   'resname,resid,starting_leaflet' on each line e.g. 'POPC,145,lower'
   If flipflopping lipids are not identified they may add some significant noise to the results.

3. The code can easily be updated to add more lipids, for now those containing the following particles
   are handled:
    - Martini: PO4, PO3, B1A

4. The colour associated to each lipid specie can be defined by supplying a colour file containing
   'resname,colour# on each line (a line with a colour MUST be defined for all species).
   Colours can be specified using single letter code (e.g. 'r'), hex code  or the name of colormap.
   In case a colormap is used, its name must be specified as the colour for each lipid specie - type
   'order_param --colour_maps' to see a list of the standard colour maps.
   If no colour is used the 'jet' colour map is used by default.

5. The size (or size group) of the cluster each protein is detected to be involved can be visualised
   with VMD. This can be done either with pdb files (output frequency controled via -w flag) or with 
   the xtc trajectory.
     - pdb file: the clustering info for each protein is stored in the beta factor column. Just open
                 the pdb with VMD and choose Draw Style > Coloring Method > Beta 
     - xtc file: the clustering info is stored in a .txt file in /4_VMD/ and you can load it into the
                 user field in the xtc by sourcing the script 'set_user_fields.tcl' and running the 
                 procedure 'set_thickness'

[ Usage ]
	
Option	      Default  	Description                    
-----------------------------------------------------
-f			: structure file [.gro]
-x			: trajectory file [.xtc] (optional)
-c			: colour definition file, see note 4
-o			: name of output folder
-b			: beginning time (ns) (the bilayer must exist by then!)
-e			: ending time (ns)	
-t 		10	: process every t-frames
-w			: write annotated pdbs every [w] processed frames (optional, see note 5)
--smooth		: nb of points to use for data smoothing (optional)
--neighbours	5	: nb of nearest opposite neighbours to use for thickness calculation

Lipids identification  
-----------------------------------------------------
--flipflops		: input file with flipflopping lipids, see note 2
--forcefield		: forcefield options, see note 3
--no-opt		: do not attempt to optimise leaflet identification (useful for huge system)

Other options
-----------------------------------------------------
--colour_maps		: show list of standard colour maps, see note 4
--version		: show version number and exit
-h, --help		: show this menu and exit
 
''')

#data options
parser.add_argument('-f', nargs=1, dest='grofilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-x', nargs=1, dest='xtcfilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-c', nargs=1, dest='colour_file', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-o', nargs=1, dest='output_folder', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-b', nargs=1, dest='t_start', default=[-1], type=int, help=argparse.SUPPRESS)
parser.add_argument('-e', nargs=1, dest='t_end', default=[10000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('-t', nargs=1, dest='frames_dt', default=[10], type=int, help=argparse.SUPPRESS)
parser.add_argument('-w', nargs=1, dest='frames_write_dt', default=[1000000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('--neighbours', nargs=1, dest='thick_nb_neighbours', default=[5], type=int, help=argparse.SUPPRESS)
parser.add_argument('--smooth', nargs=1, dest='nb_smoothing', default=[0], type=int, help=argparse.SUPPRESS)

#lipids identification
parser.add_argument('--flipflops', nargs=1, dest='selection_file_ff', default=['no'], type=float, help=argparse.SUPPRESS)
parser.add_argument('--forcefield', dest='forcefield_opt', choices=['martini'], default='martini', help=argparse.SUPPRESS)
parser.add_argument('--no-opt', dest='cutoff_leaflet', action='store_false', help=argparse.SUPPRESS)

#other options
parser.add_argument('--colour_maps', dest='show_colour_map', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--version', action='version', version='%(prog)s v' + version_nb, help=argparse.SUPPRESS)
parser.add_argument('-h','--help', action='help', help=argparse.SUPPRESS)

#store inputs
#============
args=parser.parse_args()
args.grofilename=args.grofilename[0]
args.xtcfilename=args.xtcfilename[0]
args.colour_file=args.colour_file[0]
args.output_folder=args.output_folder[0]
args.frames_dt=args.frames_dt[0]
args.frames_write_dt=args.frames_write_dt[0]
args.t_start=args.t_start[0]
args.t_end=args.t_end[0]
args.thick_nb_neighbours=args.thick_nb_neighbours[0]
args.selection_file_ff=args.selection_file_ff[0]
args.nb_smoothing=args.nb_smoothing[0]

#show colour maps
#----------------
if args.show_colour_map:
	print ""
	print "The following standard matplotlib color maps can be used:"
	print ""
	print "Spectral, summer, coolwarm, pink_r, Set1, Set2, Set3, brg_r, Dark2, hot, PuOr_r, afmhot_r, terrain_r,"
	print "PuBuGn_r, RdPu, gist_ncar_r, gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern, "
	print "gnuplot_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r, gist_rainbow, RdYlBu_r, "
	print "gist_heat_r, OrRd_r, CMRmap, bone, gist_stern_r, RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, "
	print "winter_r, PuBu, RdGy_r, spectral, flag_r, jet_r, RdPu_r, Purples_r, gist_yarg, BuGn, Paired_r, hsv_r, "
	print "bwr, cubehelix, YlOrRd, Greens, PRGn, gist_heat, spectral_r, Paired, hsv, Oranges_r, prism_r, Pastel2, "
	print "Pastel1_r, Pastel1, gray_r, PuRd_r, Spectral_r, gnuplot2_r, BuPu, YlGnBu_r, copper, gist_earth_r, "
	print "Set3_r, OrRd, PuBu_r, ocean_r, brg, gnuplot2, jet, bone_r, gist_earth, Oranges, RdYlGn_r, PiYG,"
	print "CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, gist_gray, flag, seismic_r, RdBu_r, BrBG, Reds,"
	print "BuGn_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy, PuRd, Accent_r, Blues, Greys, autumn, cubehelix_r, "
	print "nipy_spectral_r, PRGn_r, Greys_r, pink, binary, winter, gnuplot, RdBu, prism, YlOrBr, coolwarm_r,"
	print "rainbow_r, rainbow, PiYG_r, YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, bwr_r, autumn_r, ocean,"
	print "Set1_r, PuOr, PuBuGn, nipy_spectral, afmhot."
	print ""
	sys.exit(0)

#sanity check
#============
if not os.path.isfile(args.grofilename):
	print "Error: file " + str(args.grofilename) + " not found."
	sys.exit(1)
if args.colour_file!="no" and not os.path.isfile(args.colour_file):
	print "Error: file " + str(args.colour_file) + " not found."
	sys.exit(1)
if args.selection_file_ff!="no" and not os.path.isfile(selection_file_ff):
	print "Error: file " + str(args.selection_file_ff) + " not found."
	sys.exit(1)
if args.xtcfilename=="no":
	if '-t' in sys.argv:
		print "Error: -t option specified but no xtc file specified."
		sys.exit(1)
	elif '-b' in sys.argv:
		print "Error: -b option specified but no xtc file specified."
		sys.exit(1)
	elif '-e' in sys.argv:
		print "Error: -e option specified but no xtc file specified."
		sys.exit(1)
	elif '--smooth' in sys.argv:
		print "Error: --smooth option specified but no xtc file specified."
		sys.exit(1)
elif not os.path.isfile(args.xtcfilename):
	print "Error: file " + str(args.xtcfilename) + " not found."
	sys.exit(1)

#create folders and log file
#===========================
if args.output_folder=="no":
	if args.xtcfilename=="no":
		args.output_folder="thickness_" + args.grofilename[:-4]
	else:
		args.output_folder="thickness_" + args.xtcfilename[:-4]
if os.path.isdir(args.output_folder):
	print "Error: folder " + str(args.output_folder) + " already exists, choose a different output name via -o."
	sys.exit(1)
else:
	#create folders
	#--------------
	os.mkdir(args.output_folder)
	#1 species
	os.mkdir(args.output_folder + "/1_species")
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/1_species/xvg")
		os.mkdir(args.output_folder + "/1_species/png")
		if args.nb_smoothing>1:
			os.mkdir(args.output_folder + "/1_species/smoothed")
			os.mkdir(args.output_folder + "/1_species/smoothed/png")
			os.mkdir(args.output_folder + "/1_species/smoothed/xvg")
	#2 snapshots
	os.mkdir(args.output_folder + "/2_snapshots")
	#3 vmd
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/3_VMD")

	#create log
	#----------
	filename_log=os.getcwd() + '/' + str(args.output_folder) + '/thickness.log'
	output_log=open(filename_log, 'w')		
	output_log.write("[thickness v" + str(version_nb) + "]\n")
	output_log.write("\nThis folder and its content were created using the following command:\n\n")
	tmp_log="python thickness.py"
	for c in sys.argv[1:]:
		tmp_log+=" " + c
	output_log.write(tmp_log + "\n")
	output_log.close()
	#copy input files
	#----------------
	if args.colour_file!="no":
		shutil.copy2(args.colour_file,args.output_folder + "/")
	if args.selection_file_ff!="no":
		shutil.copy2(args.selection_file_ff,args.output_folder + "/")

################################################################################################################################################
# DATA LOADING
################################################################################################################################################

# Load universe
#==============
if args.xtcfilename=="no":
	print "\nLoading file..."
	U=Universe(args.grofilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=1
	nb_frames_processed=1
else:
	print "\nLoading trajectory..."
	U=Universe(args.grofilename, args.xtcfilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=U.trajectory.numframes
	nb_frames_processed=0
	U.trajectory.rewind()

# Identify ff lipids
#===================
leaflet_selection_string={}
leaflet_selection_string['martini']="name PO4 or name PO3 or name B1A"			#martini
leaflet_selection_string['gromos']="to do"										#gromos
leaflet_selection_string['charmm']="to do"										#charmm
lipids_ff_nb=0
lipids_ff_sele={}
lipids_ff_info={}
lipids_ff_species=[]
lipids_ff_leaflet=[]
lipids_ff_u2l_index=[]
lipids_ff_l2u_index=[]
#case: read specified ff lipids selection file
if args.selection_file_ff!="no":
	print "\nReading selection file for flipflopping lipids..."
	with open(args.selection_file_ff) as f:
		lines = f.readlines()
	lipids_ff_nb=len(lines)
	print " -found " + str(lipids_ff_nb) + " flipflopping lipids"
	sele_all_nff_string=str(leaflet_selection_string[args.forcefield_opt]) + " and not ("
	for l in range(0,lipids_ff_nb):
		try:
			#read the 3 comma separated field
			l_type=lines[l].split(',')[0]
			l_indx=int(lines[l].split(',')[1])
			l_start=lines[l].split(',')[2][0:-1]
	
			#build leaflet dictionary
			if l_start not in lipids_ff_leaflet:
				lipids_ff_leaflet.append(l_start)
	
			#create index list of u2l and l2u ff lipids
			if l_start=="upper":
				lipids_ff_u2l_index.append(l)
			elif l_start=="lower":
				lipids_ff_l2u_index.append(l)
			else:
				print "unknown starting leaflet '" + str(l_start) + "'."
				sys.exit(1)

			#build specie dictionary
			if l_type not in lipids_ff_species:
				lipids_ff_species.append(l_type)
	
			#build MDAnalysis atom group
			lipids_ff_info[l]=[l_type,l_indx,l_start]
			lipids_ff_sele[l]=U.selectAtoms("resname " + str(l_type) + " and resid " + str(l_indx))
			if lipids_ff_sele[l].numberOfAtoms()==0:
				sys.exit(1)
	
			#build selection string to select all PO4 without the flipflopping ones
			if l==0:
				sele_all_nff_string+="(resname " + str(l_type) + " and resid " + str(l_indx) + ")"
			else:
				sele_all_nff_string+=" or (resname " + str(l_type) + " and resid " + str(l_indx) + ")"
		except:
			print "Error: invalid flipflopping lipid selection string."
			sys.exit(1)
	sele_all_nff_string+=")"
#case: no ff lipids selection file specified
else:
	sele_all_nff_string=str(leaflet_selection_string[args.forcefield_opt])

# Identify nff leaflets
#======================
print "\nIdentifying leaflets..."
lipids_nff_sele={}
lipids_nff_sele_nb={}
for l in ["lower","upper","both"]:
	lipids_nff_sele[l]={}
	lipids_nff_sele_nb[l]={}
#identify lipids leaflet groups
if args.cutoff_leaflet:
	print " -optimising cutoff..."
	cutoff_value=MDAnalysis.analysis.leaflet.optimize_cutoff(U, sele_all_nff_string)
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, sele_all_nff_string, cutoff_value[0])
else:
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, sele_all_nff_string)
#process groups
if numpy.shape(L.groups())[0]<2:
	print "Error: imposssible to identify 2 leaflets."
	sys.exit(1)
else:
	if L.group(0).centerOfGeometry()[2] > L.group(1).centerOfGeometry()[2]:
		lipids_nff_sele["upper"]["all"]=L.group(0)
		lipids_nff_sele["lower"]["all"]=L.group(1)
	
	else:
		lipids_nff_sele["upper"]["all"]=L.group(1)
		lipids_nff_sele["lower"]["all"]=L.group(0)
	for l in ["lower","upper"]:
		lipids_nff_sele_nb[l]["all"]=lipids_nff_sele[l]["all"].numberOfResidues()
	if numpy.shape(L.groups())[0]==2:
		print " -found 2 leaflets: ", lipids_nff_sele["upper"]["all"].numberOfResidues(), '(upper) and ', lipids_nff_sele["lower"]["all"].numberOfResidues(), '(lower) lipids'
	else:
		other_lipids=0
		for g in range(2, numpy.shape(L.groups())[0]):
			other_lipids+=L.group(g).numberOfResidues()
		print " -found " + str(numpy.shape(L.groups())[0]) + " groups: " + str(lipids_nff_sele["upper"]["all"].numberOfResidues()) + "(upper), " + str(lipids_nff_sele["lower"]["all"].numberOfResidues()) + "(lower) and " + str(other_lipids) + " (others) lipids respectively"
lipids_nff_sele["both"]["all"]=lipids_nff_sele["lower"]["all"]+lipids_nff_sele["upper"]["all"]
lipids_nff_sele_nb["both"]["all"]=lipids_nff_sele["both"]["all"].numberOfResidues()

print "\nInitialising data structures..."

# Identify lipid species
#=======================
lipids_nff_species={}
lipids_nff_species_nb={}
for l in ["lower","upper","both"]:
	lipids_nff_species[l]=list(numpy.unique(lipids_nff_sele[l]["all"].resnames()))
	lipids_nff_species_nb[l]=numpy.size(lipids_nff_species[l])
membrane_comp={}
for l in ["lower","upper"]:
	membrane_comp[l]="  -" + str(l) + " (" + str(lipids_nff_sele[l]["all"].numberOfResidues()) + " lipids): "
	for s in lipids_nff_species[l]:
		membrane_comp[l]+= str(s) + "(" + str(lipids_nff_sele[l]["all"].selectAtoms("resname " + str(s)).numberOfResidues()) + "),"

#associate colours to lipids
#===========================
#color maps dictionaries
colours_lipids_nb=0
colours_lipids={}
colours_lipids_list=[]
colours_lipids_map="jet"
colormaps_possible=['Spectral', 'summer', 'coolwarm', 'pink_r', 'Set1', 'Set2', 'Set3', 'brg_r', 'Dark2', 'hot', 'PuOr_r', 'afmhot_r', 'terrain_r', 'PuBuGn_r', 'RdPu', 'gist_ncar_r', 'gist_yarg_r', 'Dark2_r', 'YlGnBu', 'RdYlBu', 'hot_r', 'gist_rainbow_r', 'gist_stern', 'gnuplot_r', 'cool_r', 'cool', 'gray', 'copper_r', 'Greens_r', 'GnBu', 'gist_ncar', 'spring_r', 'gist_rainbow', 'RdYlBu_r', 'gist_heat_r', 'OrRd_r', 'CMRmap', 'bone', 'gist_stern_r', 'RdYlGn', 'Pastel2_r', 'spring', 'terrain', 'YlOrRd_r', 'Set2_r', 'winter_r', 'PuBu', 'RdGy_r', 'spectral', 'flag_r', 'jet_r', 'RdPu_r', 'Purples_r', 'gist_yarg', 'BuGn', 'Paired_r', 'hsv_r', 'bwr', 'cubehelix', 'YlOrRd', 'Greens', 'PRGn', 'gist_heat', 'spectral_r', 'Paired', 'hsv', 'Oranges_r', 'prism_r', 'Pastel2', 'Pastel1_r', 'Pastel1', 'gray_r', 'PuRd_r', 'Spectral_r', 'gnuplot2_r', 'BuPu', 'YlGnBu_r', 'copper', 'gist_earth_r', 'Set3_r', 'OrRd', 'PuBu_r', 'ocean_r', 'brg', 'gnuplot2', 'jet', 'bone_r', 'gist_earth', 'Oranges', 'RdYlGn_r', 'PiYG', 'CMRmap_r', 'YlGn', 'binary_r', 'gist_gray_r', 'Accent', 'BuPu_r', 'gist_gray', 'flag', 'seismic_r', 'RdBu_r', 'BrBG', 'Reds', 'BuGn_r', 'summer_r', 'GnBu_r', 'BrBG_r', 'Reds_r', 'RdGy', 'PuRd', 'Accent_r', 'Blues', 'Greys', 'autumn', 'cubehelix_r', 'nipy_spectral_r', 'PRGn_r', 'Greys_r', 'pink', 'binary', 'winter', 'gnuplot', 'RdBu', 'prism', 'YlOrBr', 'coolwarm_r', 'rainbow_r', 'rainbow', 'PiYG_r', 'YlGn_r', 'Blues_r', 'YlOrBr_r', 'seismic', 'Purples', 'bwr_r', 'autumn_r', 'ocean', 'Set1_r', 'PuOr', 'PuBuGn', 'nipy_spectral', 'afmhot']
#case: group definition file
#---------------------------
if args.colour_file!="no":
	
	print "\nReading colour definition file..."
	with open(args.colour_file) as f:
		lines = f.readlines()
	colours_lipids_nb=len(lines)
	for line_index in range(0,colours_lipids_nb):
		l_content=lines[line_index].split(',')
		colours_lipids[l_content[0]]=l_content[1][:-1]					#to get rid of the returning char
	
	#display results
	print " -found the following colours definition:"
	for s in colours_lipids.keys():
		print " -" + str(s) + ": " + str(colours_lipids[s])

	#check if a custom color map has been specified or not
	if colours_lipids_nb>1 and len(numpy.unique(colours_lipids.values()))==1:
		if numpy.unique(colours_lipids.values())[0] in colormaps_possible:
			colours_lipids_map=numpy.unique(colours_lipids.values())[0]
		else:
			print "Error: either the same color was specified for all species or the color map '" + str(numpy.unique(colours_lipids.values())[0]) + "' is not valid."
			sys.exit(1)
	else:
		colours_lipids_map="custom"
		
	#check that all detected species have a colour specified
	for s in lipids_nff_species["both"]:
		if s not in colours_lipids.keys():
			print "Error: no colour specified for " + str(s) + "."
			sys.exit(1)

#case: generate colours from jet colour map
#------------------------------------------
if colours_lipids_map!="custom":
	tmp_cmap=cm.get_cmap(colours_lipids_map)
	colours_lipids_value=tmp_cmap(numpy.linspace(0, 1, len(lipids_nff_species["both"])))
	for l_index in range(0, len(lipids_nff_species["both"])):
		colours_lipids[lipids_nff_species["both"][l_index]]=colours_lipids_value[l_index]

################################################################################################################################################
# DATA STRUCTURE
################################################################################################################################################

#time
#----
time_stamp={}
time_sorted=[]
time_smoothed=[]

#lipids: thickness and selections
#--------------------------------
lipids_nff_thickness={}
lipids_nff_selection={}
lipids_nff_selection_VMD_string={}
lipids_nff_thickness["all_values"]={}
lipids_nff_thickness["all_values"]["all_frames"]=[]
for l in ["lower","upper"]:
	lipids_nff_thickness[l]={}
	lipids_nff_selection[l]={}
	lipids_nff_selection_VMD_string[l]={}
	tmp_resnames=lipids_nff_sele[l]["all"].resnames()
	tmp_resnums=lipids_nff_sele[l]["all"].resnums()
	for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
		lipids_nff_thickness[l][r_index]=[]
		lipids_nff_selection[l][r_index]=lipids_nff_sele[l]["all"].selectAtoms("resname " + str(tmp_resnames[r_index]) + " and resnum " + str(tmp_resnums[r_index])).residues.atoms
		lipids_nff_selection_VMD_string[l][r_index]="resname " + str(tmp_resnames[r_index]) + " and resid " + str(tmp_resnums[r_index])

#thickness: average by specie
#----------------------------
lipids_nff_thickness_avg={}
lipids_nff_thickness_std={}
lipids_nff_thickness_avg_smoothed={}
lipids_nff_thickness_std_smoothed={}
lipids_nff_thickness_avg["all"]=[]
lipids_nff_thickness_std["all"]=[]
for s in lipids_nff_species["both"]:
	lipids_nff_thickness_avg[s]=[]
	lipids_nff_thickness_std[s]=[]
	lipids_nff_thickness_avg_smoothed[s]=[]
	lipids_nff_thickness_std_smoothed[s]=[]
	
################################################################################################################################################
# FUNCTIONS: core
################################################################################################################################################

def calc_thickness(f_index):
	
	#array of associated thickness
	tmp_dist_t2b_dist=MDAnalysis.analysis.distances.distance_array(lipids_nff_sele["upper"]["all"].coordinates(), lipids_nff_sele["lower"]["all"].coordinates(), U.dimensions)
	tmp_dist_b2t_dist=MDAnalysis.analysis.distances.distance_array(lipids_nff_sele["lower"]["all"].coordinates(), lipids_nff_sele["upper"]["all"].coordinates(), U.dimensions)	
	tmp_dist_t2b_dist.sort()
	tmp_dist_b2t_dist.sort()
	tmp_dist_t2b_dist=tmp_dist_t2b_dist[:,:args.thick_nb_neighbours]
	tmp_dist_b2t_dist=tmp_dist_b2t_dist[:,:args.thick_nb_neighbours]
	tmp_dist_t2b_avg=numpy.zeros((lipids_nff_sele["upper"]["all"].numberOfResidues(),1))
	tmp_dist_b2t_avg=numpy.zeros((lipids_nff_sele["lower"]["all"].numberOfResidues(),1))
	tmp_dist_t2b_avg[:,0]=numpy.average(tmp_dist_t2b_dist, axis=1)
	tmp_dist_b2t_avg[:,0]=numpy.average(tmp_dist_b2t_dist, axis=1)
	
	#initialise tmp structure for specie averaging
	tmp_specie={}
	tmp_specie["all"]=[]
	lipids_nff_thickness["all_values"][f_index]=[]
	for s in lipids_nff_species["both"]:
		tmp_specie[s]=[]
		
	#store data for each particle: upper leaflet
	tmp_resnames=lipids_nff_sele["upper"]["all"].resnames()
	for r_index in range(0,lipids_nff_sele_nb["upper"]["all"]):
		tmp_specie[tmp_resnames[r_index]].append(tmp_dist_t2b_avg[r_index,0])
		tmp_specie["all"].append(tmp_dist_t2b_avg[r_index,0])
		lipids_nff_thickness["upper"][r_index].append(tmp_dist_t2b_avg[r_index,0])
		lipids_nff_thickness["all_values"][f_index].append(tmp_dist_t2b_avg[r_index,0])
		lipids_nff_thickness["all_values"]["all_frames"].append(tmp_dist_t2b_avg[r_index,0])
		
	#store data for each particle: lower leaflet
	tmp_resnames=lipids_nff_sele["lower"]["all"].resnames()
	for r_index in range(0,lipids_nff_sele_nb["lower"]["all"]):
		tmp_specie[tmp_resnames[r_index]].append(tmp_dist_b2t_avg[r_index,0])
		tmp_specie["all"].append(tmp_dist_b2t_avg[r_index,0])
		lipids_nff_thickness["lower"][r_index].append(tmp_dist_b2t_avg[r_index,0])
		lipids_nff_thickness["all_values"][f_index].append(tmp_dist_b2t_avg[r_index,0])
		lipids_nff_thickness["all_values"]["all_frames"].append(tmp_dist_b2t_avg[r_index,0])
	
	#store average data for each specie
	for s in lipids_nff_species["both"]:
		lipids_nff_thickness_avg[s].append(numpy.average(tmp_specie[s]))
		lipids_nff_thickness_std[s].append(numpy.std(tmp_specie[s]))
	lipids_nff_thickness_avg["all"].append(numpy.average(tmp_specie["all"]))
	lipids_nff_thickness_std["all"].append(numpy.std(tmp_specie["all"]))
	
	return
def rolling_avg(loc_list):
	
	loc_arr=numpy.asarray(loc_list)
	shape=(loc_arr.shape[-1]-args.nb_smoothing+1,args.nb_smoothing)
	strides=(loc_arr.strides[-1],loc_arr.strides[-1])   	
	return numpy.average(numpy.lib.stride_tricks.as_strided(loc_arr, shape=shape, strides=strides), -1)
def smooth_data():
	
	global time_smoothed
	
	#sort data into ordered lists
	#----------------------------
	for frame in sorted(time_stamp.keys()):
		time_sorted.append(time_stamp[frame])
	
	#calculate running average on sorted lists
	#-----------------------------------------
	if args.nb_smoothing>1:
		time_smoothed=rolling_avg(time_sorted)	
		for s in lipids_nff_species["both"]:
			lipids_nff_thickness_avg_smoothed[s]=rolling_avg(lipids_nff_thickness_avg[s])
			lipids_nff_thickness_std_smoothed[s]=rolling_avg(lipids_nff_thickness_std[s])
		lipids_nff_thickness_avg_smoothed["all"]=rolling_avg(lipids_nff_thickness_avg["all"])
		lipids_nff_thickness_std_smoothed["all"]=rolling_avg(lipids_nff_thickness_std["all"])
	
	return

################################################################################################################################################
# FUNCTIONS: outputs
################################################################################################################################################

#case: xtc file
#==============
def write_thickness_nff_xvg():
	
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_species/xvg/1_2_thickness_nff_species.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_species/xvg/1_2_thickness_nff_species.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by thickness v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_2_thickness_nff_species.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of bilayer thickness by lipid specie\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"thickness\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(lipids_nff_species_nb["both"]) + "\n")
	for s_index in range(0,lipids_nff_species_nb["both"]):
		s=lipids_nff_species["both"][s_index]
		output_xvg.write("@ s" + str(s_index) + " legend \"" + str(s) + " (avg)\"\n")
		output_txt.write("1_2_thickness_nff_species.xvg," + str(s_index+1) + "," + str(s) + " (avg)," + mcolors.rgb2hex(colours_lipids[s]) + "\n")
	for s_index in range(0,lipids_nff_species_nb["both"]):
		s=lipids_nff_species["both"][s_index]
		output_xvg.write("@ s" + str(lipids_nff_species_nb["both"]+s_index) + " legend \"" + str(s) + " (std)\"\n")
		output_txt.write("1_2_thickness_nff_species.xvg," + str(lipids_nff_species_nb["both"]+s_index+1) + "," + str(s) + " (std)," + mcolors.rgb2hex(colours_lipids[s]) + "\n")
	output_txt.close()
	for frame in sorted(time_stamp.iterkeys()):
		results=str(time_stamp[frame])
		frame_index=sorted(time_stamp.keys()).index(frame)
		for s in lipids_nff_species["both"]:
			results+="	" + str(round(lipids_nff_thickness_avg[s][frame_index],2))
		for s in lipids_nff_species["both"]:
			results+="	" + str(round(lipids_nff_thickness_std[s][frame_index],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	return
def write_thickness_nff_xvg_smoothed():
	
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_species/smoothed/xvg/1_4_thickness_nff_species_smoothed.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_species/smoothed/xvg/1_4_thickness_nff_species_smoothed.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by thickness v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_4_thickness_nff_species_smoothed.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of bilayer thickness by lipid specie\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"thickness\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(lipids_nff_species_nb["both"]) + "\n")
	for s_index in range(0,lipids_nff_species_nb["both"]):
		s=lipids_nff_species["both"][s_index]
		output_xvg.write("@ s" + str(s_index) + " legend \"" + str(s) + " (avg)\"\n")
		output_txt.write("1_4_thickness_nff_species_smoothed.xvg," + str(s_index+1) + "," + str(s) + " (avg)," + mcolors.rgb2hex(colours_lipids[s]) + "\n")
	for s_index in range(0,lipids_nff_species_nb["both"]):
		s=lipids_nff_species["both"][s_index]
		output_xvg.write("@ s" + str(lipids_nff_species_nb["both"]+s_index) + " legend \"" + str(s) + " (std)\"\n")
		output_txt.write("1_4_thickness_nff_species_smoothed.xvg," + str(lipids_nff_species_nb["both"]+s_index+1) + "," + str(s) + " (std)," + mcolors.rgb2hex(colours_lipids[s]) + "\n")
	output_txt.close()
	for frame_index in range(0, len(time_smoothed)):
		results=str(time_smoothed[frame_index])
		for s in lipids_nff_species["both"]:
			results+="	" + str(round(lipids_nff_thickness_avg_smoothed[s][frame_index],2))
		for s in lipids_nff_species["both"]:
			results+="	" + str(round(lipids_nff_thickness_std_smoothed[s][frame_index],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	return
def graph_thickness_nff_xvg():
	
	#create filenames
	#----------------
	filename_png=os.getcwd() + '/' + str(args.output_folder) + '/1_species/png/1_1_thickness_nff.png'
	filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/1_species/1_1_thickness_nff.svg'
	
	#create figure
	#-------------
	fig=plt.figure(figsize=(8, 4))
	fig.suptitle("Evolution of lipid bilayer thickness")
					
	#plot data:
	#----------
	ax1 = fig.add_subplot(111)
	p_upper={}
	for s in lipids_nff_species["both"]:
		p_upper[s]=plt.plot(time_sorted, lipids_nff_thickness_avg[s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_upper[str(s + "_err")]=plt.fill_between(time_sorted, numpy.asarray(lipids_nff_thickness_avg[s])-numpy.asarray(lipids_nff_thickness_std[s]), numpy.asarray(lipids_nff_thickness_avg[s])+numpy.asarray(lipids_nff_thickness_std[s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax1.legend(prop=fontP)
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('thickness ($\AA$)', fontsize="small")
	
	#save figure
	#-----------
	ax1.set_ylim(numpy.min(lipids_nff_thickness["all_values"]["all_frames"]), numpy.max(lipids_nff_thickness["all_values"]["all_frames"]))
	ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
	plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
	fig.savefig(filename_png)
	fig.savefig(filename_svg)
	plt.close()
	
	return
def graph_thickness_nff_xvg_smoothed():
	
	#create filenames
	#----------------
	filename_png=os.getcwd() + '/' + str(args.output_folder) + '/1_species/smoothed/png/1_3_thickness_nff.png'
	filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/1_species/smoothed/1_3_thickness_nff.svg'
	
	#create figure
	#-------------
	fig=plt.figure(figsize=(8, 4))
	fig.suptitle("Evolution of lipid bilayer thickness")
					
	#plot data:
	#----------
	ax1 = fig.add_subplot(111)
	p_upper={}
	for s in lipids_nff_species["both"]:
		p_upper[s]=plt.plot(time_smoothed, lipids_nff_thickness_avg_smoothed[s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_upper[str(s + "_err")]=plt.fill_between(time_smoothed, numpy.asarray(lipids_nff_thickness_avg_smoothed[s])-numpy.asarray(lipids_nff_thickness_std_smoothed[s]), numpy.asarray(lipids_nff_thickness_avg_smoothed[s])+numpy.asarray(lipids_nff_thickness_std_smoothed[s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax1.legend(prop=fontP)
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('thickness ($\AA$)', fontsize="small")
	
	#save figure
	#-----------
	ax1.set_ylim(numpy.average(lipids_nff_thickness["all_values"]["all_frames"])-5, numpy.average(lipids_nff_thickness["all_values"]["all_frames"])+5)
	ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
	plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
	fig.savefig(filename_png)
	fig.savefig(filename_svg)
	plt.close()
	
	return

#annotations
#===========
def write_frame_stat(f_nb, f_index, t):

	#case: gro file or xtc summary
	#=============================
	if f_index=="all" and t=="all":	
		#create file
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/1_species/1_0_thickness_nff.stat'
		output_stat = open(filename_details, 'w')		
		output_stat.write("[thickness statistics - written by thickness v" + str(version_nb) + "]\n")
		output_stat.write("\n")
		
		#general info
		output_stat.write("1. membrane composition: \n")
		output_stat.write(membrane_comp["upper"][:-1] + "\n")
		output_stat.write(membrane_comp["lower"][:-1] + "\n")	
		if args.xtcfilename!="no":
			output_stat.write("\n")
			output_stat.write("2. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
		output_stat.write("\n")
		
		#results data
		output_stat.write("Bilayer thickness:\n")
		output_stat.write("-----------------\n")
		output_stat.write("avg=" + str(round(numpy.average(lipids_nff_thickness["all_values"]["all_frames"]),2)) + "\n")
		output_stat.write("std=" + str(round(numpy.std(lipids_nff_thickness["all_values"]["all_frames"]),2)) + "\n")
		output_stat.write("max=" + str(round(numpy.max(lipids_nff_thickness["all_values"]["all_frames"]),2)) + "\n")
		output_stat.write("min=" + str(round(numpy.min(lipids_nff_thickness["all_values"]["all_frames"]),2)) + "\n")
		output_stat.write("\n")
		output_stat.write("Average bilayer thickness for each specie:\n")
		output_stat.write("------------------------------------------\n")
		for s in lipids_nff_species["both"]:
			output_stat.write(str(s) + "	" + str(round(numpy.average(lipids_nff_thickness_avg[s]),2)) + "\n")
		output_stat.write("\n")
		output_stat.close()

	#case: xtc snapshot
	#==================
	else:
		#create file
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_thickness_' + str(int(t)).zfill(5) + 'ns.stat'
		output_stat = open(filename_details, 'w')		
		output_stat.write("[thickness statistics - written by thickness v" + str(version_nb) + "]\n")
		output_stat.write("\n")
		
		#general info
		output_stat.write("1. membrane composition: \n")
		output_stat.write(membrane_comp["upper"][:-1] + "\n")
		output_stat.write(membrane_comp["lower"][:-1] + "\n")	
		output_stat.write("\n")
		output_stat.write("2. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
		output_stat.write("\n")
		output_stat.write("3. time: " + str(t) + "ns (frame " + str(f_nb) + "/" + str(nb_frames_xtc) + ")\n")		
		output_stat.write("\n")
		
		#results data
		output_stat.write("Bilayer thickness:\n")
		output_stat.write("-----------------\n")
		output_stat.write("avg=" + str(round(numpy.average(lipids_nff_thickness["all_values"][f_index]),2)) + "\n")
		output_stat.write("std=" + str(round(numpy.std(lipids_nff_thickness["all_values"][f_index]),2)) + "\n")
		output_stat.write("max=" + str(round(numpy.max(lipids_nff_thickness["all_values"][f_index]),2)) + "\n")
		output_stat.write("min=" + str(round(numpy.min(lipids_nff_thickness["all_values"][f_index]),2)) + "\n")
		output_stat.write("\n")
		output_stat.write("Average bilayer thickness for each specie:\n")
		output_stat.write("------------------------------------------\n")
		for s in lipids_nff_species["both"]:
			output_stat.write(str(s) + "	" + str(round(lipids_nff_thickness_avg[s][f_index],2)) + "	(" + str(round(lipids_nff_thickness_std[s][f_index],2)) + ")\n")
		output_stat.write("\n")
		output_stat.close()		
	return
def write_frame_snapshot(f_index,t):

	#store order parameter info in beta factor field
	for l in ["lower","upper"]:
		for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
			lipids_nff_selection[l][r_index].set_bfactor(lipids_nff_thickness[l][r_index][f_index])
	
	#case: gro file
	if args.xtcfilename=="no":
		all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/2_snapshots/' + args.grofilename[:-4] + '_annotated_thickness', format="PDB")

	#case: xtc file
	else:
		tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_thickness_' + str(int(t)).zfill(5) + 'ns.pdb'
		W=Writer(tmp_name, nb_atoms)
		W.write(all_atoms)
	
	return
def write_frame_annotation(f_index,t):
	
	#create file
	if args.xtcfilename=="no":
		filename_details=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.grofilename[:-4] + '_annotated_thickness.txt'
	else:
		filename_details=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_thickness_' + str(int(t)).zfill(5) + 'ns.txt'
	output_stat = open(filename_details, 'w')		

	#create selection string
	tmp_sele_string=""
	for l in ["lower","upper"]:
		for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
			tmp_sele_string+="." + str(lipids_nff_selection_VMD_string[l][r_index])
	output_stat.write(tmp_sele_string[1:] + "\n")

	#write min and max boundaries of thickness
	output_stat.write(str(round(numpy.min(lipids_nff_thickness["all_values"][f_index]),2)) + ";" + str(round(numpy.max(lipids_nff_thickness["all_values"][f_index]),2)) + "\n")
	
	#ouptut thickness for each lipid
	tmp_thickness="1"
	for l in ["lower","upper"]:
		for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
			tmp_thickness+=";" + str(round(lipids_nff_thickness[l][r_index][f_index],2))
	output_stat.write(tmp_thickness + "\n")			
	output_stat.close()

	return
def write_xtc_snapshots():
	#NB: - this will always output the first and final frame snapshots
	#    - it will also intermediate frames according to the -w option	

	loc_nb_frames_processed=0
	for ts in U.trajectory:

		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -writing snapshots...   frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:
				if ((loc_nb_frames_processed) % args.frames_write_dt)==0 or loc_nb_frames_processed==nb_frames_processed-1:
					write_frame_stat(ts.frame, loc_nb_frames_processed, ts.time/float(1000))
					write_frame_snapshot(loc_nb_frames_processed, ts.time/float(1000))
					write_frame_annotation(loc_nb_frames_processed, ts.time/float(1000))
				loc_nb_frames_processed+=1
		
		#case: frames after specified time boundaries
		#--------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break

	print ''

	return
def write_xtc_annotation():
	
	#create file
	filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_VMD/' + args.xtcfilename[:-4] + '_annotated_thickness_dt' + str(args.frames_dt) + '.txt'
	output_stat = open(filename_details, 'w')		

	#create selection string
	tmp_VMD_sele_string=""
	for l in ["lower","upper"]:
		for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
			tmp_VMD_sele_string+="." + str(lipids_nff_selection_VMD_string[l][r_index])
	output_stat.write(tmp_VMD_sele_string[1:] + "\n")

	#write min and max boundaries of thickness
	output_stat.write(str(round(numpy.min(lipids_nff_thickness["all_values"]["all_frames"]),2)) + ";" + str(round(numpy.max(lipids_nff_thickness["all_values"]["all_frames"]),2)) + "\n")
	
	#ouptut thickness for each lipid
	for frame in sorted(time_stamp.iterkeys()):
		tmp_thickness=str(frame)
		frame_index=sorted(time_stamp.keys()).index(frame)
		for l in ["lower","upper"]:
			for r_index in range(0,lipids_nff_sele_nb[l]["all"]):
				tmp_thickness+=";" + str(round(lipids_nff_thickness[l][r_index][frame_index],2))
		output_stat.write(tmp_thickness + "\n")
			
	output_stat.close()

	return

################################################################################################################################################
# ALGORITHM : Browse trajectory and process relevant frames
################################################################################################################################################

print "\nCalculating thickness..."

#case: structure only
#====================
if args.xtcfilename=="no":
	time_stamp[1]=0
	calc_thickness(0)

#case: browse xtc frames
#=======================
else:
	for ts in U.trajectory:
		
		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -processing frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '                      '  
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:						
				time_stamp[ts.frame]=ts.time/float(1000)
				calc_thickness(nb_frames_processed)
				nb_frames_processed+=1
			
		#case: frames after specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break
	print ''

################################################################################################################################################
# PRODUCE OUTPUTS
################################################################################################################################################

print "\nWriting outputs..."

#case: gro file
#==============
if args.xtcfilename=="no":
	print " -writing statistics..."
	write_frame_stat(1,"all","all")
	print " -writing annotated pdb..."
	write_frame_snapshot(0,0)
	write_frame_annotation(0,0)

#case: xtc file
#==============
else:
	#sort and if necessary smooth data
	smooth_data()
	#writing statistics
	print " -writing statistics..."
	write_frame_stat(1,"all","all")
	#output cluster snapshots
	write_xtc_snapshots()
	#write annotation files for VMD
	print " -writing VMD annotation files..."
	write_xtc_annotation()
	#write xvg and graphs
	print " -writing xvg and graphs..."
	write_thickness_nff_xvg()
	graph_thickness_nff_xvg()	
	if args.nb_smoothing>1:
		write_thickness_nff_xvg_smoothed()
		graph_thickness_nff_xvg_smoothed()
			
#exit
#====
print "\nFinished successfully! Check output in ./" + args.output_folder + "/"
print ""
sys.exit(0)
