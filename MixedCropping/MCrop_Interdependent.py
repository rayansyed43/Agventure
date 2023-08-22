#----------------------------------------- Importing Libraries -----------------------------------------

import time
import random
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors, style
from collections import Counter
from operator import itemgetter
from prettytable import PrettyTable
from deap import algorithms, base, tools, creator
style.use('seaborn')
start_time = time.time()

#-------------------------------------------- Reading Data ---------------------------------------------

# Reading CSV file
df = pd.read_csv('Gudur_Rythu_Bazar_2017.csv')

# np arrays of colomns
Harvest_time = df['Maturity_mo']
Harvest_time = np.array(Harvest_time)
Crop_name 	= df['Type']
Crop_name 	= np.array(Crop_name)
Culti_cost 	= df['Cost_Culti_acre']
Culti_cost 	= np.array(Culti_cost)
Root_depth 	= df['Root_Depth']
Root_depth 	= np.array(Root_depth)
Water_req 	= df['Water_Req']
Water_req 	= np.array(Water_req)
Price 	= df['RB_Rate_kg']
Price 	= np.array(Price)
Profit 	= df['Profit']
Profit 	= np.array(Profit)
Month 	= df['Month']
Month 	= np.array(Month)
Type 	= df['Type_Code']
Type 	= np.array(Type)

#--------------------------------------------- Variable info ----------------------------------------------

months_ = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_dict = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6, \
'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}
CM = datetime.datetime.now().month
CM_str = datetime.datetime.today().strftime('%B')

Debug = False
print_ = False

N 	= 300
M	= 6 		# No.of crops to decide
n_i = Type[0] 	# Lower limit of no.of crops
n_f	= Type[-1] 	# Upper limit of no.of crops / Total no.of crops
NGen 	= 20	# Number of generations/Number of itterations			
CXPB	= 0.5	# CXPB  is the probability with which two individuals are crossed
MUTPB 	= 0.2	# MUTPB is the probability for mutating an individual

# Weights to cal weighted avg
Profit_wt	= 0.7
Risk_wt 	= 0.3
Root_risk_wt	= 0.5
Water_risk_wt	= 0
volatile_wt		= 0.5

#----------------------------------------------- Fitness Function ----------------------------------------------
# Objective fun : [1] Maximize Profit
#				  [2] Mininize Risk 
# 			i.e : Max((W1*Profit +W2*(-Risk))/(W1+W2))
# Subject to constrains : [1] Harvest time.
#						  [2] Based on root system.
#						  [3] Based on water requirement.
#						  [4] Market Volatility.

def Fitness_value(individual, Current_month, Previous_H_m, Previous_R_d, Previous_W_r, m, profit_wt, risk_wt, \
	root_risk_wt, water_risk_wt, AllinOne, Debug):

	global profit
	global harvest_month
	global planting_month
	global harvest_time
	global root_depth
	global water_req

	profit = []
	harvest_month = []
	planting_month = []
	harvest_time = []
	root_depth = []
	water_req = []

	#---------------------------------------------- Estimating Profit -----------------------------------------

	if len(set(individual)) == m :
		for i in range(len(individual)):
			if len(Previous_H_m) == 0 : 
				current_month = [Current_month]*m
			else:
				current_month = list(map(lambda x: x + 1, Previous_H_m))
			harvest_month_itt = []
			planting_month_itt = []
			harvest_time_itt = []
			Crop = individual[i]
			for e in range(len(Type)):
				if Type[e]==Crop:
					type_id = e
					break
				else:
					pass
			profit_id = type_id + current_month[i] + Harvest_time[type_id] -1
			id_verify = current_month[i] + Harvest_time[type_id] -1
			if id_verify < 12:
				profit_i = Profit[profit_id]
				planting_month_itt=Month[profit_id - Harvest_time[type_id]]
				harvest_month_itt=Month[profit_id]
				harvest_time_itt=Harvest_time[profit_id]
				# break
			else:
				profit_i = Profit[type_id + profit_id%12]
				planting_month_itt=Month[type_id + profit_id%12 - Harvest_time[type_id]]
				harvest_month_itt=Month[type_id + profit_id%12]
				harvest_time_itt=Harvest_time[type_id + profit_id%12]
				# break
			profit.append(profit_i)
			planting_month.append(planting_month_itt)
			harvest_month.append(harvest_month_itt)
			root_depth.append(Root_depth[type_id])
			water_req.append(Water_req[type_id])
			harvest_time.append(harvest_time_itt)
	else:
		profit=[0]

	Profit_percent = sum(profit)/10**4

	#---------------------------------------------- Estimating Risk -------------------------------------------

	list_risk=[]

	# Risk due to competition over nitrogen from the soil
	# Diff root sys side by side in a same cycle
	# Diff root sys b/w corps of diff cycles
	# Diff root sys at any instant b/w present and previous cycles
	avg_abc_1 = []
	def Risk_root(Root_d):
		if len(Root_d) != 0:
			list_abc_1=[]
			for i in range(len(Root_d)-1):
				if Root_d[i] == Root_d[i+1] : list_abc_1.append(100)
				else : list_abc_1.append(0)
			avg_abc_1f = sum(list_abc_1)#/m
		else : avg_abc_1f = 100*m
		return avg_abc_1f

	# Risk in the same cycle
	avg_abc_1.append(Risk_root(root_depth))

	root_risk_cyc = []
	root_depth_cyc = []
	if len(Previous_H_m) !=0 and len(root_depth) !=0 :
		planting_mon = []
		for i in range(m): planting_mon.append(AllinOne[i][1]+1)
		previous_harvest_mon = []
		for i in range(m): previous_harvest_mon.append(AllinOne[i][1])
		previous_harvest_mon_max = max(previous_harvest_mon)
		previous_harvest_mon_min = min(previous_harvest_mon)

		# Risk at an instant with mix of crops in two cycles
		for i in range(previous_harvest_mon_max-previous_harvest_mon_min):
			verify_value = previous_harvest_mon_max-i
			root_depth_bwc = []
			for e in range(m):
				if planting_mon[e] <= verify_value : root_depth_bwc.append(root_depth[e])
				else : root_depth_bwc.append(AllinOne[e][5])
			root_depth_cyc.append(root_depth_bwc)
		for i in range(len(root_depth_cyc)):
			root_risk_cyc.append(Risk_root(root_depth_cyc[i]))
		avg_abc_1.append(sum(root_risk_cyc)/len(root_risk_cyc))

		# Risk in same line b/w crops of diff cycles
		previous_root = []
		for i in range(len(root_depth)): 
			if root_depth[i] != AllinOne[i][5]: previous_root.append(0)
			else : previous_root.append(100)
		if len(root_depth) != 0 : previous_root_avg = sum(previous_root)#/len(previous_root)
		else : previous_root_avg = 100*m
		avg_abc_1.append(previous_root_avg)
	else: pass

	list_risk.append(sum(avg_abc_1)/len(avg_abc_1))

	# Risk due to competition over water requirement
	# lower limit = m*20
	# upper limit = m*50
	list_abc_2=[]
	counts_water = Counter(water_req)
	per_L = counts_water['L']
	per_M = counts_water['M']
	per_H = counts_water['H']
	avg_abc_2 = 20*per_L + 30*per_M + 50*per_H
	# avg_abc_2 = 10*per_L + 20*per_M + 30*per_H

	list_risk.append(avg_abc_2)

	# Risk due to market volatility
	# Std(12monts)*Sqrt(12)
	list_abc_3 = []
	for i in range(len(individual)):
		price_id = (individual[i]-1)*12
		volatility_val = np.std(Price[price_id : price_id+12])*np.sqrt(12)
		list_abc_3.append(volatility_val)
	avg_abc_3 = sum(list_abc_3)/len(list_abc_3)

	list_risk.append(avg_abc_3)
	
	# risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1])/(root_risk_wt + water_risk_wt)
	# risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1])
	risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1] + volatile_wt*list_risk[2])
	# Risk_percent = risk
	Risk_percent = -risk

	#-----------------------------------------------------------------------------------------------------------
	
	# combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)/(profit_wt+risk_wt)
	# combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)
	combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)*10**2
	# Debug = True
	if Debug == True:
		print('-- Debugging --')
		print(root_risk_cyc, root_depth_cyc)
		print('Profit_val 	: %s \nRisk_val 	: %s \nCombined_val 	: %s \nRisk_root 	: %s \nRisk_water 	: %s \
			\nVolatility 	: %s \nRisk_list 	: %s' %(Profit_percent, Risk_percent, combined_val, avg_abc_1, avg_abc_2, \
				list_abc_3, list_risk) )
	else: pass

	# return sum(profit), risk
	return combined_val, 

# ------------------------------------------------ Creating class -----------------------------------------------

# creator.create('FitnessMax', base.Fitness, weights = (1.0, -1.0))
creator.create('FitnessMax', base.Fitness, weights = (1.0, ))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_value', random.sample, range(n_i, n_f+1), M)		# generator
# toolbox.register('attr_value', random.randint, n_i, n_f)	# generator

#--------------------------------------------- Evolution operation ----------------------------------------------

def Evolution(m, n, CXPB, MUTPB, NGen, Current_month, Previous_H_m, Previous_R_d, Previous_W_r, profit_wt, risk_wt, \
	root_risk_wt, water_risk_wt, AllinOne):

	Max_=[]
	Avg_=[]
	Std_=[]
	Debug = False

	# Structure initializers
	# toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_value, m)	
	toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_value)
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)

	# genetic operators required for the evolution
	toolbox.register('evaluate', Fitness_value, Current_month = Current_month, Previous_H_m = Previous_H_m, \
		Previous_R_d = Previous_R_d, Previous_W_r = Previous_W_r, m = m, profit_wt = profit_wt, risk_wt = risk_wt, \
		root_risk_wt = root_risk_wt, water_risk_wt = water_risk_wt, AllinOne = AllinOne, Debug = Debug)
	toolbox.register('mate', tools.cxTwoPoint)
	toolbox.register('mutate', tools.mutUniformInt, low=n_i, up=n_f, indpb=0.2)
	toolbox.register('select', tools.selTournament, tournsize=3)

	# create an initial population of 'n' individuals
	pop = toolbox.population(n)

	if print_ == True: print("Start of evolution")
	
	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	
	if print_ == True: print("  Evaluated %i individuals" % len(pop))

	# Extracting all the fitnesses of 
	fits = [ind.fitness.values[0] for ind in pop]

	# Begin the evolution
	for g in range(NGen):

		gen = g+1
		if print_ == True: print("-- Generation %i --" % gen)
		
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
	
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):

			# cross two individuals with probability CXPB
			if random.random() < CXPB:
				toolbox.mate(child1, child2)

				# fitness values of the children
				# must be recalculated later
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:

			# mutate an individual with probability MUTPB
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
	
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		if print_ == True: print("  Evaluated %i individuals" % len(invalid_ind))
		
		# The population is entirely replaced by the offspring
		pop[:] = offspring
		
		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]
		
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5

		Max_.append(max(fits))
		Avg_.append(mean)
		Std_.append(std)

		if print_ == True: print("  Min %s" % min(fits))
		if print_ == True: print("  Max %s" % max(fits))
		if print_ == True: print("  Avg %s" % mean)
		if print_ == True: print("  Std %s" % std, '\n')

	if print_ == True: print("-- End of successful evolution --")

	Best = tools.selBest(pop, 1)[0]	

	#---------------------------------------------- Visualisation --------------------------------------------

	# x_ = np.arange(1,len(Max_)+1)
	# plt.bar(x_-0.2, Max_, width = 0.2,align='center', label='Max')
	# plt.bar(x_, Avg_, width = 0.2,align='center', label='Avg')
	# plt.bar(x_+0.2, Std_, width = 0.2,align='center', label='Std')
	# plt.axis([0, NGen+1, 0, 1.4*max(Max_)])
	# plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))
	# plt.xlabel('Generation')
	# plt.ylabel('Total Profit')
	# plt.title('Max - Avg - Std')
	# plt.legend()
	# plt.show()

	#---------------------------------------- Storing output to 't' ------------------------------------------

	# To access global variables, To store output of each crop of 'Best' individual
	# Debug = True
	Fitness_value(Best, Current_month, Previous_H_m, Previous_R_d, Previous_W_r, m, profit_wt, risk_wt, \
		root_risk_wt, water_risk_wt, AllinOne, Debug)

	# Data in table format
	Total_profit = 0
	t = PrettyTable(['Crop','Planting Month', 'Harvest Month', 'Root Sys', \
		'Water Req', 'Culti Cost', 'Profit'])
	for i in range(len(Best)):
		val = Best[i]
		t.add_row([Crop_name[val*12-1], planting_month[i], harvest_month[i], \
		Root_depth[val*12-1], Water_req[val*12-1], len(harvest_month[i])*Culti_cost[val*12-1], profit[i]])
		Total_profit = Total_profit + profit[i]

	return Best, t, Total_profit, harvest_month, planting_month, harvest_time, root_depth, water_req

# ======================================== Running Genetic Algorithm =========================================

count_ga=0
TotalProfit = []
TotalFitness = []
visual = []
allinone = []
PHM = []
PRD = []
PWR = []

while True:
	visual_i = []
	if len(visual) != 0 : allinone = visual[-1]
	print('\nCrop cycle : ', count_ga+1)
	Best_ind, t_ind, T_p_ind, H_m_ind, P_n_ind, H_t, R_d, W_r = \
	Evolution(M, N, CXPB, MUTPB, NGen, CM, PHM, PRD, PWR, Profit_wt, Risk_wt, Root_risk_wt, Water_risk_wt, allinone)
	print("Best individual is %s, Fitness is %s" % (Best_ind, Best_ind.fitness.values))
	print(t_ind)
	print("Profit from cycle-%s : %s " % (count_ga+1, T_p_ind))
	TotalProfit.append(T_p_ind)
	TotalFitness.append(Best_ind.fitness.values[0])

	# appending visualisation parameters
	if count_ga == 0 :
		[visual_i.append([CM, CM+H_t[v], v+1, Best_ind[v], count_ga+1, R_d[v], W_r[v]]) for v in range(len(Best_ind))]
		visual.append(visual_i)
	else:
		[visual_i.append([visual[-1][v][1]+1, visual[-1][v][1]+1+H_t[v], v+1, Best_ind[v], count_ga+1, R_d[v], W_r[v]]) \
		for v in range(len(Best_ind))]
		visual.append(visual_i)

	# appending present cycles harvest months to decide next cycles planting months
	PHM = []
	PRD = []
	PWR = []
	for i in range(len(H_m_ind)): PHM.append(months_dict[H_m_ind[i]]); PRD.append(R_d); PWR.append(W_r)

	# break statment, break when 1yr of harvesting complets
	H_m_ind_1 = []
	[H_m_ind_1.append(visual[-1][i][1]-M) for i in range(len(visual[-1]))]
	H_m_ind_1=sorted(H_m_ind_1, key=int)
	if H_m_ind_1[0] >= 12: print('Total Profit : %s \nTotalFitness : %s '%(sum(TotalProfit), sum(TotalFitness))); break

	count_ga+=1

print('Time to ex : %ssec' %(time.time()-start_time))

#----------------------------------------------- Visualisation ------------------------------------------------

hex_list = []
i_ = 0
[hex_list.append(c) for c in colors.cnames]

for e in range(len(visual)):
	for i in range(len(visual[e])):
		plt.scatter(visual[e][i][0], visual[e][i][2],marker='>', color=hex_list[i_])
		plt.scatter(visual[e][i][1], visual[e][i][2],marker='o', color=hex_list[i_])
		plt.plot([visual[e][i][0], visual[e][i][1]], [visual[e][i][2], visual[e][i][2]], label=Crop_name[visual[e][i][3]*12-1], \
			color=hex_list[i_])
		plt.annotate(Crop_name[visual[e][i][3]*12-1], xy=(visual[e][i][0], visual[e][i][2]), xytext=(visual[e][i][0], \
			visual[e][i][2]+0.1), size = 8, ha='left', va='bottom', bbox=dict(boxstyle='round', edgecolor='none', \
			fc='lightsteelblue', alpha=0.5))
		plt.annotate('Cycle-'+str(visual[e][i][4])+str(', ')+str(visual[e][i][5]), \
			xy=(visual[e][i][0], visual[e][i][2]), xytext=(visual[e][i][0], visual[e][i][2]-0.2), size = 6, ha='left', \
			va='center', bbox=dict(boxstyle='round', edgecolor='none', fc='lightsteelblue', alpha=0.5), style='italic')
		i_+=1

plt.yticks(range(1, M+1), [str(x+1)+'st crop' for x in range(M)])
plt.xticks(range(1, 25), months_+months_+months_)
plt.axis([0, 25, -1, M+2])
plt.ylabel('Crops')
plt.xlabel('Months')
plt.title('Crop Cycles')
# plt.legend()
plt.show()

