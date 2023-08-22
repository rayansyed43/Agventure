#----------------------------------------- Importing Libraries -----------------------------------------

import time
import random
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from operator import add
from itertools import repeat
from collections import Counter
from prettytable import PrettyTable
from matplotlib import colors, style
from deap import base, tools, creator
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

# Returns integer value for the str input of month, Helper to the map function
def MonDic (mon): return months_dict[mon]

CM = datetime.datetime.now().month
CM_str = datetime.datetime.today().strftime('%B')	# Current month str

Debug = False
Print_ = True

N 	= 3000
C 	= 2			# No.of crops cycles
M	= 5 		# No.of crops to decide
n_i = Type[0] 	# Lower limit of no.of crops
n_f	= Type[-1] 	# Upper limit of no.of crops / Total no.of crops
NGen 	= 10	# Number of generations/Number of itterations			
CXPB	= 0.7	# CXPB  is the probability with which two individuals are crossed
MUTPB 	= 0.4	# MUTPB is the probability for mutating an individual
INDPB 	= 0.2	# INDPB is the probability for mutating each gen of an individual

# Weights to cal weighted avg
Profit_wt	= 0.5
Risk_wt 	= 0.5
Root_risk_wt	= 0.7
Water_risk_wt	= 0
Volatile_wt		= 0.3

#----------------------------------------------- Fitness Function ----------------------------------------------
# Objective fun : [1] Maximize Profit
#				  [2] Mininize Risk 
# 			i.e : Max((W1*Profit +W2*(-Risk))/(W1+W2))

# Subject to constrains : [1] Planting month.
#						  [2] Harvest time.
#						  [3] Based on root system.
#						  [4] Based on water requirement.
#						  [4] Market Volatility.

def Fitness_value(individual, Current_month, m, c, profit_wt, risk_wt, root_risk_wt, water_risk_wt, volatile_wt, debug):

	global AllinOne

	AllinOne = []
	profit_all = []
	Previous_H_m = []

	# ---------------------------------------------- Estimating Profit -----------------------------------------
	# Estimating profit based on Planting month & Harvest time.

	for i in range(c):
		profit = []
		harvest_month = []
		planting_month = []
		harvest_time = []
		root_depth = []
		water_req = []
		for e in range(m):
			if len(Previous_H_m) == 0 : current_month = [Current_month]*m
			# Planting month is previous cycle harvest month +1
			else: current_month = list(map(lambda x: x + 1, Previous_H_m))
			harvest_month_itt = []
			planting_month_itt = []
			harvest_time_itt = []
			Crop = individual[i][e]
			for j in range(len(Type)): 
				if Type[j] == Crop: type_id = j; break
			profit_id = type_id + current_month[e] + Harvest_time[type_id] -1
			id_verify = current_month[e] + Harvest_time[type_id] -1
			if id_verify < 12:
				profit_i = Profit[profit_id]
				planting_month_itt=Month[profit_id - Harvest_time[type_id]]
				harvest_month_itt=Month[profit_id]
				harvest_time_itt=Harvest_time[profit_id]
			else:
				profit_i = Profit[type_id + profit_id%12]
				planting_month_itt=Month[type_id + profit_id%12 - Harvest_time[type_id]]
				harvest_month_itt=Month[type_id + profit_id%12]
				harvest_time_itt=Harvest_time[type_id + profit_id%12]
			profit.append(profit_i)
			planting_month.append(planting_month_itt)
			harvest_month.append(harvest_month_itt)
			root_depth.append(Root_depth[type_id])
			water_req.append(Water_req[type_id])
			harvest_time.append(harvest_time_itt)
		Previous_H_m = list(map(MonDic, harvest_month))
		profit_all.append(profit)
		if i == 0 : phm = list(map(add, current_month, harvest_time))
		else : phm = list(map(add, list(map(add, AllinOne[-1][7], harvest_time)), [1]*m))
		AllinOne.append([list(individual[i]), planting_month, harvest_month, harvest_time, profit, root_depth, water_req, phm])

	# Total profit from this combination
	Profit_total = sum(map(sum,profit_all))/10**4

	#---------------------------------------------- Estimating Risk -------------------------------------------

	list_risk = []

	# Risk due to competition over nitrogen from the soil
	# Diff root sys side by side in a same cycle
	# Diff root sys b/w corps of diff cycles
	# Diff root sys at any instant b/w present and previous cycles

	def Risk_root(Root_d):
		risk_root_list = []
		for i in range(len(Root_d)-1):
			if Root_d[i] == Root_d[i+1] : risk_root_list.append(1)
			else : risk_root_list.append(0)
		# Root_risk_ind = sum(risk_root_list)/(len(risk_root_list)+1)
		Root_risk_ind = sum(risk_root_list)
		return Root_risk_ind

	avg_abc_1 = []

	# Risk at each month
	RR_instant = []
	for i in range(max(AllinOne[-1][7])-Current_month+1):
		root_sys_instant = []
		for e in range(m):
			for j in range(c):
				if Current_month+i <= AllinOne[j][7][e] :
					root_sys_instant.append(AllinOne[j][5][e])
					break
				elif j == c-1 : 
					root_sys_instant.append("Dummy"+str(e))
		RR_instant.append(Risk_root(root_sys_instant))

	# avg_abc_1.append(sum(RR_instant)/len(RR_instant))
	avg_abc_1.append(sum(RR_instant))

	# Risk in same line b/w crops of diff cycles
	previous_root = []
	for i in range(c-1):
		previous_root_itt = []
		for e in range(m):
			if AllinOne[i][5][e] == AllinOne[i+1][5][e] : previous_root_itt.append(1)
			else : previous_root_itt.append(0)
		# previous_root.append(sum(previous_root_itt)/len(previous_root_itt))
		previous_root.append(sum(previous_root_itt))
	# avg_abc_1.append(sum(previous_root)/len(previous_root))
	avg_abc_1.append(sum(previous_root))

	# list_risk.append(sum(avg_abc_1)/len(avg_abc_1))
	list_risk.append(sum(avg_abc_1)*100)

	# Risk due to competition over water requirement
	# lower limit = m*20
	# upper limit = m*50

	avg_abc_2 = []
	for i in range(c):
		counts_water = Counter(AllinOne[i][6])
		per_L = counts_water['L']
		per_M = counts_water['M']
		per_H = counts_water['H']
		avg_abc_2.append(20*per_L + 30*per_M + 50*per_H)

	list_risk.append(sum(avg_abc_2)/len(avg_abc_2))

	# Risk due to market volatility
	# Std(12monts)*Sqrt(12)

	avg_abc_3 = []
	for i in range(c):
		list_abc_3 = []
		for e in range(m):
			price_id = (AllinOne[i][0][e]-1)*12
			volatility_val = np.std(Price[price_id : price_id+12])*np.sqrt(12)
			list_abc_3.append(volatility_val)
		avg_abc_3.append(sum(list_abc_3)/len(list_abc_3))

	# list_risk.append(sum(avg_abc_3)/len(avg_abc_3))
	list_risk.append(sum(avg_abc_3))

	# Total risk from this combination
	risk = ( root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1] + volatile_wt*list_risk[2] )\
	/( root_risk_wt + water_risk_wt + volatile_wt )

	Risk_total = -risk

	#-----------------------------------------------------------------------------------------------------------

	Combined_value = (( profit_wt*Profit_total + risk_wt*Risk_total )/( profit_wt + risk_wt ))

	if debug == True:
		print('-- Debugging --')
		# print(AllinOne)
		print('Profit 		: %s \nRisk		: %s \nCombined_val 	: %s \nRisk_root 	: %s \nRisk_water 	: %s \
			\nVolatility 	: %s \nRisk_list 	: %s' %(Profit_total*10**4, Risk_total, Combined_value, avg_abc_1, \
				avg_abc_2, avg_abc_3, list_risk) )

	return Combined_value, 

# ---------------------------------------------- Custom Genetic operators -------------------------------------------

# Custom Initialiser to generate nd array of shape (c, m)
def NdIndividual(icls, low, high, m, c):
	indGenerator = []
	for i in range(c):
		indGenerator.append(random.sample(range(low, high+1), m))
	return icls(indGenerator)

# Custom Crossover function for nd arrays
def NdcxTwoPointX(ind1, ind2, m, c):
	for i in range(c):
		ind_itt1 = ind1[i]
		ind_itt2 = ind2[i]
		# print('----------------------------------')
		# print(ind_itt1, ind_itt2)
		cxpoint1 = random.randint(1, m)
		cxpoint2 = random.randint(1, m - 1)
		if cxpoint2 >= cxpoint1:
			cxpoint2 += 1
		else: 
			cxpoint1, cxpoint2 = cxpoint2, cxpoint1		# Swap the two cx points

		# print(cxpoint1, cxpoint2)

		cx1 = ind_itt1[cxpoint1:cxpoint2]
		cx2 = ind_itt2[cxpoint1:cxpoint2]

		ind_cx1 = ind_itt1[0 : cxpoint1] + ind_itt1[cxpoint2 : m]
		ind_cx2 = ind_itt2[0 : cxpoint1] + ind_itt2[cxpoint2 : m]

		for e in range(cxpoint2 - cxpoint1):
			for i in range(cxpoint2 - cxpoint1):
				if cx2[i] not in ind_itt1 :
					ind_itt1[cxpoint1+i] = cx2[i]
				elif cx2[i] in ind_cx1 :
					ind_cx1.append(cx2[i])
				elif cx2[i] in cx1 :
					pass
				elif cx2[i] not in ind_cx1 :
					ind_itt1[cxpoint1+i] = cx2[i]

		for e in range(cxpoint2 - cxpoint1):
			for i in range(cxpoint2 - cxpoint1):
				if cx1[i] not in ind_itt2 :
					ind_itt2[cxpoint1+i] = cx1[i]
				elif cx1[i] in ind_cx2 :
					ind_cx2.append(cx1[i])
				elif cx1[i] in cx2 :
					pass
				elif cx1[i] not in ind_cx2 :
					ind_itt2[cxpoint1+i] = cx1[i]

	return ind1, ind2

# # Custom Crossover function for nd arrays
# def NdcxTwoPointX(ind1, ind2, m, c):
# 	for i in range(c):
# 		ind_itt1 = ind1[i]
# 		ind_itt2 = ind2[i]
# 		cxpoint1 = random.randint(1, m)
# 		cxpoint2 = random.randint(1, m - 1)
# 		if cxpoint2 >= cxpoint1:
# 			cxpoint2 += 1
# 		else: 
# 			cxpoint1, cxpoint2 = cxpoint2, cxpoint1		# Swap the two cx points
# 		ind_itt1[cxpoint1:cxpoint2], ind_itt2[cxpoint1:cxpoint2] = \
# 		ind_itt2[cxpoint1:cxpoint2], ind_itt1[cxpoint1:cxpoint2]

# 		# Removing duplicates in child
# 		cx1 = ind_itt1[cxpoint1:cxpoint2]
# 		cx2 = ind_itt2[cxpoint1:cxpoint2]
# 		cx1ncx2 = list(set(cx1).intersection(cx2))
# 		cx1 = list(set(cx1).difference(cx1ncx2))
# 		cx2 = list(set(cx2).difference(cx1ncx2))
# 		verids = list(range(0,m))
# 		del(verids[cxpoint1:cxpoint2])
# 		idcx1 = 0
# 		idcx2 = 0		
# 		for e in verids: 
# 			if ind_itt1[e] in cx1 : ind_itt1[e] = cx2[idcx1]; idcx1+=1
# 			if ind_itt2[e] in cx2 : ind_itt2[e] = cx1[idcx2]; idcx2+=1
# 		# if len(set(ind_itt1)) != m : print(ind_itt1)		# Check for duplicates after CX
# 		# if len(set(ind_itt2)) != m : print(ind_itt2)

# 	return ind1, ind2

# Custom Mutate function for nd arrays
def NdmutUniformInt(individual, m, c, low, up, indpb):
	for i in range(c):
		mutate_sample = [*range(low, up+1)]		# To avoid duplicates
		for j in individual[i] : 
			while j in mutate_sample : mutate_sample.remove(j)
		for e in range(m):
			if random.random() < indpb:
				ind_i_e = individual[i][e]
				individual[i][e] = random.choice(mutate_sample)
				mutate_sample.remove(individual[i][e])
				mutate_sample.append(ind_i_e)
	# if sum([len(set(i)) for i in individual]) != m*c : print(individual)		# Check for duplicates after Mut

	return individual,

# ------------------------------------------------ Creating class -----------------------------------------------

# Maximising fitness function => +ve wts
creator.create('FitnessMax', base.Fitness, weights = (1.0, ))
# Individuals inheriting from np.ndarrays
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers, calls custom initialiser
toolbox.register('individual', NdIndividual, creator.Individual, low=n_i, high=n_f, m=M, c=C)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Genetic operators required for the evolution
toolbox.register('evaluate', Fitness_value, Current_month = CM, m = M, c = C, profit_wt = Profit_wt, risk_wt = Risk_wt, \
	root_risk_wt = Root_risk_wt, water_risk_wt = Water_risk_wt, volatile_wt = Volatile_wt, debug = Debug )
toolbox.register('mate', NdcxTwoPointX, m=M, c=C)
toolbox.register('mutate', NdmutUniformInt, m=M, c=C, low=n_i, up=n_f, indpb=INDPB)
toolbox.register('select', tools.selTournament, tournsize=10)

# --------------------------------------------- Evolution operation ----------------------------------------------

def Evolution(m, c, n, CXPB, MUTPB, NGen, print_):

	Max_=[]
	Avg_=[]
	Std_=[]

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

	# ---------------------------------------------- Visualisation --------------------------------------------
	# Stats of each generation

	x_ = np.arange(1,len(Max_)+1)
	plt.bar(x_-0.2, Max_, width = 0.2,align='center', label='Max')
	plt.bar(x_, Avg_, width = 0.2,align='center', label='Avg')
	plt.bar(x_+0.2, Std_, width = 0.2,align='center', label='Std')
	plt.axis([0, NGen+1, 0, 1.4*max(Max_)])
	plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.xlabel('Generation')
	plt.ylabel('Total Profit')
	plt.title('Max - Avg - Std')
	plt.legend()
	plt.show()

	# ---------------------------------------------- Tabular Output --------------------------------------------
	
	# Debuging the best individual and Accessing global variable
	debug_best = True
	Fitness_value(Best, CM, M, C, Profit_wt, Risk_wt, Root_risk_wt, Water_risk_wt, Volatile_wt, debug_best)

	t = PrettyTable(['Cycles', 'Crop', 'Planting Month', 'Harvest Month', 'Root Sys', 'Water Req', 'Profit'])

	for i in range(c):
		for e in range(m):
			t.add_row([i+1, Crop_name[AllinOne[i][0][e]*12-1], AllinOne[i][1][e], AllinOne[i][2][e], AllinOne[i][5][e], \
				AllinOne[i][6][e], AllinOne[i][4][e]])
		t.add_row(['-'*13]*7)
	t.add_row(['Total : ', '-', '-', '-', '-', '-', sum(list(map(lambda x : sum(AllinOne[x][4]), list(range(0,c)))))])

	return Best, Best.fitness.values[0], t

# ========================================= Running Genetic Algorithm ==========================================

Best_ind, Best_fitness, T = Evolution(M, C, N, CXPB, MUTPB, NGen, Print_)
print(Best_ind)
print(Best_fitness)
print(T)

print('Time to ex : %ssec' %(time.time()-start_time))

#------------------------------------------------ Visualisation ------------------------------------------------

hex_list = []
[hex_list.append(cc) for cc in colors.cnames]

for i in range(C):
	for e in range(M):
		cc_id = AllinOne[i][0][e]
		plt.scatter(AllinOne[i][7][e]-AllinOne[i][3][e], e+1, marker='>', color=hex_list[cc_id])
		plt.scatter(AllinOne[i][7][e], e+1, marker='o', color=hex_list[cc_id])
		plt.plot([AllinOne[i][7][e]-AllinOne[i][3][e], AllinOne[i][7][e]], [e+1, e+1], label=Crop_name[AllinOne[i][0][e]*12-1], \
			color=hex_list[cc_id])
		plt.annotate(Crop_name[AllinOne[i][0][e]*12-1], xy=(AllinOne[i][7][e]-AllinOne[i][3][e], e+1), \
			xytext=(AllinOne[i][7][e]-AllinOne[i][3][e], e+1+0.1), size = 8, ha='left', va='bottom', bbox=dict(boxstyle='round', \
				edgecolor='none', fc='lightsteelblue', alpha=0.5))
		plt.annotate('Cycle-'+str(i+1)+str(', ')+str(AllinOne[i][5][e]), xy=(AllinOne[i][7][e]-AllinOne[i][3][e], e+1), \
			xytext=(AllinOne[i][7][e]-AllinOne[i][3][e], e+1-0.2), size = 6, ha='left', va='center', bbox=dict(boxstyle='round', \
				edgecolor='none', fc='lightsteelblue', alpha=0.5), style='italic')

plt.yticks(range(1, M+1), [str(x+1)+'st crop' for x in range(M)])
plt.xticks(range(1, 25), months_*C)
plt.axis([0, 25, -1, M+2])
plt.ylabel('Crops')
plt.xlabel('Months')
plt.title('Crop Cycles')
plt.show()