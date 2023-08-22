#----------------------------------------- Importing Libraries -----------------------------------------

import random
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import style
from collections import Counter
from prettytable import PrettyTable
from deap import algorithms, base, tools, creator
style.use('ggplot')

#-------------------------------------------- Reading Data ---------------------------------------------

# Reading CSV file
df = pd.read_csv('Gudur_Rythu_Bazar_2017.csv')
df.drop(['Comments'], axis = 1, inplace=True)	# Dropping 'Comments' column

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
Profit 	= df['Profit']
Profit 	= np.array(Profit)
Month 	= df['Month']
Month 	= np.array(Month)
Type 	= df['Type_Code']
Type 	= np.array(Type)

#--------------------------------------------- Variable info ----------------------------------------------

months_ = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Current_month = datetime.datetime.now().month
Max_all=[]
Avg_all=[]
Std_all=[]
global NGen
Debug = False
print_ = False
loop = False

n 	= 300
m	= 5		# No.of crops to decide
n_i = Type[0] 	# Lower limit of no.of crops
n_f	= Type[-1] 	# Upper limit of no.of crops / Total no.of crops
NGen 	= 10		# Number of generations/Number of itterations			
CXPB	= 0.5		# CXPB  is the probability with which two individuals are crossed
MUTPB 	= 0.2		# MUTPB is the probability for mutating an individual

# Weights to cal weighted avg
profit_wt	= 0.7
risk_wt 	= -0.3
root_risk_wt	= 0.5
water_risk_wt	= 0.5

#----------------------------------------------- Fitness Function ----------------------------------------------
# Objective fun : [1] Maximize Profit
#				  [2] Mininize Risk 
# 			i.e : Max(W1*Profit -W2*Risk)
# Subject to constrains : [1] Harvest time.
#						  [2] Crop cycle in a year based on harvest time.
#						  [3] Based on root system.
#						  [4] Based on water requirement.

def Fitness_value(individual):

	global profit
	global harvest_month
	global planting_month

	profit = []
	harvest_month = []
	planting_month = []
	root_depth = []
	water_req = []

	#---------------------------------------------- Estimating Profit -----------------------------------------

	if len(set(individual))==m:
		for i in range(len(individual)):
			profit_itt = []
			harvest_month_itt = []
			planting_month_itt = []
			Crop = individual[i]
			for e in range(len(Type)):
				if Type[e]==Crop:
					type_id = e
					break
				else:
					pass

			for i in range(12):
				if (i+1)*(Harvest_time[type_id]+1) <= 12: 

					profit_id = type_id + Current_month + Harvest_time[type_id] -1 + i*(Harvest_time[type_id]+1)
					id_verify = Current_month + Harvest_time[type_id] -1 + i*(Harvest_time[type_id]+1)
					if id_verify < 12:
						profit_i = Profit[profit_id]
						profit_itt.append(profit_i)
						planting_month_itt.append(Month[profit_id - Harvest_time[type_id]])
						harvest_month_itt.append(Month[profit_id])
						# break
					else:
						profit_i = Profit[type_id + profit_id%12]
						profit_itt.append(profit_i)
						planting_month_itt.append(Month[type_id + profit_id%12 - Harvest_time[type_id]])
						harvest_month_itt.append(Month[type_id + profit_id%12])
						# break
				else:
					break
			profit.append(sum(profit_itt))
			planting_month.append(planting_month_itt)
			harvest_month.append(harvest_month_itt)
			root_depth.append(Root_depth[type_id])
			water_req.append(Water_req[type_id])

	else:
		profit=[0]

	Profit_percent = sum(profit)/10**4

	#---------------------------------------------- Estimating Risk -------------------------------------------

	list_risk=[]

	# Risk due to competition over nitrogen from the soil
	# lower limit = 0
	# upper limit = 100
	list_abc_1=[]
	counts = Counter(root_depth)
	per_s = counts['Shallow']*100/m
	per_m = counts['Medium']*100/m
	per_d = counts['Deep']*100/m
	
	if per_s and per_m != 0:
		a_1 = abs(per_s - per_m)
		list_abc_1.append(a_1)
	if per_s and per_d != 0:
		b_1 = abs(per_s - per_d)
		list_abc_1.append(b_1)
	if per_m and per_d != 0:
		c_1 = abs(per_m - per_d)
		list_abc_1.append(c_1)
	if len(list_abc_1) != 0:
		avg_abc_1 = sum(list_abc_1)/len(list_abc_1)
	else:
		avg_abc_1 = 100
	list_risk.append(avg_abc_1)

	# Risk due to competition over water requirement
	# lower limit = m*20
	# upper limit = m*50
	list_abc_2=[]
	counts_water = Counter(water_req)
	per_L = counts_water['L']
	per_M = counts_water['M']
	per_H = counts_water['H']
	
	avg_abc_2 = 20*per_L + 30*per_M + 50*per_H
	list_risk.append(avg_abc_2)
	
	risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1])
	Risk_percent = risk

	#-----------------------------------------------------------------------------------------------------------
	
	combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)
	
	if Debug == True:
		print('-- Debugging --')
		print('Profit_val 	: %s \nRisk_val 	: %s \nCombined_val 	: %s \nRisk_root 	: %s \nRisk_water 	: %s' \
			%(Profit_percent, Risk_percent, combined_val, avg_abc_1, avg_abc_2) )
	else:
		pass

	# return sum(profit), risk
	return combined_val, 

# ------------------------------------------------ Creating class -----------------------------------------------

# creator.create('FitnessMax', base.Fitness, weights = (1.0, -1.0))
creator.create('FitnessMax', base.Fitness, weights = (1.0, ))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_value', random.randint, n_i, n_f)	# generator
# Structure initializers
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_value, m)	
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
# genetic operators required for the evolution
toolbox.register('evaluate', Fitness_value)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=n_i, up=n_f, indpb=0.2)
toolbox.register('select', tools.selTournament, tournsize=3)

#------------------------------------------ Evolution operation ----------------------------------------------

def Evolution(n, CXPB, MUTPB, NGen):

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

		Max_all.append(Max_)
		Avg_all.append(Avg_)
		Std_all.append(Std_)
		
		if print_ == True: print("  Min %s" % min(fits))
		if print_ == True: print("  Max %s" % max(fits))
		if print_ == True: print("  Avg %s" % mean)
		if print_ == True: print("  Std %s" % std, '\n')
	
	if print_ == True: print("-- End of successful evolution --")

	Best = tools.selBest(pop, 1)[0]	

	#-------------------------------------------- Displaying output -----------------------------------------

	if loop == False: print("Best individual is %s, %s" % (Best, Best.fitness.values))

	# To access global var 'profit', To display profit due to each crop in 'Best' individual
	Fitness_value(Best)

	# Printing Data in table format
	Total_profit = 0
	t = PrettyTable(['Crop','Planting Months', 'Harvest Months', 'Cycles', 'Root Sys', \
		'Water Req', 'Culti Cost', 'Profit'])
	for i in range(len(Best)):
		val = Best[i]
		t.add_row([Crop_name[val*12-1], ', '.join(planting_month[i]), ', '.join(harvest_month[i]), \
		len(harvest_month[i]), Root_depth[val*12-1], Water_req[val*12-1], \
		len(harvest_month[i])*Culti_cost[val*12-1], profit[i]])
		Total_profit = Total_profit + profit[i]
	if loop == False: print(t)
	if loop == False: print("Total Profit : %s " % Total_profit)

	return Best, t, Total_profit

# ======================================== Running Genetic Algorithm =========================================
# ---------------------------------- Deciding crops for the present month ------------------------------------

print('Start planting now, i.e. %s' %months_[Current_month-1])
Best_ind, _, _ = Evolution(n, CXPB, MUTPB, NGen)

# ---------------------------------- Looping to find which month to start ------------------------------------

profit_month = []
t_month = []
best_month = []
print('\nIf planting starts on :')
for i in range(12):
	print_ = False
	loop = True
	Current_month = i+1
	Best_ind, t, T_profit  = Evolution(n, CXPB, MUTPB, NGen)
	print(months_[i], ':', T_profit)
	profit_month.append(T_profit)
	best_month.append(Best_ind)
	t_month.append(t)

id_month = np.argmax(profit_month)
print('Starting from %s is prefered.' %months_[id_month])
print('Total Profit : ', profit_month[id_month])
print(best_month[id_month])
print(t_month[id_month])

#---------------------------------------------- Visualisation ------------------------------------------------

# Which index value to point to, all stats values are in Max_all, Avg_all, Std_all lists
# To visualise for each month use num value of month, i.e : 1 to 12
index_visual=0

x_ = np.arange(1,len(Max_all[index_visual])+1)

plt.bar(x_-0.2, Max_all[index_visual], width = 0.2,align='center', label='Max')
plt.bar(x_, Avg_all[index_visual], width = 0.2,align='center', label='Avg')
plt.bar(x_+0.2, Std_all[index_visual], width = 0.2,align='center', label='Std')
plt.axis([0, NGen+1, 0, 1.4*max(Max_all[index_visual])])
plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel('Generation')
plt.ylabel('Total Profit')
plt.title('Max - Avg - Std')
plt.legend()
# plt.show()

#------------------------------------------------ Debugging ---------------------------------------------------

# Debug = True
# Fitness_value(best_month[id_month])
