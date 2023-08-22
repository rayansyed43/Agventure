#========================================== Importing Libraries ===========================================

import random
import datetime
import numpy as np 
import pandas as pd 
from prettytable import PrettyTable
from deap import algorithms, base, tools, creator

#Reading CSV file
df = pd.read_csv('Gudur_Rythu_Bazar_2017.csv')
df.drop(['Comments'], axis = 1, inplace=True)	#Dropping 'Comments' column

#np arrays of colomns
Harvest_time = df['Maturity_mo']
Harvest_time = np.array(Harvest_time)
Month = df['Month']
Month = np.array(Month)
Crop_name = df['Type']
Crop_name = np.array(Crop_name)
Profit = df['Profit']
Profit = np.array(Profit)
Type = df['Type_Code']
Type = np.array(Type)
Current_month = datetime.datetime.now().month
Current_month_str = datetime.datetime.today().strftime('%B')
n_i=1
n_f=20
m=5
count=0

def Harvest_month(code_val):

	crop_id_verify = (code_val-1)*12 + Harvest_time[(code_val-1)*12] + (Current_month -1)
	if crop_id_verify < 12:
		crop_id = crop_id_verify
	else :
		crop_id = (code_val-1)*12 + crop_id_verify%12
	harvest_month = Month[crop_id]
	return harvest_month

def Fitness_value(individual):

	profit = []
	if len(set(individual))==m:
		for i in range(len(individual)):
			Crop = individual[i]
			for e in range(len(Type)):
				if Type[e]==Crop:
					type_id = e
					break
				else:
					pass
			profit_id = type_id + Current_month + Harvest_time[type_id] -1
			id_verify = Current_month + Harvest_time[type_id] -1
			if profit_id < 12:
				profit_i = Profit[profit_id]
				profit.append(profit_i)
			else:
				profit_i = Profit[type_id + profit_id%12]
				profit.append(profit_i)
	else:
		profit=[0]
	return sum(profit),

#Creating class
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_value', random.randint, n_i, n_f)	#generator
#Structure initializers
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_value, m)	
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
#genetic operators required for the evolution
toolbox.register('evaluate', Fitness_value)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=n_i, up=n_f, indpb=0.2)
toolbox.register('select', tools.selTournament, tournsize=3)

#generating population 
pop = toolbox.population(n=100)

#------------------------------------ Simple formate using DEAP ------------------------------------------

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('min', np.min)
stats.register('max', np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof)

#---------------------------------------------------------------------------------------------------------

Best=tools.selBest(pop, k=1)
print(Best)
t = PrettyTable(['Crop','Planting Month', 'Harvest Month'])
for i in range(len(Best[0])):
	val = Best[0][i]
	t.add_row([Crop_name[val*12-1], Current_month_str, Harvest_month(val)])
print(t)