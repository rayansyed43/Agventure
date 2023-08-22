#Finding a set of 'm' numbers from range (0,n) whose sum is 100

# ==================================== Importing Libraries ================================================

import random
import numpy as np
from deap import algorithms, base, creator, tools
from itertools import repeat
from collections import Sequence

# ========================================== Variables =====================================================

n=1000
n_pop=100
C 	= 4			# No.of crops cycles
M	= 6 		# No.of crops to decide
n_i = 1
n_f	= 20

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
# 		if len(set(ind_itt1)) != m : print(ind_itt1)
# 		if len(set(ind_itt2)) != m : print(ind_itt2)

# 	return ind1, ind2

def NdcxTwoPoint(ind1, ind2, m, c):
	for i in range(c):
		ind_itt1 = ind1[i]
		ind_itt2 = ind2[i]
		cxpoint1 = random.randint(1, m)
		cxpoint2 = random.randint(1, m - 1)
		print(cxpoint1, cxpoint2)
		if cxpoint2 >= cxpoint1:
			cxpoint2 += 1
		else: # Swap the two cx points
			cxpoint1, cxpoint2 = cxpoint2, cxpoint1
		print(ind_itt1, ind_itt2)
		ind_itt1[cxpoint1:cxpoint2], ind_itt2[cxpoint1:cxpoint2] = ind_itt2[cxpoint1:cxpoint2], ind_itt1[cxpoint1:cxpoint2]
		print(ind_itt1, ind_itt2)

	return ind1, ind2


# def NdmutUniformInt(individual, m, c, low, up, indpb):
# 	# low = repeat(low, m*c)
# 	# up = repeat(up, m*c)
# 	for i in range(c):
# 		# for e, xl, xu in zip(range(m), low, up):
# 		for e in range(m):
# 			verr =random.random()
# 			if verr < indpb:
# 				individual[i][e] = random.randint(low, up)

# 	return individual,

# def NdmutUniformInt(individual, m, c, low, up, indpb):
# 	for i in range(c):
# 		mutate_sample = [*range(low, up+1)]
# 		for j in individual[i] : mutate_sample.remove(j)
# 		for e in range(m):
# 			if random.random() < indpb:
# 				ind_i_e = individual[i][e]
# 				individual[i][e] = random.choice(mutate_sample)
# 				mutate_sample.remove(individual[i][e])
# 				mutate_sample.append(ind_i_e)

# 	return individual,

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
	if sum([len(set(i)) for i in individual]) != m*c : print(individual)
	return individual,

def NdcxTwoPointX(ind1, ind2, m, c):
	for i in range(c):
		ind_itt1 = ind1[i]
		ind_itt2 = ind2[i]
		cxpoint1 = random.randint(1, m)
		cxpoint2 = random.randint(1, m - 1)
		if cxpoint2 >= cxpoint1:
			cxpoint2 += 1
		else: 
			cxpoint1, cxpoint2 = cxpoint2, cxpoint1		# Swap the two cx points
		ind_itt1[cxpoint1:cxpoint2], ind_itt2[cxpoint1:cxpoint2] = ind_itt2[cxpoint1:cxpoint2], ind_itt1[cxpoint1:cxpoint2]
		cx1 = ind_itt1[cxpoint1:cxpoint2]
		cx2 = ind_itt2[cxpoint1:cxpoint2]
		cx1ncx2 = list(set(cx1).intersection(cx2))
		cx1 = list(set(cx1).difference(cx1ncx2))
		cx2 = list(set(cx2).difference(cx1ncx2))
		verids = list(range(0,m))
		del(verids[cxpoint1:cxpoint2])
		idcx1 = 0
		idcx2 = 0		
		for e in verids: 
			if ind_itt1[e] in cx1 : ind_itt1[e] = cx2[idcx1]; idcx1+=1
			if ind_itt2[e] in cx2 : ind_itt2[e] = cx1[idcx2]; idcx2+=1
		if len(set(ind_itt1)) != m : print('===============',ind_itt1)
		if len(set(ind_itt2)) != m : print('===============',ind_itt2)

	return ind1, ind2

#objective function
def sum_error(individual):
	ind_verify = sum([len(set(i)) for i in individual])
	if ind_verify == 6*4 : 
		sum_all = 0
		for i in range(len(individual)):
			sum_ = sum(individual[i])
			sum_all = sum_all + sum_
	else : sum_all = 0
	return sum_all,

def init2d(icls, low, high, shape):

	# indGenerator = np.random.randint(low, high, shape)
	indGenerator = []
	for i in range(4):
		indGenerator.append(random.sample(range(1,21), 6))
	return icls(indGenerator)


# Creating class
creator.create('FitnessMin', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# toolbox.register('attr_bool', random.randint, 1, n)	#generator
# Structure initializers
# toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, m)
toolbox.register("individual", init2d, creator.Individual, low=1, high=20, shape=(4,6))
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# genetic operators required for the evolution
toolbox.register('evaluate', sum_error)
toolbox.register('mate', NdcxTwoPointX, m=M, c=C)
toolbox.register('mutate', NdmutUniformInt, m=M, c=C, low=n_i, up=n_f, indpb=0.4)
toolbox.register('select', tools.selTournament, tournsize=3)

#------------------------------------ Simple formate using DEAP ------------------------------------------


# pop = toolbox.population(n_pop)
# # print(pop[0])





# hof = tools.HallOfFame(1)
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register('avg', np.mean)
# stats.register('std', np.std)
# stats.register('min', np.min)
# stats.register('max', np.max)

# pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)
# # [print(sum(ind)) for ind in pop]

# best_=tools.selBest(pop, k=5)
# print(best_)

#------------------------------------------ Evolution operation ----------------------------------------------
#---------------------------------------------- ( lengthy ) --------------------------------------------------

def main():

	# create an initial population of 300 individuals
	pop = toolbox.population(n=1000)
	# print(pop)
	# CXPB  is the probability with which two individuals are crossed
	# MUTPB is the probability for mutating an individual
	# Number of generations/Number of itterations
	global NGen
	CXPB, MUTPB, NGen = 0.5, 0.2, 30

	print("Start of evolution")
	
	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	
	print("  Evaluated %i individuals" % len(pop))

	# Extracting all the fitnesses of 
	fits = [ind.fitness.values[0] for ind in pop]

	# Begin the evolution
	for g in range(NGen):

		gen = g+1
		print("-- Generation %i --" % gen)
		
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
	
		# print(offspring)

		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			# print('----',child1, child2)
			# cross two individuals with probability CXPB
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				# print(toolbox.mate(child1, child2))
				# fitness values of the children
				# must be recalculated later
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			# print('--',mutant)
			# mutate an individual with probability MUTPB
			rrr = random.random()
			if rrr < MUTPB:
				# print(rrr, MUTPB)
				toolbox.mutate(mutant)
				# print('==',toolbox.mutate(mutant))
				del mutant.fitness.values
	
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		print("  Evaluated %i individuals" % len(invalid_ind))
		
		# The population is entirely replaced by the offspring
		pop[:] = offspring
		
		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]
		
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5
		
		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)
		print("  Std %s" % std, '\n')
	
	print("-- End of successful evolution --")
	Best = tools.selBest(pop, 1)[0]
	# print("Best individual is %s, %s" % (Best, Best.fitness.values))
	print("Best individual is %s " % Best)	
	print("Fitness value is %s " % Best.fitness.values)

if __name__ == "__main__":
	main()

# pop = toolbox.population(n=100)
# # print(pop)

# for i in range(len(pop)):
# 	indi = pop[i]
# 	for e in range(len(indi)):
# 		if len(set(indi[e])) != len(indi[e]) : print(indi, indi[e])
# M = 6
# C = 2
# indi1 = [[ 1, 20,  5,  7, 19, 15], [ 5, 16,  1, 20, 18,  3]]
# indi2 = [[ 7,  1, 12,  3,  5,  2], [ 6,  5,  1,  3, 16,  8]]
# print(NdcxTwoPointX(indi1, indi2, M, C))


# individual = [[15, 5, 8, 20, 1, 16], [16, 9, 5, 11, 17, 2], [19, 20, 6, 7, 5, 2], [7, 11, 13, 15, 2, 17]]
# print(sum_error(individual))