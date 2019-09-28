import numpy as np
import Gen_Algo1 as GA


initial_value=np.random.uniform(low=-3.14, high=3.14, size=60)
    
new_population=initial_value.reshape(-1,6)
num_generation = 10
num_parents_mating = 5

for generation in range(num_generation):
    print("Generation : ",generation)
    #Measuring the fitness of each chromosome in the population
    print(new_population)
    [fitness , err] = GA.cal_fitness(new_population)

    #Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)
    #print(parents)
    
    #Generating next generation using crossover.
    offspring_size=(new_population.shape[0]-parents.shape[0],new_population.shape[1])
    offspring_crossover = GA.crossover(parents, offspring_size)
    #print(offspring_crossover)

    #Adding mutation to do final magic.

    offspring_mutation = GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # The best result in the current iteration.
    print(new_population)
    best_match_idx = fitness.argmin()
    fitness = np.array(fitness).T
    print("Best solution fitness : ", fitness[best_match_idx])

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
[fitness_V, err] = GA.cal_fitness(new_population)
fitness = np.array(fitness_V).T
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = fitness.argmin()
#print(new_population)
print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


    

			  
