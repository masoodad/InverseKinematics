import numpy as np
import modern_robotics as mr
import random as rand


T=np.array([[0,1,0,-0.5],
            [0,0,-1,0.1],
            [-1,0,0,0.1],
            [0,0,0,1]])
eomg = 0.001
ev = 0.0001

L1=0.425
L2=0.392
W1=0.109
W2=0.082
H1=0.089
H2=0.095

Blist = np.array([[0, 1, 0, W1+W2, 0,  L1+L2],
                  [0, 0,  1, H2, -L1-L2,   0],
                  [0, 0,  1, H2, -L2, 0],
                  [0, 0,  1, H2, 0, 0],
                  [0, -1,  0, -W2, 0, 0],
                  [0, 0,  1, 0, 0, 0]]).T

M = np.array([[-1, 0,  0, L1+L2],
              [ 0, 0,  1, W1+W2],
              [ 0, 1,  0, H1-H2],
              [ 0, 0,  0, 1]])
			  

def randomly(seq):
    shuffled = list(seq)
    rand.shuffle(shuffled)
    
    return iter(shuffled)

def cal_fitness(population):
	
    population_all = np.array(population).copy()
        
    for i in range(population_all.shape[0]):
        se3 = mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, \
                                                                population_all[i])), T))
        Vb = mr.se3ToVec(se3)
        ew = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        el = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        
        if i==0:
            fitness_norm = np.linalg.norm(Vb)
            err = ew > eomg \
                or el > ev
        else:
            fitness_norm = np.append(fitness_norm,np.linalg.norm(Vb))
            err = np.append(err, (ew > eomg \
                or el > ev))
     
    fitness_array = fitness_norm.reshape(-1,10)
    
    return (fitness_array , err)

def select_mating_pool(pop , fitness_v, num_parents):
    #Selecting the best and making it parent for the next generation
    parents = np.empty((num_parents,pop.shape[1]))
    fitness=np.array(fitness_v).T
    for parent_num in range(num_parents):
        max_fitness_idx = fitness.argmin()
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = 9999
    return parents


def crossover(parents, offspring_size):
    offspring=np.empty(offspring_size)
    
    # The point at which crossover takes place between two parents. Usually it is at the center.
    print(offspring_size)
    for random_pop in randomly(range(offspring_size[0])):
        print(random_pop)
        # Index of the first parent to mate.
        parent1_idx = random_pop%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (rand.randint(0,parents.shape[0]+1))%parents.shape[0]

        for m in range (offspring_size[1]):
            #print(offspring[random_pop][m])
            st = str(offspring[random_pop][m])
            #print(st)
            par_string = str(parents[parent1_idx][m])
            par2_string = str(parents[parent2_idx][m])
            #print("initial value",par_string[0],par2_string[0])
            if(par_string[0]!='-' and par2_string[0]!='-'):
                st = par_string[:3]
                #print(st)
                st = st + par2_string[3:11]
                #print("-",st,"-")
            elif(par_string[0]!='-' and par2_string[0]!='+'):
                st = par_string[:3]
                #print("-",st,"+")
                st = st + par2_string[4:11]
                #print(st)
            elif(par_string[0]!='+' and par2_string[0]!='+'):
                st = par_string[:4]
                #print("+",st,"+")
                st = st + par2_string[4:11]
                #print(st)
            elif(par_string[0]!='+' and par2_string[0]!='-'):
                st = par_string[:4]
                #print("+",st,"-")
                st = st + par2_string[3:11]
                #print(st)
                
            offspring[random_pop][m] = float(st)
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for k in range(offspring_crossover.shape[0]):
        for random_col in randomly(range(offspring_crossover.shape[1])):
            st = str(offspring_crossover[k][random_col])
            st1 = list(st)
            rand_val_col = rand.randint(0,6)
            st1[rand_val_col] = rand.randint(0,11)
            offspring_crossover[k][random_col] = float(st)
    return offspring_crossover
