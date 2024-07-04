"""
Vikrant Tripathi (2103141)
"""

# Import important libraries
import random 
import numpy as np
from collections import defaultdict

# QuickSort function to sort a list
def quickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quickSort(less_than_pivot) + [pivot] + quickSort(greater_than_pivot)
    
# Topological Sort function for variables based on conditional probability tables (cpts)
def topologicalSort (cpts, vars):
    topoStack = []
    visitedArray = []
    
    # Initialize the visited array
    i = 0 
    for _ in vars:
        visitedArray.append(False)
    
    # Traverse the variables and build a topological order
    for cell in range(len(visitedArray)):
        cell_stack = []
        index_stack = []

        if(not visitedArray[cell]):
            cell_stack.append(cell)
            visitedArray[cell] =True
            index_stack.append(cell)

            while(len(cell_stack)!=0):
                stack_top = cell_stack[-1]
                cell_stack.pop()

                for n in cpts:
                    if vars[stack_top] in n[1]:
                        if (not visitedArray[vars.index(n[0][0])]):
                            cell_stack.append(vars.index(n[0][0]))
                            index_stack.append(vars.index(n[0][0]))
                            visitedArray[vars.index(n[0][0])]=True
        index_stack.reverse()
        topoStack+=index_stack

    topoStack.reverse()
    result = []

    for i in topoStack:
        result.append(vars[i])

    return result

# Check if conditions in a query are satisfied by a sample
def checkQueryConditions(conditions, generateSamples):
    for key, value in conditions:
        key = key.strip()
        if generateSamples[key] != value:
            return False
    return True

# Extract variable values and remaining lines from the input
def valExtraction(lines, varCount):
    vals = {}
    i = 0

    for i in range(varCount):
        val = lines[i].strip().split(', ')
        vals[val[0]] = val[1:]
    remaining_lines = lines[varCount:]
    
    return vals, remaining_lines

# Extract the number of variables and variable names
def varExtraction(line):
    extractedVars = line.strip().split(', ')
    varCount = int(extractedVars[0])
    extractedVars = extractedVars[1:]
    return varCount, extractedVars

# Extract conditional probability tables (CPTs) and queries from the input lines
def cptsExtraction(lines):
    quers = []
    cpts = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Query:'):
            quers_parts = line.split('Query: P(')[1][:-1].split('| ')
            cv = []
            ucv = []

            if len(quers_parts) > 1:
                cv = [tuple(var.split('=')) for var in quers_parts[1].split(', ')]

            ucv = [tuple(var.split('=')) for var in quers_parts[0].split(', ')]
            quers.append((cv, ucv))
            i += 1
        else:
            vars_parts = line.split('|')
            vars_parts = [var.strip() for var in vars_parts]
            cv = tuple(quickSort(vars_parts[0].split(', ')))

            if len(vars_parts) > 1:
                ucv = tuple(quickSort(vars_parts[1].split(',')))
            else:
                ucv = ()

            i += 1
            cpt = {}
            while i < len(lines) and '|' not in lines[i] and not lines[i].strip().startswith('Query'):
                line_parts = lines[i].strip().split(', ')
                cpt[tuple(quickSort(line_parts[:-1]))] = float(line_parts[-1])
                i += 1

            cpts[(cv, ucv)] = cpt
    
    del cpts[(('',), ())]

    return cpts, quers
    
# Read input from a file and parse variables, values, CPTs, and queries
def takeFile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    varCount, vars = varExtraction(lines[0])
    vals, lines = valExtraction(lines[1:], varCount)
    cpts, quers = cptsExtraction(lines)

    vars = topologicalSort(cpts, vars)
    return vars, vals, cpts, quers

# Generate a sample value for a variable using conditional probabilities
def generateSamples(values, var, evidence, table):
    sampled_value = None
    cumulative_prob = 0

    if var not in values:
        raise ValueError("Variable not found in values dictionary")

    if not evidence:
        evidence = []

    for val in values[var]:
        conditional_values = evidence + [val]
        key = tuple(quickSort(conditional_values))

        if key not in table:
            raise ValueError("Conditional values not found in table")

        conditional_probability = table[key]

        cumulative_prob += conditional_probability

        if cumulative_prob >= random.random():
            sampled_value = val
            break

    return sampled_value

# Resample a variable using Gibbs sampling
def gibbsResample(cpts,var,values,newSample):
    table={}
    
    # Iterate over possible values of the variable 'var'
    for val in values[var]:
        p1=1

        # Iterate over the conditional probability tables (CPTs)
        for key in cpts:
            cpt=cpts[key]
            new_arr = tuple()

            # Build a tuple 'new_arr' containing values of parent variables (excluding 'var')
            for i in key[0]:
                if i!= var:
                    if i!= '':
                        new_arr+=(newSample[i],)

            # Include values of other parent variables (excluding 'var')
            for i in key[1]:
                if i!= var:
                    if i!= '':
                        new_arr+=(newSample[i],)
        
            new_arr += (val,)

            # Determine if 'var' is in the set of parent variables
            new_arr=(*quickSort(list(new_arr)),)
            if var in list(key[0]):
                newprob = cpt[(new_arr)]
                p1=p1*newprob

            elif var in key[1]:
                newprob = cpt[(new_arr)]
                p1=p1*newprob
        table[(val,)]=p1

    acc = sum([value for key,value in table.items()])
    table = {key: value / acc for key, value in table.items()}

    return generateSamples(values,var,[],table)

# Perform rejection sampling
def rejection_sampling(cpts, quer, vars, values, samplesCount):
    success_count = 0
    total_count = 0
    temp = list(quer[0])
    temp.reverse()

    # Continue until the desired number of samples is reached
    while(total_count<samplesCount):
        newSample = {}

        # Iterate over all variables in the Bayesian network
        for var in vars:
            evidence_values =[]
            table = None

            # Find the conditional probability table (CPT) for the current variable
            for key, value in cpts.items():
                if var in list(key[0]) :
                    table = value
                    break

            # If there is no CPT for the current variable, skip to the next variable
            if not table:
                continue

            # Collect values of evidence variables for the current variable
            for evidence in list(key[1]):
                if evidence in newSample:
                    evidence_values.append(newSample[evidence])

            newSample[var] = generateSamples(values, var, evidence_values, table)

        # Check if the sampled assignment satisfies the query conditions
        if checkQueryConditions(temp[1], newSample):
            total_count += 1
            if checkQueryConditions(temp[0], newSample):
                success_count += 1
        else:
            continue

    # Calculate the probability based on successful matches
    if total_count == 0:
        probability = None
    else:
        probability = success_count / total_count

    print("Rejection sampling probability: ", probability, "\n")

    return success_count, total_count, probability

# Perform prior sampling
def priorSampling(cpts, quer, vars, values, samplesCount):
    success_count = 0
    total_count = 0
    temp = list(quer[0])
    temp.reverse()
    # print(temp)

    # Continue until the desired number of samples is reached
    for _ in range(samplesCount):
        newSample = dict()

        # Iterate over all variables in the Bayesian network
        for var in vars:
            table = None
            for key, value in cpts.items():
                if var in list(key[0])[0]:
                    table = value
                    break

            if not table:
                continue

            # Collect values of evidence variables for the current variable
            evidence_values = []
            for evidence in list(key[1]):
                if evidence in newSample:
                    evidence_values.append(newSample[evidence])
        
            newSample[var] = generateSamples(values, var, evidence_values, table)

        # Check if the sampled assignment satisfies the query conditions
        if checkQueryConditions(temp[1], newSample):
            total_count += 1
            if checkQueryConditions(temp[0], newSample):
                success_count += 1

    # Calculate the probability based on successful matches
    if total_count == 0:
        probability = 0.0
    else:
        probability = success_count / total_count

    print("Prior sampling probability: ", probability, "\n")

    return success_count, total_count, probability

# Perform likelihood sampling
def likelihoodWeighting(cpts, quer, vars, values, samplesCount):
    acc,acc_total=0,0
    temp = list(quer[0])
    temp.reverse()

    # Separate and organize the query and evidence conditions
    ucv,cv=[],[]
    for i in temp[1]:
        ucv.append(i)
    for i in temp[0]:
        cv.append(i)

    # Repeat the sampling process for the desired number of samples
    for _ in range(samplesCount):
        weight=1
        newSample=dict()

        # Iterate over all variables in the Bayesian network
        for var in vars:
            h=None
            for i in ucv:
                if var in i:

                    h=i[1]

            # If the variable is not part of evidence, sample it based on CPT
            if (var, h) not in ucv:
                for key in cpts:
                    table=cpts[key]
                    if var in list(key[0]):
                        evidence=[]

                        for i in list(key[1]):
                            if i in newSample:
                                evidence.append(newSample[i]) 


                        newSample[var] = generateSamples(values, var, evidence, table)

            # If the variable is part of evidence, assign its value
            else:
                newSample[var] = h

                # Calculate the weight associated with this variable and its assignment
                for key in cpts.keys():
                    if var in list(key[0]):
                        x = [h] + [newSample[i] for i in key[1] if i in newSample]
                        x=quickSort(x)
                        sorted_x = tuple(x)
                        weight *= cpts[key][sorted_x]

        # If the sampled assignment satisfies the query conditions, add its weight to the successful matches
        acc_total+=weight
        if checkQueryConditions(temp[0], newSample):
            acc += weight 

    
    print("For likelihood sampling (acc, acc_total, probability): ", acc, acc_total, acc/acc_total, "\n")

# Perform Gibb's sampling
def gibbsSampling(cpts, quer, vars, values, samplesCount):
    acc=0
    temp = list(quer[0])
    temp.reverse()
    quer1 = []

    # Create separate lists for query and evidence conditions
    for key, var in temp[1]:
        quer1.append((key,var))
    quer2 = []
    
    # Organize variables into those that are part of the query (quer1) and the rest (quer2)
    for var in vars:
        signal1=False
        for j in quer1:
            if var==j[0]:
                signal1= not signal1

        if  not signal1:
            quer2.append(var)
    
    # Repeat the sampling process for the desired number of samples
    for _ in range(samplesCount):
        # Assign values for variables in quer1 based on query conditions
        newSample={}
        for i in quer1:
            newSample[i[0]]=i[1]
        
        # Sample variables in quer2 (remaining variables)
        for var in quer2:
            if var in values:
                newSample[var] = random.choice(values[var])
                
            else:
                raise ValueError(f"Variable '{var}' not found in 'values' dictionary.")
        
        # Perform several iterations of Gibbs sampling (typically 500 iterations)
        iter = 500
        for _ in range(iter):
            randarr = []
            per = [var for var in quer2]

            # Randomly permute the order in which variables are sampled
            for i in range(len(per)):
                a=random.choice(per)
                per.pop(per.index(a))
                randarr += [a]
            
            # Iterate through the variables and perform Gibbs sampling for each
            for var in quer2:
                found = False

                # Find the CPT for the variable and perform Gibbs resampling
                for key in cpts:
                    if var in list(key[0]) :
                        sample = gibbsResample(cpts, var, values, newSample)
                        newSample.update({var:sample}) 
                        found = True
                        break
                if not found:
                    raise ValueError(f"No CPT found for var '{var}' in 'cpts'.")
        
        # Check if the sampled assignment satisfies the query conditions
        signal2 = True
        for key, value in temp[0]:
            key = key.strip()
            if newSample.get(key) != value:
                signal2 = False
        
        # If the sampled assignment satisfies the query conditions, increment the accumulator
        if signal2:
            acc += 1


    print("For Gibb's sampling (acc, probability): ", acc,acc/samplesCount, "\n")

    return acc
            
# Main program to print the results
vars, values, cpts, quers = takeFile('input.txt')
print("Values: ", values, "\n")
print ("CPTs:", cpts, "\n")

priorSampling(cpts, quers, vars, values, 10000)
rejection_sampling(cpts, quers, vars, values, 1000)
likelihoodWeighting(cpts, quers, vars, values, 1000)
topologicalSort(cpts, vars)
gibbsSampling(cpts,quers,vars,values,100)
