def fractional_knapsack(value,weight,capacity):
    index = list(range(len(value)))

    ratio = [v/w for v,w in zip(value,weight)]

    index.sort(key=lambda i: ratio[i],reverse=True)

    max_value = 0
    fractions = [0]*len(value)
    

    for i in index:
        if weight[i] <= capacity:
            fractions[i]=1
            max_value += value[i]
            capacity -= weight[i]
        else:
            fractions[i] = capacity/weight[i]
            max_value += value[i]*(capacity/weight[i])
            break
    return max_value,fractions

n = int(input("Enter the no. of items: "))

value=input("Enter the value of {n} item(s) in order:").split()
value=[int(v) for v in value]

weight=input("Enter the weight of {n} item(s) in order:").split()
weight=[int(w) for w in weight]

capacity = int(input("Enter the max weight: "))

max_value, fractions = fractional_knapsack(value, weight, capacity) 

print('The maximum value of items that can be carried:', max_value) 
print('The fractions in which the items should be taken:', fractions) 
            



