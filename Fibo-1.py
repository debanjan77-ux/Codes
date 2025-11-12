#Recursive

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

n = int(input("Enter the no. of terms: "))
print("Fibonacci seq(recursive)")

for i in range(n):
        print(fibonacci(i))



#Non-Recursive

nterm = int(input("Enter the no. of terms: "))

n1, n2 = 0, 1
count = 0

if nterm <= 0:
    print("Enter the positive no.")

elif nterm == 1:
    print("Fibonacci seq upto", nterm,":")
    print(n1)

else:
    print("Fibonacci Seq(Non Recursive")
    while count < nterm:
        print(n1)
        nth = n1 + n2
        n1 = n2
        n2 = nth
        count += 1
    
