#print ('hello world')
#print ("let's try")


# sum of two numbetrs
# num1 = (float(input('Enter the first number: ')))
# num2 = (float(input('Enter the second number: ')))

#sum = num1 + num2
#print('Here is the summation: ',sum)

#num = int (input('Enter your number: '))

#if num%2==0:
#    print ('The %d is even', num)
#else:
#    print ('The %d is odd', odd)

# Take input from the user
#num = int(input("Enter a number: "))

# Print the multiplication table

#for i in range(1, 100):
#    print(num, "x", i, "=", num * i)


#num = int(input('Enter the number: '))

#print ('The multiplication table is stated below for this number:',num)
#for i in range (1, 11):
# print (num, 'X',i,'=',num*i)


# Take input from the user
num = int(input("Enter a number: "))

# Check if the number is prime
#is_prime = True
#if num > 1:
#    for i in range(2, int(num**0.5) + 1):
#        if num % i == 0:
#            is_prime = False
#            break
#else:
#    is_prime = False
#
# Print the result
#if is_prime:
#    print(num, "is a prime number.")
#else:
#    print(num, "is not a prime number.")

try:
    # Take input from the user
    num = int(input("Enter a number: "))

    # Check if the number is prime
    is_prime = True
    if num > 1:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
    else:
        is_prime = False  # Numbers <= 1 are not prime

    # Print the result
    if is_prime:
        print(num, "is a prime number.")
    else:
        print(num, "is not a prime number.")

except ValueError:
    print("Please enter a valid number.")