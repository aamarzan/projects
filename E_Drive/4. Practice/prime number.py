while True:
    try:
        # Take input from the user
        num = int(input("Enter a number: "))
        break  # Exit the loop if input is valid
    except ValueError:
        print("Please enter a valid number.")

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