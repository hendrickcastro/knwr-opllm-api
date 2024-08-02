def int_to_string(n):
    return str(n)

# Pide un entero al usuario
user_input = input("Enter an integer: ")

try:
    # Convierte el input en un entero
    n = int(user_input)
    # Convierte el entero en una cadena y muestra el resultado
    print(f"The integer {n} converted to a string is: {int_to_string(n)}")
except ValueError:
    print("Invalid input. Please enter an integer.")
