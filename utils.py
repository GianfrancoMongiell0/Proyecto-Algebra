# utils.py (funciones necesarias para main)
import numpy as np

def load_system():
    source = input("¿Cargar desde archivo? (s/n): ").lower()
    if source == 's':
        filename = input("Nombre del archivo (ej. sistema.txt): ")
        return load_from_file(filename)
    else:
        return manual_input()

def load_from_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        n = int(lines[0])
        A = np.array([list(map(float, line.split())) for line in lines[1:n+1]])
        b = np.array(list(map(float, lines[n+1].split())))
        return A, b
    except:
        raise Exception("Formato de archivo incorrecto")

def manual_input():
    n = int(input("Tamaño del sistema (n): "))
    print("\nIngrese la matriz A (fila por fila):")
    A = np.array([list(map(float, input(f"Fila {i+1}: ").split())) for i in range(n)])
    print("\nIngrese el vector b:")
    b = np.array(list(map(float, input("Vector b: ").split())))
    return A, b

def print_solution(solution):
    if solution is None:
        print("El sistema no tiene solución única")
    else:
        for i, x in enumerate(solution):
            print(f"x{i+1} = {x:.6f}")