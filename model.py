import numpy as np
import matplotlib.pyplot as plt

# Параметры
m = 1.0  # масса (кг)
k = 10.0  # коэффициент жесткости (Н/м)
b = 0.5  # коэффициент сопротивления (Н·с/м)
x0 = 1.0  # начальное положение (м)
v0 = 0.0  # начальная скорость (м/с)
t_max = 10.0  # максимальное время (с)
dt = 0.01  # шаг по времени (с)
n_steps = int(t_max / dt)

time = np.linspace(0, t_max, n_steps)
x = np.zeros(n_steps)
v = np.zeros(n_steps)
E_kinetic = np.zeros(n_steps)
E_potential = np.zeros(n_steps)
E_total = np.zeros(n_steps)

x[0] = x0
v[0] = v0

for i in range(1, n_steps):
    a = -(k / m) * x[i - 1] - (b / m) * v[i - 1]

    v[i] = v[i - 1] + a * dt
    x[i] = x[i - 1] + v[i - 1] * dt

    # Кинетическая энергия
    E_kinetic[i] = 0.5 * m * v[i] ** 2

    # Потенциальная энергия
    E_potential[i] = 0.5 * k * x[i] ** 2

    # Полная энергия
    E_total[i] = E_kinetic[i] + E_potential[i]

plt.figure(figsize=(10, 6))

plt.plot(time, E_kinetic, label='Кинетическая энергия', color='r')
plt.plot(time, E_potential, label='Потенциальная энергия', color='b')
plt.plot(time, E_total, label='Полная энергия', color='g', linestyle='--')

plt.title('Энергии колебательной системы')
plt.xlabel('Время (с)')
plt.ylabel('Энергия (Дж)')
plt.legend()
plt.grid(True)
plt.show()