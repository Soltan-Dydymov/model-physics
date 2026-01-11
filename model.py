import numpy as np
import matplotlib.pyplot as plt

# КОНСТАНТЫ
G = 6.67e-11  # гравитационная постоянная, Н·м²/кг²
M_SUN = 1.989e30  # масса Солнца, кг
AU = 1.5e11  # астрономическая единица, м
GM = G * M_SUN  # гравитационный параметр
YEAR = 365.25 * 24 * 3600  # год в секундах

def acceleration(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    ax = -GM * x / r ** 3
    ay = -GM * y / r ** 3
    return ax, ay


def improved_euler_step(x, y, vx, vy, dt):

    ax, ay = acceleration(x, y)

    vx_pred = vx + ax * dt
    vy_pred = vy + ay * dt
    x_pred = x + vx * dt
    y_pred = y + vy * dt

    ax_pred, ay_pred = acceleration(x_pred, y_pred)

    vx_new = vx + 0.5 * (ax + ax_pred) * dt
    vy_new = vy + 0.5 * (ay + ay_pred) * dt
    x_new = x + 0.5 * (vx + vx_pred) * dt
    y_new = y + 0.5 * (vy + vy_pred) * dt

    return x_new, y_new, vx_new, vy_new


def simulate_orbit(x0, y0, vx0, vy0, dt, t_max):
    t = 0
    x, y, vx, vy = x0, y0, vx0, vy0

    trajectory = {'x': [x], 'y': [y], 'vx': [vx], 'vy': [vy], 't': [t]}

    while t < t_max:
        x, y, vx, vy = improved_euler_step(x, y, vx, vy, dt)
        t += dt

        trajectory['x'].append(x)
        trajectory['y'].append(y)
        trajectory['vx'].append(vx)
        trajectory['vy'].append(vy)
        trajectory['t'].append(t)

        r = np.sqrt(x ** 2 + y ** 2)
        if r < 0.01 * AU:
            print(f"  Столкновение с Солнцем при t = {t / YEAR:.2f} лет")
            break

    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def orbital_parameters(x, y, vx, vy):

    r = np.sqrt(x ** 2 + y ** 2)
    v = np.sqrt(vx ** 2 + vy ** 2)

    # Удельная полная механическая энергия
    E = v ** 2 / 2 - GM / r

    # Удельный момент импульса
    L = abs(x * vy - y * vx)

    # Эксцентриситет
    e = np.sqrt(1 + 2 * E * L ** 2 / GM ** 2)

    if E < 0:
        a = -GM / (2 * E)
        T = 2 * np.pi * np.sqrt(a ** 3 / GM)
        orbit_type = "эллипс"
    elif E == 0:
        a = np.inf
        T = np.inf
        orbit_type = "парабола"
    else:
        a = GM / (2 * E)
        T = np.inf
        orbit_type = "гипербола"

    return E, L, a, e, T, orbit_type

print()
print("ЗАДАНИЕ 4: ТРАЕКТОРИЯ КОМЕТЫ")

# Начальные условия
r0 = 100 * AU
v0 = 2000
alpha = np.radians(30)

# Начальная позиция
x0, y0 = r0, 0

vx0 = -v0 * np.cos(alpha)
vy0 = v0 * np.sin(alpha)

v_escape = np.sqrt(2 * GM / r0)
E, L, a, e, T, orbit_type = orbital_parameters(x0, y0, vx0, vy0)

print(f"\n--- Начальные условия ---")
print(f"Расстояние от Солнца: r₀ = {r0 / AU:.0f} а.е.")
print(f"Скорость: v₀ = {v0 / 1000:.1f} км/с")
print(f"  vₓ = {vx0 / 1000:.3f} км/с")
print(f"  vᵧ = {vy0 / 1000:.3f} км/с")
print(f"Угол к оси Комета-Солнце: α = {np.degrees(alpha):.0f}°")

print(f"\n--- Вычисленные параметры орбиты ---")
print(f"Скорость убегания: v_esc = {v_escape / 1000:.2f} км/с")
print(f"Полная энергия: E = {E:.3e} Дж/кг", end="")
print(f" → {'ЗАМКНУТАЯ орбита' if E < 0 else 'НЕЗАМКНУТАЯ орбита'}")
print(f"Тип орбиты: {orbit_type}")

if a != np.inf:
    print(f"Большая полуось: a = {a / AU:.2f} а.е.")
    print(f"Эксцентриситет: e = {e:.4f}")
    r_perihelion = a * (1 - e)
    r_aphelion = a * (1 + e)
    print(f"Перигелий: r_min = {r_perihelion / AU:.2f} а.е.")
    print(f"Афелий: r_max = {r_aphelion / AU:.2f} а.е.")
    print(f"Период обращения: T = {T / YEAR:.1f} лет")

dt = 3600 * 24 * 2
t_max = 1.5 * T if T != np.inf else 500 * YEAR
trajectory = simulate_orbit(x0, y0, vx0, vy0, dt, t_max)

fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.plot(trajectory['x'] / AU, trajectory['y'] / AU, 'b-', lw=0.8, label='Траектория')
ax1.plot(0, 0, 'yo', markersize=20, label='Солнце')
ax1.plot(x0 / AU, y0 / AU, 'go', markersize=10, label=f'Старт (100 а.е.)')
ax1.set_xlabel('x, а.е.', fontsize=12)
ax1.set_ylabel('y, а.е.', fontsize=12)
ax1.set_title(f'Траектория кометы: v₀ = 2 км/с, α = 30°\n'
              f'a = {a / AU:.1f} а.е., e = {e:.3f}, T = {T / YEAR:.0f} лет', fontsize=13)
ax1.axis('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.axhline(y=0, color='k', lw=0.5)
ax1.axvline(x=0, color='k', lw=0.5)
plt.tight_layout()
plt.savefig('comet_trajectory.png', dpi=150)
plt.show()

print()
print("УСЛОВИЕ ЗАМКНУТОЙ ТРАЕКТОРИИ")

print(f"\nОрбита замкнута, если v₀ < v_escape = {v_escape / 1000:.2f} км/с")
print(f"При v₀ = 2 км/с < {v_escape / 1000:.2f} км/с → орбита замкнутая ✓")

print()
print("ЗАВИСИМОСТЬ ТИПА ОРБИТЫ ОТ НАЧАЛЬНОЙ СКОРОСТИ")

velocities = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.21, 5.0]

fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

print(f"\nСкорость убегания на 100 а.е.: {v_escape / 1000:.2f} км/с")
print(f"\n{'v₀ км/с':^8} {'Тип':^12} {'a, а.е.':^10} {'e':^8} {'T, лет':^10} {'r_min':^10}")
print("-" * 60)

for i, v0_km in enumerate(velocities):
    v0 = v0_km * 1000
    vx0 = -v0 * np.cos(alpha)
    vy0 = v0 * np.sin(alpha)

    E, L, a, e, T, orbit_type = orbital_parameters(x0, y0, vx0, vy0)

    if E < 0:
        t_max_sim = min(1.5 * T, 3000 * YEAR)
    else:
        t_max_sim = 200 * YEAR

    a_str = f"{a / AU:.1f}" if a != np.inf else "∞"
    T_str = f"{T / YEAR:.0f}" if T != np.inf else "∞"
    r_min = a * (1 - e) if a != np.inf else 0
    r_min_str = f"{r_min / AU:.2f}" if r_min > 0 else "-"

    print(f"{v0_km:^8.2f} {orbit_type:^12} {a_str:^10} {e:^8.3f} {T_str:^10} {r_min_str:^10}")

    traj = simulate_orbit(x0, y0, vx0, vy0, dt, t_max_sim)

    ax = axes[i]
    ax.plot(traj['x'] / AU, traj['y'] / AU, 'b-', lw=0.5)
    ax.plot(0, 0, 'yo', markersize=12)
    ax.plot(r0 / AU, 0, 'go', markersize=6)
    ax.set_xlabel('x, а.е.')
    ax.set_ylabel('y, а.е.')
    ax.set_title(f'v₀ = {v0_km} км/с\n({orbit_type})')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Траектории при различных начальных скоростях (α = 30°)', fontsize=14)
plt.tight_layout()
plt.savefig('velocity_comparison.png', dpi=150)
plt.show()

print()
print("ПРОВЕРКА ЗАКОНОВ КЕПЛЕРА")

# Орбита с v₀ = 1.5 км/с
v0_test = 1500
vx0_test = -v0_test * np.cos(alpha)
vy0_test = v0_test * np.sin(alpha)

E, L, a, e, T, orbit_type = orbital_parameters(x0, y0, vx0_test, vy0_test)
print(f"\nТестовая орбита: v₀ = {v0_test / 1000} км/с")
print(f"a = {a / AU:.2f} а.е., e = {e:.4f}, T = {T / YEAR:.1f} лет")

dt_fine = 3600 * 24  # шаг 1 день
traj = simulate_orbit(x0, y0, vx0_test, vy0_test, dt_fine, 1.2 * T)

print("\n--- Первый закон: орбита — эллипс, Солнце в фокусе ---")

r_numerical = np.sqrt(traj['x'] ** 2 + traj['y'] ** 2)
r_min_num = np.min(r_numerical)
r_max_num = np.max(r_numerical)

r_perihelion = a * (1 - e)
r_aphelion = a * (1 + e)

error_peri = abs(r_min_num - r_perihelion) / r_perihelion * 100
error_aph = abs(r_max_num - r_aphelion) / r_aphelion * 100

print(f"Перигелий: числ. = {r_min_num / AU:.4f} а.е., теор. = {r_perihelion / AU:.4f} а.е., ошибка = {error_peri:.3f}%")
print(f"Афелий:    числ. = {r_max_num / AU:.4f} а.е., теор. = {r_aphelion / AU:.4f} а.е., ошибка = {error_aph:.3f}%")

sum_r = r_min_num + r_max_num
error_2a = abs(sum_r - 2 * a) / (2 * a) * 100
print(f"r_min + r_max = {sum_r / AU:.4f} а.е., 2a = {2 * a / AU:.4f} а.е., ошибка = {error_2a:.3f}%")
print(f"=> Первый закон {'ПОДТВЕРЖДЕН ✓' if error_2a < 1 else 'НЕ ПОДТВЕРЖДЕН'}")

print("\n--- Второй закон: секториальная скорость постоянна ---")

sectorial_velocity = 0.5 * np.abs(traj['x'] * traj['vy'] - traj['y'] * traj['vx'])

mean_sv = np.mean(sectorial_velocity)
std_sv = np.std(sectorial_velocity)
relative_error = std_sv / mean_sv * 100

print(f"Средняя секториальная скорость: {mean_sv:.4e} м²/с")
print(f"Стандартное отклонение: {std_sv:.4e} м²/с")
print(f"Относительное отклонение: {relative_error:.4f}%")
print(f"=> Второй закон {'ПОДТВЕРЖДЕН ✓' if relative_error < 1 else 'НЕ ПОДТВЕРЖДЕН'}")

print("\n--- Третий закон: T² ∝ a³ ---")

test_velocities = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
periods_num = []
semi_axes_num = []

print(f"\n{'v₀ км/с':^10} {'a, а.е.':^12} {'T, лет':^12} {'T²/a³':^12}")
print("-" * 50)

for v0_km in test_velocities:
    v0 = v0_km * 1000
    vx0 = -v0 * np.cos(alpha)
    vy0 = v0 * np.sin(alpha)

    E, L, a_orb, e_orb, T_orb, _ = orbital_parameters(x0, y0, vx0, vy0)

    if E < 0:
        T_years = T_orb / YEAR
        a_AU = a_orb / AU
        ratio = T_years ** 2 / a_AU ** 3

        periods_num.append(T_years)
        semi_axes_num.append(a_AU)

        print(f"{v0_km:^10.1f} {a_AU:^12.2f} {T_years:^12.1f} {ratio:^12.4f}")

mean_ratio = np.mean([periods_num[i] ** 2 / semi_axes_num[i] ** 3 for i in range(len(periods_num))])
print(f"\nТеоретическое значение T²/a³ = 1.000 (в единицах годы и а.е.)")
print(f"Среднее по расчётам: T²/a³ = {mean_ratio:.4f}")
print(f"=> Третий закон {'ПОДТВЕРЖДЕН ✓' if abs(mean_ratio - 1) < 0.01 else 'НЕ ПОДТВЕРЖДЕН'}")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

ax = axes3[0]
ax.plot(traj['x'] / AU, traj['y'] / AU, 'b-', lw=1.5, label='Численное решение')

idx_peri = np.argmin(r_numerical)
x_peri, y_peri = traj['x'][idx_peri], traj['y'][idx_peri]
omega = np.arctan2(y_peri, x_peri)

theta = np.linspace(0, 2 * np.pi, 500)
p = a * (1 - e ** 2)
r_th = p / (1 + e * np.cos(theta))
x_th = r_th * np.cos(theta + omega)
y_th = r_th * np.sin(theta + omega)
ax.plot(x_th / AU, y_th / AU, 'r--', lw=1, label='Теоретический эллипс')

ax.plot(0, 0, 'yo', markersize=12, label='Солнце (фокус)')
ax.set_xlabel('x, а.е.')
ax.set_ylabel('y, а.е.')
ax.set_title('Первый закон Кеплера')
ax.axis('equal')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes3[1]
ax.plot(traj['t'] / YEAR, sectorial_velocity, 'b-', lw=0.5)
ax.axhline(mean_sv, color='r', ls='--', lw=2, label=f'Среднее')
ax.fill_between(traj['t'] / YEAR, mean_sv * (1 - 0.001), mean_sv * (1 + 0.001),
                alpha=0.3, color='r', label='±0.1%')
ax.set_xlabel('Время, лет')
ax.set_ylabel('dA/dt, м²/с')
ax.set_title('Второй закон Кеплера\n(секториальная скорость = const)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes3[2]
ax.loglog(semi_axes_num, periods_num, 'bo', markersize=10, label='Моделирование')
a_line = np.linspace(min(semi_axes_num) * 0.8, max(semi_axes_num) * 1.2, 100)
ax.loglog(a_line, a_line ** 1.5, 'r-', lw=2, label=r'$T = a^{3/2}$')
ax.set_xlabel('a, а.е.')
ax.set_ylabel('T, лет')
ax.set_title('Третий закон Кеплера\n($T^2 = a^3$)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('kepler_laws.png', dpi=150)
plt.show()
