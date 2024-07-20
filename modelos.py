import streamlit as st
import numpy as np
from scipy.integrate import odeint
import simpy
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Simulación de Sistemas Dinámicos y Agentes", layout="wide")

# Función para modelar un sistema dinámico (ejemplo: crecimiento poblacional)
def population_model(y, t, r, K):
    dydt = r * y * (1 - y / K)
    return dydt

# Simulación del modelo de crecimiento poblacional
def simulate_population_growth(initial_population, r, K, t):
    y0 = initial_population
    sol = odeint(population_model, y0, t, args=(r, K))
    return sol.flatten()

# Configuración de parámetros de simulación de sistemas dinámicos
st.sidebar.header("Simulación de Sistemas Dinámicos")
initial_population = st.sidebar.number_input("Población Inicial", value=10)
growth_rate = st.sidebar.slider("Tasa de Crecimiento (r)", 0.0, 1.0, 0.1)
carrying_capacity = st.sidebar.number_input("Capacidad de Carga (K)", value=100)
time_horizon = st.sidebar.slider("Horizonte de Tiempo", 1, 100, 50)
time_points = np.linspace(0, time_horizon, 500)

# Ejecutar simulación y mostrar resultados
if st.sidebar.button("Simular Crecimiento Poblacional"):
    result = simulate_population_growth(initial_population, growth_rate, carrying_capacity, time_points)
    df = pd.DataFrame({"Tiempo": time_points, "Población": result})
    st.subheader("Resultados de la Simulación de Crecimiento Poblacional")
    st.line_chart(df.set_index("Tiempo"))
    st.dataframe(df)

# Simulación de agentes usando SimPy
st.sidebar.header("Simulación Basada en Agentes")

class Agent:
    def __init__(self, env, name, process_time):
        self.env = env
        self.name = name
        self.process_time = process_time
        self.action = env.process(self.run())
        self.completion_times = []

    def run(self):
        while True:
            yield self.env.timeout(self.process_time)
            self.completion_times.append(self.env.now)

# Función para ejecutar la simulación de agentes
def simulate_agents(num_agents, process_time, simulation_time):
    env = simpy.Environment()
    agents = [Agent(env, f"Agente {i+1}", process_time) for i in range(num_agents)]
    env.run(until=simulation_time)
    return agents

num_agents = st.sidebar.number_input("Número de Agentes", value=5)
process_time = st.sidebar.slider("Tiempo de Proceso por Agente", 1, 10, 3)
simulation_time = st.sidebar.slider("Tiempo de Simulación", 1, 100, 30)

if st.sidebar.button("Simular Agentes"):
    agents = simulate_agents(num_agents, process_time, simulation_time)
    
    st.subheader("Resultados de la Simulación de Agentes")
    for agent in agents:
        st.write(f"{agent.name} completó tareas en los tiempos: {agent.completion_times}")
        
    all_times = {"Tiempo": [], "Agente": []}
    for agent in agents:
        all_times["Tiempo"].extend(agent.completion_times)
        all_times["Agente"].extend([agent.name] * len(agent.completion_times))
    
    df_agents = pd.DataFrame(all_times)
    st.line_chart(df_agents.pivot(index="Tiempo", columns="Agente", values="Tiempo"))

# Footer
st.sidebar.write("---")
st.sidebar.write("Desarrollado por [Profesor Bill Papas](https://billpapas2025.github.io/pyquickstart/)")
