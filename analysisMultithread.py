import matplotlib.pyplot as plt
import concurrent.futures
from SchedulerMain import (
    random_task_scheduling,
    performance_to_cost_scheduling,
    genetic_algorithm_scheduling,
    PSGS1_task_scheduling,
    simulated_annealing_task_scheduling,
    PSGS1_GA_task_scheduling,
    particle_swarm_optimization_scheduling,
    enhanced_hybrid_genetic_algorithm,
    base_psgs_scheduling,
    hybrid_psgs_pso_scheduling,
    create_tasks,
    create_nodes
)

# Algorithms dictionary
algorithms = {
    "Random": random_task_scheduling,
    "P2C": performance_to_cost_scheduling,
    "GA": genetic_algorithm_scheduling,
    "EMA": PSGS1_task_scheduling,
    "Simulated Annealing": simulated_annealing_task_scheduling,
    "EMA + GA": PSGS1_GA_task_scheduling,
    "PSO": particle_swarm_optimization_scheduling,
    "E-HGA": enhanced_hybrid_genetic_algorithm,
    "Base PSGS": base_psgs_scheduling,
    "Hybrid PSGS + PSO": hybrid_psgs_pso_scheduling
}

num_fog_nodes = 10
num_cloud_nodes = 5
task_sizes = [100, 200, 300, 400, 500]

results = {algo: {"makespan": [], "energy_consumption": [], 
                  "processing_cost": [], "fitness_value": []} 
           for algo in algorithms}

def run_algorithm(algo_name, algo_func, tasks_list, fog_nodes, cloud_nodes, num_tasks):
    res = algo_func(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
    return (algo_name, num_tasks, res)

# Parallel execution using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for num_tasks in task_sizes:
        tasks_list = create_tasks(num_tasks)
        fog_nodes, cloud_nodes = create_nodes(num_fog_nodes, num_cloud_nodes)
        for algo_name, algo_func in algorithms.items():
            futures.append(executor.submit(run_algorithm, algo_name, algo_func, tasks_list, fog_nodes, cloud_nodes, num_tasks))

    for future in concurrent.futures.as_completed(futures):
        algo_name, num_tasks_completed, res = future.result()
        idx = task_sizes.index(num_tasks_completed)
        results[algo_name]["makespan"].insert(idx, res["makespan"])
        results[algo_name]["energy_consumption"].insert(idx, res["energy_consumption"])
        results[algo_name]["processing_cost"].insert(idx, res["processing_cost"])
        results[algo_name]["fitness_value"].insert(idx, res["fitness_value"])

# Plotting function (unchanged)
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(10,6))
    for algo_name in algorithms.keys():
        plt.plot(task_sizes, results[algo_name][metric_name], marker='o', label=algo_name)
    
    plt.title(f"{ylabel} vs Number of Tasks")
    plt.xlabel("Number of Tasks")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main metrics plots
plot_metric("makespan", "Makespan (ms)")
plot_metric("energy_consumption", "Energy Consumption (units)")
plot_metric("processing_cost", "Processing Cost (units)")
plot_metric("fitness_value", "Fitness Value")
