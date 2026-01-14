import matplotlib.pyplot as plt
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

# Parameters
num_fog_nodes = 10
num_cloud_nodes = 5
task_sizes = [100, 200, 300, 400, 500]

# Collecting results
results = {algo: {"makespan": [], "energy_consumption": [], "processing_cost": [], "fitness_value": []}
           for algo in algorithms}

for num_tasks in task_sizes:
    print(f"Running simulations for {num_tasks} tasks...")
    tasks_list = create_tasks(num_tasks)
    fog_nodes, cloud_nodes = create_nodes(num_fog_nodes, num_cloud_nodes)

    for algo_name, algo_func in algorithms.items():
        print(f" - Executing {algo_name}")
        res = algo_func(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
        results[algo_name]["makespan"].append(res["makespan"])
        results[algo_name]["energy_consumption"].append(res["energy_consumption"])
        results[algo_name]["processing_cost"].append(res["processing_cost"])
        results[algo_name]["fitness_value"].append(res["fitness_value"])

# Plotting function
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

# Plotting main metrics
plot_metric("makespan", "Makespan (ms)")
plot_metric("energy_consumption", "Energy Consumption (units)")
plot_metric("processing_cost", "Processing Cost (units)")
plot_metric("fitness_value", "Fitness Value")

# Additional insightful graphs: Normalized comparison at largest task size (500 tasks)
def normalized_comparison(num_tasks_index=-1):
    metrics = ["makespan", "energy_consumption", "processing_cost"]
    
    for metric in metrics:
        values = [results[algo][metric][num_tasks_index] for algo in algorithms]
        min_val = min(values)
        normalized_values = [val/min_val for val in values]

        plt.figure(figsize=(10,6))
        plt.bar(algorithms.keys(), normalized_values, color='skyblue')
        plt.title(f"Normalized {metric.replace('_',' ').title()} Comparison at {task_sizes[num_tasks_index]} Tasks")
        plt.ylabel(f"{metric.replace('_',' ').title()} (Normalized to Best=1.0)")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

normalized_comparison()

# Additional graph: Fitness Value improvement over Random Scheduling (% improvement)
def fitness_improvement_over_random():
    random_fitness = results["Random"]["fitness_value"]
    
    plt.figure(figsize=(10,6))
    
    for algo_name in algorithms:
        if algo_name == "Random":
            continue
        improvement_percent = [(fv - rf)/rf * 100 for fv, rf in zip(results[algo_name]["fitness_value"], random_fitness)]
        plt.plot(task_sizes, improvement_percent, marker='o', label=algo_name)
    
    plt.title("Fitness Value Improvement (%) over Random Scheduling")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Improvement (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

fitness_improvement_over_random()
