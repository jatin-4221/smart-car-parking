import random
import math
import numpy as np

MAX_VALUE = 1000
NUMBER_OF_RUNS = 20


class ComputingTask:
    def __init__(self, task_id, size=0, memory_requirement=0, deadline=0, penalty_factor=0.0, quality_factor=0.0, input_data_size=0, output_data_size=0, response_time=0):
        self.task_id = task_id
        self.size = size
        self.memory_requirement = memory_requirement
        self.deadline = deadline
        self.penalty_factor = penalty_factor
        self.quality_factor = quality_factor
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.response_time = response_time


class ComputingNode:
    def __init__(self, node_id, processing_capacity=0, memory_capacity=0, bandwidth=0, available_time=0, delay=0, power_efficiency=0.0, cost_efficiency=0.0, processing_cost_coefficient=0.0, min_power_consumption=0.0, max_power_consumption=0.0):
        self.node_id = node_id
        self.processing_capacity = processing_capacity
        self.memory_capacity = memory_capacity
        self.bandwidth = bandwidth
        self.available_time = available_time
        self.delay = delay
        self.power_efficiency = power_efficiency
        self.cost_efficiency = cost_efficiency
        self.processing_cost_coefficient = processing_cost_coefficient
        self.min_power_consumption = min_power_consumption
        self.max_power_consumption = max_power_consumption
        self.energy_consumed = 0.0
        self.processing_cost = 0.0
        self.makespan_share = 0.0
        self.fitness_value = 0.0


def create_tasks(num_tasks):
    task_list = []
    for i in range(num_tasks):
        task = ComputingTask(i)
        x = random.randint(0, 2)
        if x == 0:
            task.size = random.randint(100, 10000)
            task.deadline = random.randint(100, 500)
        elif x == 1:
            task.size = random.randint(1028, 4280)
            task.deadline = random.randint(500, 2500)
        else:
            task.size = random.randint(5123, 9784)
            task.deadline = random.randint(2500, 10000)
        task.memory_requirement = random.randint(50, 200)
        task.penalty_factor = random.uniform(0.01, 0.5)
        task.quality_factor = random.uniform(9000.0, 10000.0) / 100.0
        task.input_data_size = random.randint(100, 10000)
        task.output_data_size = random.randint(1, 1000)
        task_list.append(task)
    return task_list


def create_nodes(num_fog_nodes, num_cloud_nodes):
    fog_nodes = []
    cloud_nodes = []

    for i in range(num_fog_nodes):
        node = ComputingNode(i)
        node.processing_capacity = random.randint(500, 1500)
        node.memory_capacity = random.randint(150, 250)
        node.bandwidth = random.randint(10, 1000)
        node.processing_cost_coefficient = random.uniform(0.1, 0.4)
        node.delay = random.randint(1, 10)
        node.max_power_consumption = random.uniform(40, 100)
        node.min_power_consumption = random.uniform(0.6, 1.0) * node.max_power_consumption
        fog_nodes.append(node)

    for i in range(num_cloud_nodes):
        node = ComputingNode(i)
        node.processing_capacity = random.randint(3000, 5000)
        node.memory_capacity = random.randint(8192, 65536)
        node.bandwidth = random.randint(100, 10000)
        node.processing_cost_coefficient = random.uniform(0.7, 1.0)
        node.delay = random.randint(200, 500)
        node.max_power_consumption = random.uniform(200, 400)
        node.min_power_consumption = random.uniform(0.6, 1.0) * node.max_power_consumption
        cloud_nodes.append(node)

    return fog_nodes, cloud_nodes


def sort_tasks_by_deadline(tasks):
    return sorted(tasks, key=lambda task: task.deadline)


def random_task_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        makespan = 0
        total_penalty = 0
        tasks_meeting_deadline = 0
        deadline_violation_cost = 0
        processing_cost = 0
        energy_consumption = 0
        fitness_value = 0

        for task in tasks:
            x = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)
            if x < num_fog_nodes:
                node = fog[x]
            else:
                node = cloud[x - num_fog_nodes]

            execution_time = (float(task.size) / node.processing_capacity) * 1000
            node.available_time += execution_time
            task.response_time = node.available_time + node.delay
            node.processing_cost += (float(task.size) / node.processing_capacity * node.processing_cost_coefficient)

        for task in tasks:
            time_over_deadline = task.response_time - task.deadline
            if time_over_deadline < 0:
                time_over_deadline = 0
            quality_temp = time_over_deadline * 100 / task.deadline - 100 + task.quality_factor
            if quality_temp > 0:
                deadline_violation_cost += (quality_temp * task.penalty_factor)

            if task.response_time > task.deadline:
                total_penalty += task.response_time - task.deadline
            else:
                tasks_meeting_deadline += 1

        for node in fog + cloud:
            if node.available_time > makespan:
                makespan = node.available_time
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost

        if energy_consumption == 0:
            energy_consumption = 1e-6
        if processing_cost == 0:
            processing_cost = 1e-6
        if makespan == 0:
            makespan = 1e-6

        fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value

    print("\n*************Final Results (Random Scheduling)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }

def calculate_minimum_values(tasks, fog_nodes, cloud_nodes):
    total_task_size = sum(task.size for task in tasks)
    total_cpu_fog = sum(node.processing_capacity for node in fog_nodes)
    total_cpu_cloud = sum(node.processing_capacity for node in cloud_nodes)

    min_makespan = total_task_size / (total_cpu_fog + total_cpu_cloud)

    index_f_max_pe = max(fog_nodes, key=lambda node: node.processing_capacity / node.max_power_consumption).node_id
    index_c_max_pe = max(cloud_nodes, key=lambda node: node.processing_capacity / node.max_power_consumption).node_id
    index_f_max_ce = max(fog_nodes, key=lambda node: node.processing_capacity / node.processing_cost_coefficient).node_id
    index_c_max_ce = max(cloud_nodes, key=lambda node: node.processing_capacity / node.processing_cost_coefficient).node_id

    min_energy_consumption = min(
        total_task_size / fog_nodes[index_f_max_pe].processing_capacity * fog_nodes[index_f_max_pe].max_power_consumption,
        total_task_size / cloud_nodes[index_c_max_pe].processing_capacity * cloud_nodes[index_c_max_pe].max_power_consumption,
    )

    min_processing_cost = min(
        total_task_size / fog_nodes[index_f_max_ce].processing_capacity * fog_nodes[index_f_max_ce].processing_cost_coefficient,
        total_task_size / cloud_nodes[index_c_max_ce].processing_capacity * cloud_nodes[index_c_max_ce].processing_cost_coefficient,
    )

    return min_makespan, min_energy_consumption, min_processing_cost


def performance_to_cost_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        makespan = 0
        total_penalty = 0
        tasks_meeting_deadline = 0
        deadline_violation_cost = 0
        processing_cost = 0
        energy_consumption = 0
        fitness_value = 0

        for task in tasks:
            x = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)
            y = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)

            if x < num_fog_nodes:
                fog_node_x = fog[x]
            else:
                fog_node_x = cloud[x - num_fog_nodes]

            if y < num_fog_nodes:
                fog_node_y = fog[y]
            else:
                fog_node_y = cloud[y - num_fog_nodes]

            if fog_node_x.available_time + (task.size / fog_node_x.processing_capacity) * 1000 < fog_node_y.available_time + (task.size / fog_node_y.processing_capacity) * 1000:
                node = fog_node_x
            else:
                node = fog_node_y

            execution_time = (float(task.size) / node.processing_capacity) * 1000
            node.available_time += execution_time
            task.response_time = node.available_time + node.delay
            node.processing_cost += (float(task.size) / node.processing_capacity * node.processing_cost_coefficient)

        for task in tasks:
            time_over_deadline = task.response_time - task.deadline
            if time_over_deadline < 0:
                time_over_deadline = 0
            quality_temp = time_over_deadline * 100 / task.deadline - 100 + task.quality_factor
            if quality_temp > 0:
                deadline_violation_cost += (quality_temp * task.penalty_factor)

            if task.response_time > task.deadline:
                total_penalty += task.response_time - task.deadline
            else:
                tasks_meeting_deadline += 1

        for node in fog + cloud:
            if node.available_time > makespan:
                makespan = node.available_time
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost

        if energy_consumption == 0:
            energy_consumption = 1e-6
        if processing_cost == 0:
            processing_cost = 1e-6
        if makespan == 0:
            makespan = 1e-6

        fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value

    print("\n*************Final Results (P2C Scheduling)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def genetic_algorithm_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0
    mutation_rate = 0.05
    population_size = 150
    num_iterations = 1000
    num_elite_individuals = 2

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        population_solutions = [[random.randint(0, num_fog_nodes + num_cloud_nodes - 1) for _ in range(num_tasks)] for _ in range(population_size)]
        solution_makespan = [0] * population_size
        solution_cost = [0] * population_size
        solution_objective_function = [0] * population_size

        for itr in range(num_iterations):
            for i in range(population_size):
                for node in fog + cloud:
                    node.available_time = 0
                    node.processing_cost = 0

                for j, task in enumerate(tasks):
                    x = population_solutions[i][j]
                    if x < num_fog_nodes:
                        node = fog[x]
                    else:
                        node = cloud[x - num_fog_nodes]

                    execution_time = (float(task.size) / node.processing_capacity) * 1000
                    node.available_time += execution_time
                    node.processing_cost += (float(task.size) / node.processing_capacity * node.processing_cost_coefficient)

                solution_makespan[i] = max(node.available_time for node in fog + cloud)
                solution_cost[i] = sum(node.processing_cost for node in fog + cloud)
                solution_objective_function[i] = 0.67 * min_makespan / solution_makespan[i] + 0.33 * min_processing_cost / solution_cost[i]

            sorted_indices = sorted(range(population_size), key=lambda i: solution_objective_function[i], reverse=True)
            next_generation_solutions = [population_solutions[i] for i in sorted_indices[:num_elite_individuals]]

            sum_fitness = sum(solution_objective_function)
            normalized_objective_function = [obj_func / sum_fitness for obj_func in solution_objective_function]
            for i in range(1, population_size):
                normalized_objective_function[i] += normalized_objective_function[i - 1]

            while len(next_generation_solutions) < population_size:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(population_size) if z1 <= normalized_objective_function[k])
                y = next(k for k in range(population_size) if z2 <= normalized_objective_function[k])

                child1, child2 = population_solutions[x][:num_tasks // 2] + population_solutions[y][num_tasks // 2:], population_solutions[y][:num_tasks // 2] + population_solutions[x][num_tasks // 2:]
                next_generation_solutions.append(child1)
                if len(next_generation_solutions) < population_size:
                    next_generation_solutions.append(child2)

            for i in range(population_size):
                if random.uniform(0, 1) < mutation_rate:
                    j = random.randint(0, num_tasks - 1)
                    next_generation_solutions[i][j] = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)

            population_solutions = next_generation_solutions

        best_solution_index = max(range(population_size), key=lambda i: solution_objective_function[i])

        makespan, energy_consumption, processing_cost = 0, 0, 0

        for node in fog + cloud:
            node.available_time = 0
            node.processing_cost = 0
            node.energy_consumed = 0

        for j, task in enumerate(tasks):
            x = population_solutions[best_solution_index][j]
            if x < num_fog_nodes:
                node = fog[x]
            else:
                node = cloud[x - num_fog_nodes]

            execution_time = (float(task.size) / node.processing_capacity) * 1000
            node.available_time += execution_time
            task.response_time = node.available_time + node.delay
            node.processing_cost += (float(task.size) / node.processing_capacity * node.processing_cost_coefficient)

        for node in fog + cloud:
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost
            if node.available_time > makespan:
                makespan = node.available_time

        if energy_consumption == 0:
            energy_consumption = 1e-6
        if processing_cost == 0:
            processing_cost = 1e-6
        if makespan == 0:
            makespan = 1e-6

        fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value

    print("\n*************Final Results (Genetic Algorithm)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def PSGS1_task_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        makespan = 0
        total_penalty = 0
        tasks_meeting_deadline = 0
        deadline_violation_cost = 0
        processing_cost = 0
        energy_consumption = 0
        fitness_value = 0

        fog_max_power_efficiency = max(fog, key=lambda x: x.power_efficiency).power_efficiency if fog else 0
        fog_max_cost_efficiency = max(fog, key=lambda x: x.cost_efficiency).cost_efficiency if fog else 0
        cloud_max_power_efficiency = max(cloud, key=lambda x: x.power_efficiency).power_efficiency if cloud else 0
        cloud_max_cost_efficiency = max(cloud, key=lambda x: x.cost_efficiency).cost_efficiency if cloud else 0

        if fog_max_power_efficiency != 0:
            for node in fog:
                node.power_efficiency /= fog_max_power_efficiency
        if fog_max_cost_efficiency != 0:
            for node in fog:
                node.cost_efficiency /= fog_max_cost_efficiency

        if cloud_max_power_efficiency != 0:
            for node in cloud:
                node.power_efficiency /= cloud_max_power_efficiency
        if cloud_max_cost_efficiency != 0:
            for node in cloud:
                node.cost_efficiency /= cloud_max_cost_efficiency

        for task in tasks:
            min_makespan_so_far = float('inf')
            best_node_for_makespan = None

            for node in fog + cloud:
                execution_time = (float(task.size) / node.processing_capacity) * 1000
                current_makespan = node.available_time + execution_time

                if current_makespan < min_makespan_so_far:
                    min_makespan_so_far = current_makespan
                    best_node_for_makespan = node

            max_fitness = -float('inf')
            best_node_overall = None

            for node in fog + cloud:
                execution_time = (float(task.size) / node.processing_capacity) * 1000
                makespan_metric = (min_makespan_so_far / (node.available_time + execution_time))
                current_fitness = 0.25 * node.power_efficiency + 0.25 * node.cost_efficiency + 0.5 * makespan_metric

                if current_fitness > max_fitness:
                    max_fitness = current_fitness
                    best_node_overall = node

            best_node_overall.available_time += (float(task.size) / best_node_overall.processing_capacity) * 1000
            task.response_time = best_node_overall.available_time + best_node_overall.delay
            best_node_overall.processing_cost += (float(task.size) / best_node_overall.processing_capacity * best_node_overall.processing_cost_coefficient)

        for task in tasks:
            time_over_deadline = task.response_time - task.deadline
            if time_over_deadline < 0:
                time_over_deadline = 0
            quality_temp = time_over_deadline * 100 / task.deadline - 100 + task.quality_factor
            if quality_temp > 0:
                deadline_violation_cost += (quality_temp * task.penalty_factor)

            if task.response_time > task.deadline:
                total_penalty += task.response_time - task.deadline
            else:
                tasks_meeting_deadline += 1

        for node in fog + cloud:
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost
            if node.available_time > makespan:
                makespan = node.available_time

        if energy_consumption == 0:
            energy_consumption = 1e-6
        if processing_cost == 0:
            processing_cost = 1e-6
        if makespan == 0:
            makespan = 1e-6

        fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value

    print("\n*************Final Results (EMA Scheduling)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def simulated_annealing_task_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0
    initial_temperature = 1000.0
    final_temperature = 1.0
    cooling_rate = 0.9
    iterations_per_temperature = 1000

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        current_solution = [random.randint(0, num_fog_nodes + num_cloud_nodes - 1) for _ in range(num_tasks)]
        current_cost = evaluate_current_solution(tasks, fog, cloud, current_solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

        best_solution = current_solution[:]
        best_cost = current_cost

        current_temperature = initial_temperature

        while current_temperature > final_temperature:
            for _ in range(iterations_per_temperature):
                neighbor_solution = current_solution[:]
                task_to_modify = random.randint(0, num_tasks - 1)
                neighbor_solution[task_to_modify] = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)

                neighbor_cost = evaluate_current_solution(tasks, fog, cloud, neighbor_solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

                cost_difference = (current_cost - neighbor_cost) / current_temperature
                cost_difference = np.clip(cost_difference, -700, 700)
                acceptance_probability = np.exp(cost_difference)

                if neighbor_cost < current_cost or random.random() < acceptance_probability:
                    current_solution = neighbor_solution[:]
                    current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution[:]
                    best_cost = current_cost

            current_temperature *= cooling_rate

        final_makespan, final_energy_consumption, final_processing_cost, final_fitness_value = evaluate_current_solution(tasks, fog, cloud, best_solution,
                                                                                   min_makespan, min_energy_consumption, min_processing_cost,
                                                                                   num_fog_nodes, num_cloud_nodes, return_all=True)

        total_makespan_sum += final_makespan
        total_energy_consumption_sum += final_energy_consumption
        total_processing_cost_sum += final_processing_cost
        total_fitness_value_sum += final_fitness_value

    print("\n*************Final Results (Simulated Annealing)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def PSGS1_GA_task_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0
    mutation_rate = 0.05
    population_size = 150
    num_iterations = 1000
    num_elite_individuals = 2

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        population_solutions = [[random.randint(0, num_fog_nodes + num_cloud_nodes - 1) for _ in range(num_tasks)] for _ in range(population_size)]
        solution_makespan = [0] * population_size
        solution_cost = [0] * population_size
        solution_objective_function = [0] * population_size

        for itr in range(num_iterations):
            for i in range(population_size):
                solution_makespan[i], solution_cost[i], _, solution_objective_function[i] = evaluate_current_solution(tasks, fog, cloud, population_solutions[i], min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes, return_all=True)

            sorted_indices = sorted(range(population_size), key=lambda i: solution_objective_function[i], reverse=True)
            next_generation_solutions = [population_solutions[i] for i in sorted_indices[:num_elite_individuals]]

            sum_fitness = sum(solution_objective_function)
            normalized_objective_function = [obj_func / sum_fitness for obj_func in solution_objective_function]
            for i in range(1, population_size):
                normalized_objective_function[i] += normalized_objective_function[i - 1]

            while len(next_generation_solutions) < population_size:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(population_size) if z1 <= normalized_objective_function[k])
                y = next(k for k in range(population_size) if z2 <= normalized_objective_function[k])

                child1, child2 = population_solutions[x][:num_tasks // 2] + population_solutions[y][num_tasks // 2:], population_solutions[y][:num_tasks // 2] + population_solutions[x][num_tasks // 2:]
                next_generation_solutions.append(child1)
                if len(next_generation_solutions) < population_size:
                    next_generation_solutions.append(child2)

            for i in range(population_size):
                if random.uniform(0, 1) < mutation_rate:
                    j = random.randint(0, num_tasks - 1)
                    next_generation_solutions[i][j] = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)

            for i in range(num_elite_individuals):
                next_generation_solutions[i] = refine_solution_with_proposed_method(tasks, fog, cloud, next_generation_solutions[i], num_fog_nodes, num_cloud_nodes)

            population_solutions = next_generation_solutions

        best_solution_index = max(range(population_size), key=lambda i: solution_objective_function[i])

        makespan, energy_consumption, processing_cost = 0, 0, 0

        for node in fog + cloud:
            node.available_time = 0
            node.processing_cost = 0
            node.energy_consumed = 0

        for j, task in enumerate(tasks):
            x = population_solutions[best_solution_index][j]
            if x < num_fog_nodes:
                node = fog[x]
            else:
                node = cloud[x - num_fog_nodes]

            execution_time = (float(task.size) / node.processing_capacity) * 1000
            node.available_time += execution_time
            task.response_time = node.available_time + node.delay
            node.processing_cost += (float(task.size) / node.processing_capacity * node.processing_cost_coefficient)

        for node in fog + cloud:
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost
            if node.available_time > makespan:
                makespan = node.available_time

        if energy_consumption == 0:
            energy_consumption = 1e-6
        if processing_cost == 0:
            processing_cost = 1e-6
        if makespan == 0:
            makespan = 1e-6

        fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value

    print("\n*************Final Results (GA + EMA)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def refine_solution_with_proposed_method(tasks, fog_nodes, cloud_nodes, solution, num_fog_nodes, num_cloud_nodes):
    for task_idx, node_idx in enumerate(solution):
        task = tasks[task_idx]
        min_makespan_so_far = float('inf')
        best_node_for_makespan = None

        for node in fog_nodes + cloud_nodes:
            execution_time = (float(task.size) / node.processing_capacity) * 1000
            current_makespan = node.available_time + execution_time

            if current_makespan < min_makespan_so_far:
                min_makespan_so_far = current_makespan

        max_fitness = -float('inf')
        best_node_overall = None

        for node in fog_nodes + cloud_nodes:
            execution_time = (float(task.size) / node.processing_capacity) * 1000
            makespan_metric = (min_makespan_so_far / (node.available_time + execution_time))
            current_fitness = 0.25 * node.power_efficiency + 0.25 * node.cost_efficiency + 0.5 * makespan_metric

            if current_fitness > max_fitness:
                max_fitness = current_fitness
                best_node_overall = node

        best_node_overall.available_time += (float(task.size) / best_node_overall.processing_capacity) * 1000
        solution[task_idx] = best_node_overall.node_id

    return solution


def evaluate_current_solution(tasks, fog_nodes, cloud_nodes, solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes, return_all=False):
    makespan = 0
    processing_cost = 0
    energy_consumption = 0

    for node in fog_nodes + cloud_nodes:
        node.available_time = 0
        node.processing_cost = 0
        node.energy_consumed = 0

    for task_idx, node_idx in enumerate(solution):
        if node_idx < num_fog_nodes:
            node = fog_nodes[node_idx]
        else:
            node = cloud_nodes[node_idx - num_fog_nodes]

        execution_time = (float(tasks[task_idx].size) / node.processing_capacity) * 1000
        node.available_time += execution_time
        tasks[task_idx].response_time = node.available_time + node.delay
        node.processing_cost += (float(tasks[task_idx].size) / node.processing_capacity * node.processing_cost_coefficient)

    for node in fog_nodes + cloud_nodes:
        node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
        energy_consumption += node.energy_consumed
        processing_cost += node.processing_cost
        if node.available_time > makespan:
            makespan = node.available_time

    if energy_consumption == 0:
        energy_consumption = 1e-6
    if processing_cost == 0:
        processing_cost = 1e-6
    if makespan == 0:
        makespan = 1e-6

    normalized_makespan = min_makespan / makespan
    normalized_energy_consumption = min_energy_consumption / energy_consumption
    normalized_processing_cost = min_processing_cost / processing_cost

    fitness_value = 0.34 * normalized_energy_consumption + 0.33 * normalized_processing_cost + 0.33 * normalized_makespan

    if return_all:
        return makespan, energy_consumption, processing_cost, fitness_value
    else:
        return fitness_value


def particle_swarm_optimization_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0
    num_particles = 50
    max_iterations = 100
    inertia_weight = 0.5
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        particles = [np.random.randint(0, num_fog_nodes + num_cloud_nodes, num_tasks) for _ in range(num_particles)]
        velocities = [np.random.rand(num_tasks) for _ in range(num_particles)]
        personal_best_positions = particles.copy()
        personal_best_scores = [evaluate_current_solution(tasks, fog, cloud, p, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)
                                for p in particles]
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
        global_best_score = max(personal_best_scores)

        for iter in range(max_iterations):
            for i in range(num_particles):
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coefficient * np.random.rand(num_tasks) * (personal_best_positions[i] - particles[i]) +
                                 social_coefficient * np.random.rand(num_tasks) * (global_best_position - particles[i]))

                particles[i] = np.clip(particles[i] + velocities[i], 0, num_fog_nodes + num_cloud_nodes - 1).astype(int)

                current_score = evaluate_current_solution(tasks, fog, cloud, particles[i], min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

                if current_score > personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = current_score

            best_particle_index = np.argmax(personal_best_scores)
            if personal_best_scores[best_particle_index] > global_best_score:
                global_best_position = personal_best_positions[best_particle_index].copy()
                global_best_score = personal_best_scores[best_particle_index]

        final_makespan, final_energy_consumption, final_processing_cost, final_fitness_value = evaluate_current_solution(tasks, fog, cloud,
                                                                                   global_best_position,
                                                                                   min_makespan, min_energy_consumption, min_processing_cost,
                                                                                   num_fog_nodes, num_cloud_nodes, return_all=True)

        total_makespan_sum += final_makespan
        total_energy_consumption_sum += final_energy_consumption
        total_processing_cost_sum += final_processing_cost
        total_fitness_value_sum += final_fitness_value

    print("\n*************Final Results (PSO Algorithm)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }


def enhanced_hybrid_genetic_algorithm(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0
    mutation_rate = 0.05
    population_size = 100
    num_iterations = 500
    num_elite_individuals = 5

    for r in range(NUMBER_OF_RUNS):
        tasks = task_list.copy()
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        population_solutions = [[random.randint(0, num_fog_nodes + num_cloud_nodes - 1) for _ in range(num_tasks)] for _ in range(population_size)]
        solution_makespan = [0] * population_size
        solution_cost = [0] * population_size
        solution_objective_function = [0] * population_size

        for itr in range(num_iterations):
            for i in range(population_size):
                solution_objective_function[i] = evaluate_current_solution(tasks, fog, cloud, population_solutions[i], min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

            sorted_indices = sorted(range(population_size), key=lambda i: solution_objective_function[i], reverse=True)
            next_generation_solutions = [population_solutions[i] for i in sorted_indices[:num_elite_individuals]]

            sum_fitness = sum(solution_objective_function)
            normalized_objective_function = [obj_func / sum_fitness for obj_func in solution_objective_function]
            for i in range(1, population_size):
                normalized_objective_function[i] += normalized_objective_function[i - 1]

            while len(next_generation_solutions) < population_size:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(population_size) if z1 <= normalized_objective_function[k])
                y = next(k for k in range(population_size) if z2 <= normalized_objective_function[k])

                child1, child2 = perform_crossover(population_solutions[x], population_solutions[y], num_tasks, num_fog_nodes, num_cloud_nodes)
                next_generation_solutions.append(child1)
                if len(next_generation_solutions) < population_size:
                    next_generation_solutions.append(child2)

            for i in range(population_size):
                if random.uniform(0, 1) < mutation_rate:
                    perform_mutation(next_generation_solutions[i], num_tasks, num_fog_nodes, num_cloud_nodes)

            population_solutions = next_generation_solutions

        best_solution_index = max(range(population_size), key=lambda i: solution_objective_function[i])
        best_solution = population_solutions[best_solution_index]
        best_solution = perform_local_search(tasks, fog, cloud, best_solution, min_makespan, min_energy_consumption, min_processing_cost, num_tasks, num_fog_nodes, num_cloud_nodes)

        final_makespan, final_energy_consumption, final_processing_cost, final_fitness_value = evaluate_current_solution(tasks, fog, cloud, best_solution,
                                                                                   min_makespan, min_energy_consumption, min_processing_cost,
                                                                                   num_fog_nodes, num_cloud_nodes, return_all=True)

        total_makespan_sum += final_makespan
        total_energy_consumption_sum += final_energy_consumption
        total_processing_cost_sum += final_processing_cost
        total_fitness_value_sum += final_fitness_value

    print("\n*************Final Results (E-HGA)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }

def base_psgs_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0

    for _ in range(NUMBER_OF_RUNS):
        tasks = [ComputingTask(**vars(t)) for t in task_list]
        fog, cloud = create_nodes(num_fog_nodes, num_cloud_nodes)
        min_makespan, min_energy_consumption, min_processing_cost = calculate_minimum_values(tasks, fog, cloud)

        makespan = energy_consumption = processing_cost = fitness_value = 0

        for task in tasks:
            DSL = []
            for node in fog:
                execution_time = (task.size / node.processing_capacity) * 1000
                response_time = node.available_time + execution_time + node.delay
                if response_time <= task.deadline:
                    DSL.append(node)

            if DSL:
                DSL.sort(key=lambda n: (n.available_time, n.processing_cost_coefficient))
                selected_node = DSL[0]
            else:
                selected_node = random.choice(cloud)

            execution_time = (task.size / selected_node.processing_capacity) * 1000
            selected_node.available_time += execution_time
            task.response_time = selected_node.available_time + selected_node.delay
            selected_node.processing_cost += (task.size / selected_node.processing_capacity * selected_node.processing_cost_coefficient)

        for node in fog + cloud:
            if node.available_time > makespan:
                makespan = node.available_time

        for node in fog + cloud:
            node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan - node.available_time) / 1000.0) * node.min_power_consumption
            energy_consumption += node.energy_consumed
            processing_cost += node.processing_cost

        energy_consumption = max(energy_consumption, 1e-6)
        processing_cost = max(processing_cost, 1e-6)
        makespan = max(makespan, 1e-6)

        fitness_value = (0.34 * min_energy_consumption / energy_consumption +
                         0.33 * min_processing_cost / processing_cost +
                         0.33 * min_makespan / makespan * 1000)

        total_makespan_sum += makespan
        total_energy_consumption_sum += energy_consumption
        total_processing_cost_sum += processing_cost
        total_fitness_value_sum += fitness_value
        
    print("\n*************Final Results (Base PSGS)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }

def hybrid_psgs_pso_scheduling(task_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes):
    total_makespan_sum, total_energy_consumption_sum, total_processing_cost_sum, total_fitness_value_sum = 0, 0, 0, 0

    def fitness(solution, tasks_local, nodes_local):
        energy = latency = penalty = 0
        for i, node_idx in enumerate(solution):
            task_local = tasks_local[i]
            node_local = nodes_local[node_idx]
            exec_time_local = (task_local.size / node_local.processing_capacity) * 1000
            response_local = node_local.available_time + exec_time_local + node_local.delay
            energy += (exec_time_local/1000)*node_local.max_power_consumption
            latency += response_local
            if response_local > task_local.deadline:
                penalty += (response_local - task_local.deadline)*10
        return energy + latency + penalty

    for _ in range(NUMBER_OF_RUNS):
        tasks=[ComputingTask(**vars(t)) for t in task_list]
        fog, cloud=create_nodes(num_fog_nodes,num_cloud_nodes)
        nodes=fog+cloud

        min_makespan,min_energy,min_proc=calculate_minimum_values(tasks,fog,cloud)

        swarm_size=20
        iterations=30
        num_all_nodes=num_fog_nodes+num_cloud_nodes

        swarm_pos=np.random.randint(num_all_nodes,size=(swarm_size,num_tasks))
        swarm_vel=np.zeros((swarm_size,num_tasks))

        p_best_pos=swarm_pos.copy()
        p_best_scores=np.array([fitness(p,tasks,nodes)for p in swarm_pos])
        g_best_pos=p_best_pos[np.argmin(p_best_scores)]

        w,c1,c2=0.5,1.5,1.5

        for _ in range(iterations):
            r1,r2=np.random.rand(swarm_size,num_tasks),np.random.rand(swarm_size,num_tasks)
            swarm_vel=w*swarm_vel+c1*r1*(p_best_pos-swarm_pos)+c2*r2*(g_best_pos-swarm_pos)
            swarm_pos=(swarm_pos+np.round(swarm_vel).astype(int))%num_all_nodes

            for i in range(swarm_size):
                score=fitness(swarm_pos[i],tasks,nodes)
                if score<p_best_scores[i]:
                    p_best_scores[i]=score
                    p_best_pos[i]=swarm_pos[i]

            g_best_pos=p_best_pos[np.argmin(p_best_scores)]

        makespan=energy=proc_cost=fit_val=0

        for i,node_idx in enumerate(g_best_pos):
            task=tasks[i]
            node=nodes[node_idx]
            exec_t=(task.size/node.processing_capacity)*1000
            node.available_time+=exec_t
            task.response_time=node.available_time+node.delay
            node.processing_cost+=(task.size/node.processing_capacity*node.processing_cost_coefficient)

        for n in nodes:
            if n.available_time>makespan:
                makespan=n.available_time

        for n in nodes:
            n.energy_consumed=(n.available_time/1000)*n.max_power_consumption+((makespan-n.available_time)/1000)*n.min_power_consumption
            energy+=n.energy_consumed
            proc_cost+=n.processing_cost

        energy=max(energy,1e-6)
        proc_cost=max(proc_cost,1e-6)
        makespan=max(makespan,1e-6)

        fit_val=(0.34*min_energy/energy+
                 0.33*min_proc/proc_cost+
                 0.33*min_makespan/makespan*1000)

        total_makespan_sum+=makespan
        total_energy_consumption_sum+=energy
        total_processing_cost_sum+=proc_cost
        total_fitness_value_sum+=fit_val
    print("\n*************Final Results (Hybrid PSGS PSO)*************")
    print(f"Makespan: {total_makespan_sum / 1000.0 / NUMBER_OF_RUNS}")
    print(f"Energy Consumption: {total_energy_consumption_sum / NUMBER_OF_RUNS}")
    print(f"Processing Cost: {total_processing_cost_sum / NUMBER_OF_RUNS}")
    print(f"Fitness Value: {total_fitness_value_sum / NUMBER_OF_RUNS}")
    return {
        "makespan": total_makespan_sum / NUMBER_OF_RUNS,
        "energy_consumption": total_energy_consumption_sum / NUMBER_OF_RUNS,
        "processing_cost": total_processing_cost_sum / NUMBER_OF_RUNS,
        "fitness_value": total_fitness_value_sum / NUMBER_OF_RUNS
    }

def perform_crossover(parent1, parent2, num_tasks, num_fog_nodes, num_cloud_nodes):
    cut_point = num_tasks // 2
    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]
    return child1, child2


def perform_mutation(solution, num_tasks, num_fog_nodes, num_cloud_nodes):
    task_to_modify = random.randint(0, num_tasks - 1)
    solution[task_to_modify] = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)


def perform_local_search(tasks, fog_nodes, cloud_nodes, solution, min_makespan, min_energy_consumption, min_processing_cost, num_tasks, num_fog_nodes, num_cloud_nodes):
    best_solution = solution[:]
    best_cost = evaluate_current_solution(tasks, fog_nodes, cloud_nodes, best_solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

    for _ in range(100):
        neighbor_solution = best_solution[:]
        task_to_modify = random.randint(0, num_tasks - 1)
        neighbor_solution[task_to_modify] = random.randint(0, num_fog_nodes + num_cloud_nodes - 1)
        neighbor_cost = evaluate_current_solution(tasks, fog_nodes, cloud_nodes, neighbor_solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes)

        if neighbor_cost > best_cost:
            best_solution = neighbor_solution[:]
            best_cost = neighbor_cost

    return best_solution


def evaluate_current_solution(tasks, fog_nodes, cloud_nodes, solution, min_makespan, min_energy_consumption, min_processing_cost, num_fog_nodes, num_cloud_nodes, return_all=False):
    makespan = 0
    processing_cost = 0
    energy_consumption = 0

    for node in fog_nodes + cloud_nodes:
        node.available_time = 0
        node.processing_cost = 0
        node.energy_consumed = 0

    for task_idx, node_idx in enumerate(solution):
        if node_idx < num_fog_nodes:
            node = fog_nodes[node_idx]
        else:
            node = cloud_nodes[node_idx - num_fog_nodes]

        execution_time = (float(tasks[task_idx].size) / node.processing_capacity) * 1000
        node.available_time += execution_time
        tasks[task_idx].response_time = node.available_time + node.delay
        node.processing_cost += (float(tasks[task_idx].size) / node.processing_capacity * node.processing_cost_coefficient)

    for node in fog_nodes + cloud_nodes:
        node.energy_consumed = (node.available_time / 1000.0) * node.max_power_consumption + ((makespan / 1000.0) - (node.available_time / 1000.0)) * node.min_power_consumption
        energy_consumption += node.energy_consumed
        processing_cost += node.processing_cost
        if node.available_time > makespan:
            makespan = node.available_time

    if energy_consumption == 0:
        energy_consumption = 1e-6
    if processing_cost == 0:
        processing_cost = 1e-6
    if makespan == 0:
        makespan = 1e-6

    fitness_value = 0.34 * min_energy_consumption / energy_consumption + 0.33 * min_processing_cost / processing_cost + 0.33 * min_makespan / makespan * 1000

    if return_all:
        return makespan, energy_consumption, processing_cost, fitness_value
    else:
        return fitness_value


def main():
    while True:
        num_tasks = int(input("\nEnter # of Tasks:\n"))
        num_fog_nodes = 10
        num_cloud_nodes = 5
        tasks_list = create_tasks(num_tasks)
        fog_nodes, cloud_nodes = create_nodes(num_fog_nodes, num_cloud_nodes)
        print("_________________________")
        print("\n1-Random")
        print("\n2-P2C")
        print("\n3-GA")
        print("\n4-EMA")
        print("\n5-Start New Iteration")
        print("\n6-Simulated Annealing Algorithm")
        print("\n7-GA + EMA")
        print("\n8-PSO Algorithm")
        print("\n9- E-HGA")
        print("\n10- Base PSGS")
        print("\n11- Hybrid PSGS + PSO")
        print("\n0 -Exit")
        while True:
            
            choice_code = int(input("\n\nEnter Your Choice: "))

            if choice_code == 1:
                random_task_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 2:
                performance_to_cost_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 3:
                genetic_algorithm_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 4:
                PSGS1_task_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 5:
                tasks_list = create_tasks(num_tasks)
                fog_nodes, cloud_nodes = create_nodes(num_fog_nodes, num_cloud_nodes)
            elif choice_code == 6:
                simulated_annealing_task_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 7:
                PSGS1_GA_task_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 8:
                particle_swarm_optimization_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 9:
                enhanced_hybrid_genetic_algorithm(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 10:
                base_psgs_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 11:
                hybrid_psgs_pso_scheduling(tasks_list, fog_nodes, cloud_nodes, num_tasks, num_fog_nodes, num_cloud_nodes)
            elif choice_code == 0:
            
                exit(0)


if __name__ == "__main__":
    main()
