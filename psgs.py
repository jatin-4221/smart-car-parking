import random
from dataclasses import dataclass
from typing import List, Dict
from operator import attrgetter

# Define the Fog Node class
@dataclass
class FogNode:
    id: int
    cpu_speed: float  # KB/s processing speed
    energy_rate: float  # Energy consumption rate (energy units per time unit)
    active_time: float = 0.0
    idle_time: float = 0.0
    current_load: int = 0  # Number of active tasks
    max_load: int = 5      # Maximum tasks a node can handle

    def update_times(self, time_taken: float):
        self.active_time += time_taken
        # Idle time will be computed later based on overall makespan

# Define the Task class
@dataclass
class Task:
    id: int
    deadline: float       # Deadline in seconds
    arrival_time: float   # Arrival time in seconds
    input_size: float     # Input file size in KB
    priority: float = 0.0 # Dynamic priority score

    def calculate_priority(self, w1=0.5, w2=0.3, w3=0.2):
        # Calculate dynamic priority: lower value means higher priority.
        # Here, input_size is normalized by 1000.
        self.priority = -(w1 * self.deadline + w2 * self.arrival_time + w3 * (self.input_size / 1000))

# Function to calculate the response time (RT) of a task on a given fog node.
def calculate_rt(task: Task, node: FogNode, network_delay: float = 0.01, queue_delay: float = 0.05) -> float:
    transmission_time = task.input_size / 1000  # Assume a fixed bandwidth of 1000 KB/s.
    processing_time = task.input_size / node.cpu_speed
    return transmission_time + processing_time + network_delay + queue_delay

# Function to calculate the energy consumption for a node.
def calculate_econ(node: FogNode, task_assignment_time: float) -> float:
    idle_energy_rate = 0.1  # Constant idle energy rate.
    return (task_assignment_time * node.energy_rate * node.active_time +
            node.idle_time * idle_energy_rate)

# PSGS Scheduler class with weighted random selection of nodes.
class PSGSScheduler:
    def __init__(self, fog_nodes: List[FogNode]):
        self.fog_nodes = fog_nodes
        self.schedule = {}      # Maps task IDs to fog node IDs or "cloud"
        self.cloud_tasks = []   # List of task IDs offloaded to the cloud

    def schedule_tasks(self, tasks: List[Task]) -> Dict:
        # Calculate dynamic priorities and sort tasks (ascending order: higher priority first).
        for task in tasks:
            task.calculate_priority()
        tasks.sort(key=attrgetter('priority'))

        # Evaluate and schedule each task.
        for task in tasks:
            dsl = []  # Deadline Satisfied List: nodes that can complete the task before its deadline.
            for node in self.fog_nodes:
                rt = calculate_rt(task, node)
                if rt < task.deadline and node.current_load < node.max_load:
                    dsl.append((node, rt))

            if dsl:
                # Calculate selection score based on energy consumption and load.
                prl = []  # Priority Resource List: tuples (node, score, response time)
                for node, rt in dsl:
                    econ = calculate_econ(node, task_assignment_time=0.1)
                    load_factor = 1 - (node.current_load / node.max_load)  # Prefer nodes with lower load.
                    score = 1 / (econ + 0.01) * load_factor  # Higher score is better.
                    prl.append((node, score, rt))

                total_score = sum(score for _, score, _ in prl)
                weights = [score / total_score for _, score, _ in prl]
                selected_node = random.choices([n for n, _, _ in prl], weights=weights, k=1)[0]
                
                # Assign the task to the selected node.
                self.schedule[task.id] = f"Fog Node {selected_node.id}"
                selected_node.current_load += 1
                selected_node.update_times(calculate_rt(task, selected_node))
            else:
                # Offload task to cloud if no fog node can meet the deadline.
                self.schedule[task.id] = "cloud"
                self.cloud_tasks.append(task.id)

        return self.schedule

    def calculate_metrics(self, tasks: List[Task]) -> Dict:
        # Calculate Deadline Satisfied metrics: tasks not offloaded to the cloud.
        deadline_satisfied_count = sum(1 for task in tasks if self.schedule.get(task.id) != "cloud")
        ds_percent = (deadline_satisfied_count / len(tasks)) * 100

        # Compute the makespan: maximum active time among all fog nodes.
        makespan = max(node.active_time for node in self.fog_nodes)
        # Update each node's idle time based on the makespan.
        for node in self.fog_nodes:
            node.idle_time = makespan - node.active_time

        # Total energy consumption aggregated over all fog nodes.
        total_econ = sum(calculate_econ(node, 0.1) for node in self.fog_nodes)

        return {
            "Deadline Satisfied Count": deadline_satisfied_count,
            "Deadline Satisfaction Percentage": ds_percent,
            "Total Energy Consumption": total_econ,
            "Makespan": makespan
        }

# Example usage with random initialization for fog nodes and tasks.
def main():
    # Randomly initialize a list of fog nodes.
    # Adjusting CPU speeds to a higher range to reduce processing times.
    num_nodes = 5
    fog_nodes = []
    for i in range(1, num_nodes + 1):
        cpu_speed = random.randint(20, 40)  # CPU speed between 20 and 40 KB/s.
        energy_rate = round(random.uniform(0.5, 1.0), 2)  # Energy rate between 0.5 and 1.0.
        fog_nodes.append(FogNode(id=i, cpu_speed=cpu_speed, energy_rate=energy_rate))
    
    # Randomly initialize a list of tasks.
    # Adjust deadlines to be higher (to allow processing) and input_size to a lower range.
    num_tasks = 10
    tasks = []
    for i in range(1, num_tasks + 1):
        deadline = round(random.uniform(15, 30), 2)       # Deadline between 15 and 30 seconds.
        arrival_time = round(random.uniform(0.1, 5.0), 2)   # Arrival time between 0.1 and 5.0 seconds.
        input_size = random.randint(50, 150)                # Input size between 50 and 150 KB.
        tasks.append(Task(id=i, deadline=deadline, arrival_time=arrival_time, input_size=input_size))
    
    # Initialize the scheduler and schedule the tasks.
    scheduler = PSGSScheduler(fog_nodes)
    schedule = scheduler.schedule_tasks(tasks)
    
    # Print the task assignment results.
    print("Task Schedule:")
    for task_id, assignment in schedule.items():
        print(f"  Task {task_id} -> {assignment}")
    print("\nOffloaded to Cloud:", scheduler.cloud_tasks)
    
    # Calculate and display performance metrics.
    metrics = scheduler.calculate_metrics(tasks)
    print("\nPerformance Metrics:")
    print(f"  Deadline Satisfied Tasks: {metrics['Deadline Satisfied Count']} out of {len(tasks)}")
    print(f"  Deadline Satisfaction Percentage: {metrics['Deadline Satisfaction Percentage']:.2f}%")
    print(f"  Total Energy Consumption: {metrics['Total Energy Consumption']:.4f}")
    print(f"  Makespan: {metrics['Makespan']:.4f}")

if __name__ == "__main__":
    main()

