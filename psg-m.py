import random
import copy
from dataclasses import dataclass
from typing import List, Dict
from operator import attrgetter

# Define the Fog Node class
@dataclass
class FogNode:
    id: int
    cpu_speed: float  # KB/s processing speed
    energy_rate: float  # Energy consumption rate per unit time
    active_time: float = 0.0
    idle_time: float = 0.0
    current_load: int = 0  # Number of active tasks
    max_load: int = 5      # Maximum tasks a node can handle

    def update_times(self, time_taken: float):
        self.active_time += time_taken

# Define the Task class
@dataclass
class Task:
    id: int
    deadline: float       # Deadline in seconds
    arrival_time: float   # Arrival time in seconds
    input_size: float     # Input file size in KB
    priority: float = 0.0 # Dynamic priority score

    def calculate_priority(self, w1=0.5, w2=0.3, w3=0.2):
        # Lower value means higher priority.
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

# PSGS Scheduler class (implements the base algorithm)
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
                    load_factor = 1 - (node.current_load / node.max_load)  # Prefer less loaded nodes.
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
        # Deadline satisfied: tasks not offloaded to the cloud.
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

# PSG-M (Multistart) function: run multiple independent iterations and select the best schedule.
def run_psg_multistart(num_starts: int, fog_nodes: List[FogNode], tasks: List[Task]):
    best_schedule = None
    best_metrics = None
    best_run = -1
    best_score = -float('inf')
    for i in range(num_starts):
        # Use deepcopy to ensure each run has independent initializations.
        nodes_copy = copy.deepcopy(fog_nodes)
        tasks_copy = copy.deepcopy(tasks)
        scheduler = PSGSScheduler(nodes_copy)
        schedule = scheduler.schedule_tasks(tasks_copy)
        metrics = scheduler.calculate_metrics(tasks_copy)
        # Score criteria: primarily the Deadline Satisfaction Percentage.
        score = metrics["Deadline Satisfaction Percentage"]
        # If tie: use lower energy consumption.
        if best_schedule is None or (score > best_score or (score == best_score and metrics["Total Energy Consumption"] < best_metrics["Total Energy Consumption"])):
            best_schedule = schedule
            best_metrics = metrics
            best_run = i
            best_score = score
    return best_schedule, best_metrics, best_run

# Example usage with random initialization for fog nodes and tasks.
def main():
    # Initialize fog nodes with varied performance.
    num_nodes = 5
    fog_nodes = []
    for i in range(1, num_nodes + 1):
        cpu_speed = random.randint(10, 30)  # CPU speed between 10 and 30 KB/s.
        energy_rate = round(random.uniform(0.5, 1.0), 2)  # Energy rate between 0.5 and 1.0.
        fog_nodes.append(FogNode(id=i, cpu_speed=cpu_speed, energy_rate=energy_rate))
    
    # Initialize tasks with varied deadlines and arrival times.
    num_tasks = 10
    tasks = []
    for i in range(1, num_tasks + 1):
        deadline = round(random.uniform(5, 15), 2)       # Deadline between 5 and 15 seconds.
        arrival_time = round(random.uniform(0.1, 3.0), 2)  # Arrival time between 0.1 and 3.0 seconds.
        input_size = random.randint(100, 300)              # Input size between 100 and 300 KB.
        tasks.append(Task(id=i, deadline=deadline, arrival_time=arrival_time, input_size=input_size))
    
    num_starts = 5  # Number of independent runs for the multistart approach.
    best_schedule, best_metrics, best_run = run_psg_multistart(num_starts, fog_nodes, tasks)
    
    print(f"Best Schedule Found in Run: {best_run}")
    print("Task Schedule:")
    for task_id, assignment in best_schedule.items():
        print(f"  Task {task_id} -> {assignment}")
    print("\nPerformance Metrics:")
    print(f"  Deadline Satisfied Tasks: {best_metrics['Deadline Satisfied Count']} out of {num_tasks}")
    print(f"  Deadline Satisfaction Percentage: {best_metrics['Deadline Satisfaction Percentage']:.2f}%")
    print(f"  Total Energy Consumption: {best_metrics['Total Energy Consumption']:.4f}")
    print(f"  Makespan: {best_metrics['Makespan']:.4f}")

if __name__ == "__main__":
    main()