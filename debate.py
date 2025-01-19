from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import asyncio
import time

from debateTypes import (
    TaskType,
    TaskInput,
    ArithmeticInput,
    GSM8KInput,
    ChessInput,
    BiographiesInput,
    MMLUInput,
    ChessValidityInput,
    GeneralInput,
    Task,
    Agent,
    GENERAL_TASK,
    CHESS_VALIDITY_TASK,
    BIOGRAPHIES_TASK,
    ARITHMETIC_TASK,
    GSM8K_TASK,
    CHESS_TASK,
    MMLU_TASK,
)

from generateInput import generate_arithmetic_input, get_random_gsm8k_question
import regex as re
import concurrent.futures


class RaftConsensus:
    """Manages RAFT-style leader election and consensus within debates."""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.current_leader = None
        self.agent_roles = {}
        self.majority_threshold = (
            (num_agents // 2) + 1 if num_agents % 2 != 0 else (num_agents // 2)
        )

    async def generate_roles(self, task_type: TaskType, agent: Agent) -> Dict[int, str]:
        """Generates appropriate roles for agents based on task type."""
        # Define default roles based on task type
        default_roles = {
            TaskType.ARITHMETIC: [
                "Mathematics Professor",
                "Data Scientist",
                "High School Math Teacher",
                "Local Town Idiot",
            ],
            TaskType.GSM8K: [
                "Mathematical Modeler",
                "IMO Gold Medalist",
                "Math Competition Coach",
                "Local Town Idiot",
            ],
            TaskType.CHESS: [
                "Chess Grandmaster",
                "Professional Chess Player",
                "Amateur Chess Enthusiast",
                "Local Town Idiot",
            ],
            TaskType.MMLU: [
                "Subject Matter Expert",
                "University Professor",
                "Graduate Student",
                "Local Town Idiot",
            ],
            TaskType.GENERAL: [
                "Domain Expert",
                "Research Analyst",
                "Brilliant Child Genius",
                "Local Town Idiot",
            ],
        }

        role_prompt = f"""Generate {self.num_agents} roles for a debate about a {task_type.value} problem.
        Roles should be relevant to the task and range from highly qualified to less qualified.
        Format your response as a simple comma-separated list like this:
        Mathematics Professor, High School Teacher, Student
        Your response should ONLY contain the comma-separated list, nothing else."""

        try:
            response = await agent.generate_response(role_prompt)

            if response:
                # Clean the response and split by commas
                roles_list = [
                    role.strip().strip('"').strip("'")
                    for role in response.strip().split(",")
                    if role.strip()
                ]

                if roles_list:
                    # If we got more roles than needed, take the first N
                    # If we got fewer roles than needed, we'll cycle through them
                    roles = {
                        i: roles_list[i % len(roles_list)]
                        for i in range(self.num_agents)
                    }
                    return roles

            # If parsing fails, use default roles
            base_roles = default_roles.get(task_type, default_roles[TaskType.GENERAL])
            roles = {i: base_roles[i % len(base_roles)] for i in range(self.num_agents)}
            return roles

        except Exception as e:
            print(f"Error in role generation: {e}")
            # Ultimate fallback
            return {i: f"Agent {i}" for i in range(self.num_agents)}

    async def initial_election(self, agent: Agent, agent_roles: Dict[int, str]) -> int:
        """Conducts initial leader election based on agent roles."""
        election_prompt = f"""Given these agent roles for a debate:
        {agent_roles}
        
        Each agent should vote for who they think should lead the debate based on qualifications.
        Generate {self.num_agents} votes (one for each agent). Format response as a Python list 
        of agent IDs that each agent votes for."""

        response = await agent.generate_response(election_prompt)
        try:
            # Extract list from response using regex or eval
            import re

            votes_str = re.search(r"\[.*\]", response).group()
            votes = eval(votes_str)

            # Count votes and determine leader
            from collections import Counter

            vote_counts = Counter(votes)
            leader_id = max(vote_counts.items(), key=lambda x: x[1])[0]

            return leader_id
        except Exception as e:
            print(f"Error in election: {e}")
            return 0  # Default to first agent as leader

    async def evaluate_leadership(
        self,
        agent: Agent,
        current_leader: int,
        round_responses: List[str],
        agent_roles: Dict[int, str],
    ) -> int:
        """Evaluates current leadership and determines if re-election is needed."""
        eval_prompt = f"""Given the current debate state:
        Current leader: Agent {current_leader} ({agent_roles[current_leader]})
        
        Round responses from agents:
        {round_responses}
        
        Agent roles:
        {agent_roles}
        
        Should leadership change? Consider:
        1. Quality and correctness of responses
        2. Leadership effectiveness
        3. Alternative qualified candidates

        Respond with either:
        - "maintain" to keep current leader
        - Or the agent ID (0-{self.num_agents-1}) of the new proposed leader"""

        response = await agent.generate_response(eval_prompt)

        if "maintain" in response.lower():
            return current_leader

        try:
            # Try to extract new leader ID
            import re

            new_leader = int(re.search(r"\d+", response).group())
            if 0 <= new_leader < self.num_agents:
                return new_leader
        except:
            pass

        return current_leader


class Debate:
    """Manages a multi-agent debate process with parallel agent responses within rounds."""

    def __init__(self, model_name: str, use_llama_api: bool = False):
        self.model_name = model_name
        self.use_llama_api = use_llama_api
        self.agents = []
        self.debate_history = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def _clean_response(self, response: str) -> str:
        """Cleans the model response to extract just the content."""
        # Split by "Output:" and take the last part
        if "Output:" in response:
            response = response.split("Output:")[-1]

        # Remove redundant "The final answer is:" statements except the last one
        parts = response.split("The final answer is:")
        if len(parts) > 1:
            response = "".join(parts[:-1]) + "The final answer is:" + parts[-1]

        # Clean up excessive newlines and spacing
        response = re.sub(r"\n\s*\n", "\n\n", response)
        return response.strip()

    async def _run_reflection(
        self, agent: Agent, response: str, reflection_prompt: str
    ) -> str:
        """Has an agent reflect on and potentially revise their response."""
        formatted_prompt = reflection_prompt.format(original_response=response)
        reflection = await agent.generate_response(formatted_prompt)
        return self._clean_response(reflection)

    async def _summarize_context(self, debate_round: List[str]) -> str:
        """Summarizes multiple agent responses into a single context."""
        summary_prompt = "Given these agent responses:\n\n"
        for i, resp in enumerate(debate_round, 1):
            summary_prompt += f"Agent {i}: {resp}\n"
        summary_prompt += "\nProvide a single synthesized summary."

        summarizer = self.agents[0] if self.agents else Agent(0, self.model_name)
        summary = await summarizer.generate_response(summary_prompt)
        return self._clean_response(summary)

    async def _get_parallel_responses(
        self, prompt: str, agents: List[Agent], leader_context: Optional[Dict] = None
    ) -> List[str]:
        """Gets responses from multiple agents in parallel for a single round."""
        tasks = []
        for agent in agents:
            # Add leader context to prompt if available
            if leader_context:
                agent_role = leader_context["roles"][agent.agent_id]
                is_leader = agent.agent_id == leader_context["leader"]
                contextualized_prompt = f"""You are acting as: {agent_role}
                {' (Current Debate Leader)' if is_leader else ''}
                
                Current debate leader is: {leader_context['roles'][leader_context['leader']]}
                
                {prompt}"""
            else:
                contextualized_prompt = prompt

            task = asyncio.create_task(agent.generate_response(contextualized_prompt))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return responses

    async def run_debate(
        self,
        task: Task,
        task_input: TaskInput,
        num_rounds: int,
        num_agents: int,
        summarize_context: bool = False,
        use_reflection: bool = False,
        raft: bool = False,
        summarize_final_answer: bool = False,
    ) -> List[List[str]]:
        """Runs a debate for specified number of rounds with given number of agents."""
        # Initialize agents
        self.agents = [
            Agent(i, self.model_name, use_llama_api=self.use_llama_api)
            for i in range(num_agents)
        ]
        self.debate_history = []

        # Initialize RAFT consensus if enabled
        raft_consensus = None
        leader_context = None

        if raft:
            raft_consensus = RaftConsensus(num_agents)
            # Generate roles using first agent as coordinator
            agent_roles = await raft_consensus.generate_roles(
                task.task_type, self.agents[0]
            )
            # Conduct initial leader election
            leader_id = await raft_consensus.initial_election(
                self.agents[0], agent_roles
            )
            leader_context = {"leader": leader_id, "roles": agent_roles}

        # Run all rounds
        for round_num in range(num_rounds):
            # Format prompt based on round number
            if round_num == 0:
                prompt = task.format_prompt(task_input)
            else:
                context = (
                    await self._summarize_context(self.debate_history[-1])
                    if summarize_context
                    else "\n".join(self.debate_history[-1])
                )
                prompt = task.debate_prompt.format(context=context)

            # Get responses for this round
            responses = await self._get_parallel_responses(
                prompt, self.agents, leader_context
            )
            cleaned_responses = [
                self._clean_response(response) for response in responses
            ]

            # Run reflection if enabled
            if use_reflection:
                reflection_tasks = []
                for i, response in enumerate(cleaned_responses):
                    reflection_task = asyncio.create_task(
                        self._run_reflection(
                            self.agents[i], response, task.reflection_prompt
                        )
                    )
                    reflection_tasks.append(reflection_task)
                reflected_responses = await asyncio.gather(*reflection_tasks)
                cleaned_responses = [
                    self._clean_response(response) for response in reflected_responses
                ]

            self.debate_history.append(cleaned_responses)

            # Evaluate leadership after each round if RAFT is enabled
            if (
                raft and raft_consensus and round_num < num_rounds - 1
            ):  # Skip last round
                new_leader = await raft_consensus.evaluate_leadership(
                    self.agents[0],  # Use first agent as coordinator
                    leader_context["leader"],
                    cleaned_responses,
                    leader_context["roles"],
                )
                leader_context["leader"] = new_leader

        # Optionally summarize final round responses
        if summarize_final_answer:
            final_summary = await self._summarize_context(self.debate_history[-1])
            self.debate_history.append([final_summary])

        return self.debate_history


def print_debate_results(responses):
    """
    Prints debate responses in a clear format showing rounds and agents.

    Args:
        responses: List[List[str]] - The debate responses where:
            - First dimension is rounds
            - Second dimension is agents
            - Last round might be summary if summarize_final_answer=True
    """
    print("\n===== DEBATE RESULTS =====\n")

    # Print all rounds except possibly the last (if it's a summary)
    for round_num, round_responses in enumerate(responses[:-1], 1):
        print(f"\n----- ROUND {round_num} -----")
        for agent_num, response in enumerate(round_responses, 1):
            print(f"\nAGENT {agent_num}:")
            print("-" * 40)
            print(response.strip())
            print("-" * 40)

    # Check if last round is a summary (will have only one response)
    if len(responses[-1]) == 1:
        print("\n===== FINAL SUMMARY =====")
        print("-" * 40)
        print(responses[-1][0].strip())
        print("-" * 40)
    else:
        # Last round was a normal round
        print(f"\n----- ROUND {len(responses)} -----")
        for agent_num, response in enumerate(responses[-1], 1):
            print(f"\nAGENT {agent_num}:")
            print("-" * 40)
            print(response.strip())
            print("-" * 40)

    print("\n===== END OF DEBATE =====\n")


# Example usage
async def main():

    # Example 1: Arithmetic task with individual responses
    # result, int_list = generate_arithmetic_input()
    # print("check int list:", int_list, "result:", result)

    # arithmetic_input = ArithmeticInput(numbers=int_list)
    debate = Debate(model_name="llama70b", use_llama_api=True)

    # start_time = time.time()

    # arithmetic_responses = await debate.run_debate(
    #     task=ARITHMETIC_TASK,
    #     task_input=arithmetic_input,
    #     num_rounds=1,
    #     num_agents=4,
    #     summarize_context=False,
    #     use_reflection=True,
    #     summarize_final_answer=True,
    # )
    # print_debate_results(arithmetic_responses)

    # duration = time.time() - start_time
    # print(f"Execution time: {duration:.2f} seconds")

    # # Example 2: Arithmetic task with summarized final answer
    # arithmetic_summary = await debate.run_debate(
    #     task=ARITHMETIC_TASK,
    #     task_input=arithmetic_input,
    #     num_rounds=3,
    #     num_agents=2,
    #     summarize_context=True,
    #     use_reflection=True,
    #     summarize_final_answer=True,
    # )
    # print("Summarized arithmetic response:", arithmetic_summary[0])

    # Example 3: General task with arbitrary prompt
    # general_input = GeneralInput(
    #     prompt="What are the potential implications of quantum computing on cryptography?"
    # )

    general_input = GeneralInput(
        prompt="""
            what is the total number of characters here across val test train in all categories. Give one final number representing the total sum.
            --------------------------------------------------------------------------------
            Category                       Vocab Size        Train        Val       Test
            --------------------------------------------------------------------------------
            Arts and Literature.txt             1,439    5,529,600    307,200    307,200
            Biographies.txt                     1,995    8,013,600    445,200    445,200
            Culture and Society.txt             1,434    3,931,200    218,400    218,400
            Economics and Business.txt          1,066    3,110,400    172,800    172,800
            Education and Academia.txt            736    1,209,600     67,200     67,200
            Entertainment and Media.txt         1,316    5,354,848    297,492    297,492
            Geography.txt                       1,727    5,875,200    326,400    326,400
            Health and Medicine.txt             1,246    3,196,800    177,600    177,600
            History.txt                         1,653    6,501,600    361,200    361,200
            Language and Linguistics.txt        2,670    3,304,800    183,600    183,600
            Military and Warfare.txt            1,194    3,132,000    174,000    174,000
            Politics and Government.txt         1,978    9,136,800    507,600    507,600
            Religion and Philosophy.txt         1,963    8,380,800    465,600    465,600
            Science.txt                         2,442   11,318,400    628,800    628,800
            Sports and Recreation.txt           1,145    3,952,800    219,600    219,600
            Technology.txt                      1,763    7,711,200    428,400    428,400
        """
    )

    start_time = time.time()
    general_responses = await debate.run_debate(
        task=GENERAL_TASK,
        task_input=general_input,
        num_rounds=5,
        num_agents=5,
        summarize_context=True,
        use_reflection=False,
        raft=False,
        summarize_final_answer=True,
    )
    print_debate_results(general_responses)

    duration = time.time() - start_time
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
