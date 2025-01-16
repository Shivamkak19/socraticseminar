from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import asyncio
import random
from enum import Enum
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


class Debate:
    """Manages a multi-agent debate process with parallel agent responses within rounds."""

    def __init__(self, model_name: str, use_llama_api: bool = False):
        self.model_name = model_name
        self.use_llama_api = use_llama_api
        self.agents = []
        self.debate_history = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    async def _get_parallel_responses(
        self, prompt: str, agents: List[Agent]
    ) -> List[str]:
        """Gets responses from multiple agents in parallel for a single round."""
        # Create tasks for each agent's response
        tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.generate_response(prompt))
            tasks.append(task)

        # Wait for all responses
        responses = await asyncio.gather(*tasks)
        return responses

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

    async def run_debate(
        self,
        task: Task,
        task_input: TaskInput,
        num_rounds: int,
        num_agents: int,
        summarize_context: bool = False,
        use_reflection: bool = False,
        summarize_final_answer: bool = False,
    ) -> List[List[str]]:
        """Runs a debate for specified number of rounds with given number of agents."""
        # Initialize agents
        self.agents = [
            Agent(i, self.model_name, use_llama_api=self.use_llama_api)
            for i in range(num_agents)
        ]
        self.debate_history = []

        # Run all rounds including the first one
        for round_num in range(num_rounds):
            # Format prompt based on round number
            if round_num == 0:
                # First round uses the initial prompt
                prompt = task.format_prompt(task_input)
            else:
                # Get context from previous round
                context = (
                    await self._summarize_context(self.debate_history[-1])
                    if summarize_context
                    else "\n".join(self.debate_history[-1])
                )
                # Subsequent rounds use the debate prompt
                prompt = task.debate_prompt.format(
                    context=context, round_num=round_num + 1
                )

            # Get responses for this round
            responses = await self._get_parallel_responses(prompt, self.agents)
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
        summarize_final_answer=True,
    )
    print_debate_results(general_responses)

    duration = time.time() - start_time
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
