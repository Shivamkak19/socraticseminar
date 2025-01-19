import asyncio
from typing import List, Tuple, Dict, Any
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from debate import Debate
from debateTypes import (
    TaskType,
    ARITHMETIC_TASK,
    GSM8K_TASK,
    CHESS_TASK,
    MMLU_TASK,
    ArithmeticInput,
    Agent,
    GSM8KInput,
    ChessInput,
    MMLUInput,
)
from generateInput import (
    generate_arithmetic_input,
    get_random_gsm8k_question,
    get_random_chess_position,
    get_random_mmlu_question,
)


class DebateEvaluator:
    """Framework for evaluating debate performance across different tasks and configurations."""

    def __init__(self, model_name: str = "llama70b", use_llama_api: bool = True):
        self.model_name = model_name
        self.use_llama_api = use_llama_api

    async def evaluate_arithmetic(
        self,
        num_samples: int,
        num_agents_range: List[int],
        num_rounds_range: List[int],
    ) -> Dict:
        """Evaluate arithmetic task performance."""
        results = {
            "standard": {"agent_accuracies": [], "round_accuracies": []},
            "raft": {"agent_accuracies": [], "round_accuracies": []},
        }

        # Test both with and without RAFT
        for use_raft in [False, True]:
            mode = "raft" if use_raft else "standard"

            for num_agents in tqdm(
                num_agents_range, desc=f"Testing agent counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    # Generate random arithmetic problem
                    true_result, numbers = generate_arithmetic_input()
                    task_input = ArithmeticInput(numbers=numbers)

                    # Run debate with max rounds
                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=ARITHMETIC_TASK,
                        task_input=task_input,
                        num_rounds=min(num_rounds_range),
                        num_agents=num_agents,
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    # Extract final answer and check correctness
                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.ARITHMETIC,
                        responses[-1][0],  # Final debate response
                        true_result,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["agent_accuracies"].append(accuracy)

            # Similar evaluation across different round counts
            for num_rounds in tqdm(
                num_rounds_range, desc=f"Testing round counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    true_result, numbers = generate_arithmetic_input()
                    task_input = ArithmeticInput(numbers=numbers)

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=ARITHMETIC_TASK,
                        task_input=task_input,
                        num_rounds=num_rounds,
                        num_agents=min(num_agents_range),
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.ARITHMETIC,
                        responses[-1][0],  # Final debate response
                        true_result,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["round_accuracies"].append(accuracy)

        return results

    async def evaluate_gsm8k(
        self,
        num_samples: int,
        num_agents_range: List[int],
        num_rounds_range: List[int],
    ) -> Dict:
        """Evaluate GSM8K task performance."""
        results = {
            "standard": {"agent_accuracies": [], "round_accuracies": []},
            "raft": {"agent_accuracies": [], "round_accuracies": []},
        }

        # Test both with and without RAFT
        for use_raft in [False, True]:
            mode = "raft" if use_raft else "standard"

            for num_agents in tqdm(
                num_agents_range, desc=f"Testing agent counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    question, true_answer = get_random_gsm8k_question()
                    task_input = GSM8KInput(problem=question)

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=GSM8K_TASK,
                        task_input=task_input,
                        num_rounds=min(num_rounds_range),
                        num_agents=num_agents,
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.GSM8K,
                        responses[-1][0],  # Final debate response
                        true_answer,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["agent_accuracies"].append(accuracy)

            # Similar evaluation for rounds
            for num_rounds in tqdm(
                num_rounds_range, desc=f"Testing round counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    question, true_answer = get_random_gsm8k_question()
                    task_input = GSM8KInput(problem=question)

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=GSM8K_TASK,
                        task_input=task_input,
                        num_rounds=num_rounds,
                        num_agents=min(num_agents_range),
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.GSM8K,
                        responses[-1][0],  # Final debate response
                        true_answer,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["round_accuracies"].append(accuracy)

        return results

    async def evaluate_chess(
        self,
        num_samples: int,
        num_agents_range: List[int],
        num_rounds_range: List[int],
    ) -> Dict:
        """Evaluate chess position task performance."""
        results = {
            "standard": {"agent_accuracies": [], "round_accuracies": []},
            "raft": {"agent_accuracies": [], "round_accuracies": []},
        }

        # Test both with and without RAFT
        for use_raft in [False, True]:
            mode = "raft" if use_raft else "standard"

            for num_agents in tqdm(
                num_agents_range, desc=f"Testing agent counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    position, valid_moves = get_random_chess_position()
                    task_input = ChessInput(moves=position)

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=CHESS_TASK,
                        task_input=task_input,
                        num_rounds=min(num_rounds_range),
                        num_agents=num_agents,
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.CHESS,
                        responses[-1][0],  # Final debate response
                        valid_moves,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["agent_accuracies"].append(accuracy)

            # Similar evaluation for rounds
            for num_rounds in tqdm(
                num_rounds_range, desc=f"Testing round counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    position, valid_moves = get_random_chess_position()
                    task_input = ChessInput(moves=position)

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=CHESS_TASK,
                        task_input=task_input,
                        num_rounds=num_rounds,
                        num_agents=min(num_agents_range),
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.CHESS,
                        responses[-1][0],  # Final debate response
                        valid_moves,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["round_accuracies"].append(accuracy)

        return results

    async def evaluate_mmlu(
        self,
        num_samples: int,
        num_agents_range: List[int],
        num_rounds_range: List[int],
    ) -> Dict:
        """Evaluate MMLU task performance."""
        results = {
            "standard": {"agent_accuracies": [], "round_accuracies": []},
            "raft": {"agent_accuracies": [], "round_accuracies": []},
        }

        # Test both with and without RAFT
        for use_raft in [False, True]:
            mode = "raft" if use_raft else "standard"

            for num_agents in tqdm(
                num_agents_range, desc=f"Testing agent counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    mmlu_input, true_answer = get_random_mmlu_question()

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=MMLU_TASK,
                        task_input=mmlu_input,
                        num_rounds=min(num_rounds_range),
                        num_agents=num_agents,
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.MMLU,
                        responses[-1][0],  # Final debate response
                        true_answer,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["agent_accuracies"].append(accuracy)

            # Similar evaluation for rounds
            for num_rounds in tqdm(
                num_rounds_range, desc=f"Testing round counts ({mode})"
            ):
                correct = 0
                for _ in range(num_samples):
                    mmlu_input, true_answer = get_random_mmlu_question()

                    debate = Debate(self.model_name, self.use_llama_api)
                    responses = await debate.run_debate(
                        task=MMLU_TASK,
                        task_input=mmlu_input,
                        num_rounds=num_rounds,
                        num_agents=min(num_agents_range),
                        raft=use_raft,
                        summarize_final_answer=True,
                    )

                    is_correct = await self._validate_answer_with_gpt(
                        TaskType.MMLU,
                        responses[-1][0],  # Final debate response
                        true_answer,
                    )

                    if is_correct:
                        correct += 1

                accuracy = correct / num_samples
                results[mode]["round_accuracies"].append(accuracy)

        return results

    async def _validate_answer_with_gpt(
        self, task_type: TaskType, proposed_answer: str, true_answer: Any
    ) -> bool:
        """
        Validates an answer using GPT by comparing proposed answer to true answer.
        Returns True if GPT determines the answer is correct, False otherwise.

        Args:
            task_type: Type of task being evaluated
            proposed_answer: The answer string from the debate
            true_answer: The known correct answer
        """
        # Create a prompt based on task type
        task_prompts = {
            TaskType.ARITHMETIC: f"""Determine if these answers are equivalent:
    Proposed answer: {proposed_answer}
    Correct answer: {true_answer}
    Answer TRUE if they are equivalent, FALSE if not. Reply with only TRUE or FALSE.""",
            TaskType.GSM8K: f"""Determine if these answers are equivalent:
    Proposed answer: {proposed_answer}
    Correct answer: {true_answer}
    The answers should be considered equivalent if they represent the same numerical value, ignoring formatting.
    Answer TRUE if they are equivalent, FALSE if not. Reply with only TRUE or FALSE.""",
            TaskType.CHESS: f"""Determine if the proposed chess move destination square is in the list of valid moves:
    Proposed move destination: {proposed_answer}
    Valid moves: {true_answer}
    Answer TRUE if the proposed move is in the list, FALSE if not. Reply with only TRUE or FALSE.""",
            TaskType.MMLU: f"""Determine if these multiple choice answers are equivalent:
    Proposed answer: {proposed_answer}
    Correct answer: {true_answer}
    Answer TRUE if they match exactly (ignoring case), FALSE if not. Reply with only TRUE or FALSE.""",
        }

        prompt = task_prompts.get(task_type)
        if not prompt:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Create temporary agent for validation
        validator = Agent(0, self.model_name, use_llama_api=self.use_llama_api)

        try:
            # Get validation response
            response = await validator.generate_response(prompt)

            # Parse response - look for TRUE/true only
            return "true" in response.lower().strip()
        except Exception as e:
            print(f"Error in GPT validation: {e}")
            return False

    def _extract_arithmetic_answer(self, response: str) -> int:
        """Extract numerical answer from arithmetic response.

        Handles various response formats like:
        - "The answer is 42"
        - "42"
        - "... final answer is 42"
        - "... = 42"
        """
        try:
            # Look for patterns like "answer is X", "= X", or just "X"
            import re

            # Try to find a number after specific keywords first
            patterns = [
                r"answer\s+is\s+(-?\d+)",  # "answer is X"
                r"result\s+is\s+(-?\d+)",  # "result is X"
                r"=\s*(-?\d+)\s*$",  # "= X" at end
                r"(-?\d+)\s*$",  # just "X" at end
            ]

            for pattern in patterns:
                match = re.search(pattern, response.lower())
                if match:
                    return int(match.group(1))

            # If no match found with specific patterns,
            # get all numbers and take the last one
            numbers = re.findall(r"-?\d+", response)
            if numbers:
                return int(numbers[-1])

            return None
        except Exception as e:
            print(f"Error extracting arithmetic answer: {e}")
            return None

    def _extract_gsm8k_answer(self, response: str) -> int:
        """Extract numerical answer from GSM8K response, handling commas in numbers."""
        try:
            # Look for answer in \boxed{X} format
            import re

            # First try to find boxed answer
            pattern = r"\\boxed{([0-9,]+)}"
            match = re.search(pattern, response)

            if match:
                # Remove commas and convert to integer
                answer_str = match.group(1).replace(",", "")
                return int(answer_str)

            # Fallback: look for the last number in the response
            # This pattern matches numbers that may include commas
            number_pattern = r"(?:^|[^\d,])([0-9,]+)(?:$|[^\d,])"
            matches = re.finditer(number_pattern, response)

            # Get the last match
            last_number = None
            for match in matches:
                last_number = match.group(1)

            if last_number:
                # Remove commas and convert to integer
                return int(last_number.replace(",", ""))

            return None

        except Exception as e:
            print(f"Error extracting GSM8K answer: {e}")
            return None

    def _extract_chess_move(self, response: str) -> str:
        """Extract chess move from response, returning only the destination square."""
        try:
            import re

            # First find the full move
            patterns = [
                r"14\.\s*([KQRBN]?[a-h][1-8])",  # With move number
                r"([KQRBN]?[a-h][1-8])\s*$",  # Without move number
            ]

            full_move = None
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    full_move = match.group(1)
                    break

            if full_move:
                # Strip any piece notation (K,Q,R,B,N) to get just the square
                dest_square = re.sub(r"^[KQRBN]", "", full_move)
                return dest_square

            return None

        except Exception as e:
            print(f"Error extracting chess move: {e}")
            return None

    def _extract_mmlu_answer(self, response: str) -> str:
        """Extract MMLU answer choice (A/B/C/D) from response."""
        try:
            import re

            # Look for answers in format (X) where X is A, B, C, or D
            pattern = r"\(([A-D])\)"
            matches = re.findall(pattern, response)

            if matches:
                # Return the last matched answer (in case of discussion of multiple options)
                return matches[-1].upper()

            return None

        except Exception as e:
            print(f"Error extracting MMLU answer: {e}")
            return None

    def plot_results(
        self,
        results: Dict,
        num_agents_range: List[int],
        num_rounds_range: List[int],
        task_name: str,
    ):

        print("RESULTS:", results)
        """Plot and save accuracy results comparing standard and RAFT approaches."""
        os.makedirs("debate_results", exist_ok=True)

        # Plot accuracy vs number of agents
        plt.figure(figsize=(10, 6))
        plt.plot(
            num_agents_range,
            results["standard"]["agent_accuracies"],
            "b-o",
            label="Standard",
        )
        plt.plot(
            num_agents_range, results["raft"]["agent_accuracies"], "r-o", label="RAFT"
        )
        plt.xlabel("Number of Agents")
        plt.ylabel("Accuracy")
        plt.title(f"{task_name} Accuracy vs Number of Agents")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"debate_results/{task_name.lower()}_agents_accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot accuracy vs number of rounds
        plt.figure(figsize=(10, 6))
        plt.plot(
            num_rounds_range,
            results["standard"]["round_accuracies"],
            "b-o",
            label="Standard",
        )
        plt.plot(
            num_rounds_range, results["raft"]["round_accuracies"], "r-o", label="RAFT"
        )
        plt.xlabel("Number of Rounds")
        plt.ylabel("Accuracy")
        plt.title(f"{task_name} Accuracy vs Number of Rounds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"debate_results/{task_name.lower()}_rounds_accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save numerical results
        results_data = {
            "num_agents_range": num_agents_range,
            "num_rounds_range": num_rounds_range,
            "standard": {
                "agent_accuracies": results["standard"]["agent_accuracies"],
                "round_accuracies": results["standard"]["round_accuracies"],
            },
            "raft": {
                "agent_accuracies": results["raft"]["agent_accuracies"],
                "round_accuracies": results["raft"]["round_accuracies"],
            },
        }
        with open(
            f"debate_results/{task_name.lower()}_results_comparison.json", "w"
        ) as f:
            json.dump(results_data, f, indent=2)


async def main():
    # Initialize evaluator
    evaluator = DebateEvaluator()

    # Define evaluation parameters
    num_samples = 5
    # num_agents_range = [1]
    # num_rounds_range = [1, 2]
    num_agents_range = [1, 2, 3, 4, 5]
    num_rounds_range = [1, 2, 3, 4, 5]

    # Run evaluations for each task
    tasks = ["Arithmetic", "GSM8K", "Chess", "MMLU"]

    for task in tasks:
        print(f"\nEvaluating {task} task...")

        if task == "Arithmetic":
            results = await evaluator.evaluate_arithmetic(
                num_samples, num_agents_range, num_rounds_range
            )
        elif task == "GSM8K":
            results = await evaluator.evaluate_gsm8k(
                num_samples, num_agents_range, num_rounds_range
            )
        elif task == "Chess":
            results = await evaluator.evaluate_chess(
                num_samples, num_agents_range, num_rounds_range
            )
        else:  # MMLU
            results = await evaluator.evaluate_mmlu(
                num_samples, num_agents_range, num_rounds_range
            )

        # Plot comparative results
        evaluator.plot_results(results, num_agents_range, num_rounds_range, task)


if __name__ == "__main__":
    asyncio.run(main())
