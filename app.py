from flask import Flask, jsonify, request
from debateTypes import GeneralInput, GENERAL_TASK, Agent
from debate import Debate
import asyncio
import threading
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store active debates
active_debates = {}


class DebateManager:
    def __init__(self):
        self.debate = None
        self.debate_history = []
        self.is_complete = False
        self.final_answer = None
        self.current_round = 0


def run_debate_async(
    debate_id,
    prompt,
    num_rounds,
    num_agents,
    summarize_context,
    use_reflection,
    summarize_final_answer=True,
):
    """Run debate asynchronously and store results."""

    async def run():
        try:
            # Create new debate instance
            debate = Debate(model_name="llama70b", use_llama_api=True)

            # Store debate instance
            active_debates[debate_id].debate = debate

            # Initialize debate parameters
            task = GENERAL_TASK
            augment_prompt = prompt + "Keep answer short, sweet, to the point."
            task_input = GeneralInput(prompt=augment_prompt)

            # Initialize agents
            debate.agents = [
                Agent(i, debate.model_name, use_llama_api=True)
                for i in range(num_agents)
            ]

            # Run debate round by round
            for round_num in range(num_rounds):
                # Format prompt based on round number
                if round_num == 0:
                    current_prompt = task.format_prompt(task_input)
                else:
                    context = (
                        await debate._summarize_context(
                            active_debates[debate_id].debate_history[-1]
                        )
                        if summarize_context
                        else "\n".join(active_debates[debate_id].debate_history[-1])
                    )
                    current_prompt = task.debate_prompt.format(
                        context=context, round_num=round_num + 1
                    )

                # Get responses for this round
                responses = await debate._get_parallel_responses(
                    current_prompt, debate.agents
                )
                cleaned_responses = [
                    debate._clean_response(response) for response in responses
                ]

                # Run reflection if enabled
                if use_reflection:
                    reflection_tasks = []
                    for i, response in enumerate(cleaned_responses):
                        reflection_task = asyncio.create_task(
                            debate._run_reflection(
                                debate.agents[i], response, task.reflection_prompt
                            )
                        )
                        reflection_tasks.append(reflection_task)
                    reflected_responses = await asyncio.gather(*reflection_tasks)
                    cleaned_responses = [
                        debate._clean_response(response)
                        for response in reflected_responses
                    ]

                # Update debate history in real-time
                active_debates[debate_id].debate_history.append(cleaned_responses)
                active_debates[debate_id].current_round = round_num + 1

            # Generate final summary if enabled
            if summarize_final_answer:
                final_summary = await debate._summarize_context(
                    active_debates[debate_id].debate_history[-1]
                )
                active_debates[debate_id].final_answer = final_summary

            active_debates[debate_id].is_complete = True

        except Exception as e:
            print(f"Error in debate {debate_id}: {str(e)}")
            active_debates[debate_id].is_complete = True
            active_debates[debate_id].error = str(e)

    # Create event loop and run debate
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())
    loop.close()


@app.route("/debate", methods=["POST"])
def start_debate():
    data = request.json

    # Validate required input
    required_fields = [
        "prompt",
        "num_rounds",
        "num_agents",
        "summarize_context",
        "use_reflection",
        "stream",
    ]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Set default for summarize_final_answer
    summarize_final_answer = data.get("summarize_final_answer", True)

    # Generate unique debate ID
    debate_id = str(len(active_debates))

    # Initialize debate manager
    active_debates[debate_id] = DebateManager()

    if data["stream"]:
        # Start debate in background
        thread = threading.Thread(
            target=run_debate_async,
            args=(
                debate_id,
                data["prompt"],
                data["num_rounds"],
                data["num_agents"],
                data["summarize_context"],
                data["use_reflection"],
                summarize_final_answer,
            ),
        )
        thread.start()

        return jsonify({"debate_id": debate_id, "status": "started", "stream": True})
    else:
        # Run debate synchronously
        run_debate_async(
            debate_id,
            data["prompt"],
            data["num_rounds"],
            data["num_agents"],
            data["summarize_context"],
            data["use_reflection"],
            summarize_final_answer,
        )

        # Return complete results
        return jsonify(
            {
                "debate_history": active_debates[debate_id].debate_history,
                "final_answer": active_debates[debate_id].final_answer,
            }
        )


@app.route("/debate/<debate_id>/status", methods=["GET"])
def get_debate_status(debate_id):
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404

    debate_manager = active_debates[debate_id]

    if hasattr(debate_manager, "error"):
        return jsonify({"status": "error", "error": debate_manager.error}), 500

    response = {
        "status": "complete" if debate_manager.is_complete else "in_progress",
        "debate_history": debate_manager.debate_history,
        "current_round": debate_manager.current_round,
    }

    if debate_manager.is_complete:
        response["final_answer"] = debate_manager.final_answer

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
