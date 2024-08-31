from flask import Flask, request, jsonify
from pathlib import Path
from typing import Optional, List
import uuid

from models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from models.llama3.reference_impl.generation import Llama

app = Flask(__name__)

# In-memory store for sessions
sessions = {}

# Initialize the generator outside the route functions to avoid reinitialization
tokenizer_path = "/home/ubuntu/redteam-ai/models/llama3/api/tokenizer.model"
ckpt_dir = "/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-8B-Instruct"
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=512,
    max_batch_size=4,
    model_parallel_size=1,
)

@app.route('/create_session', methods=['POST'])
def create_session():
    data = request.json
    system_message_content = data.get("system_message")

    if not system_message_content:
        return jsonify({"error": "System message is required"}), 400

    # Create a unique session ID
    session_id = str(uuid.uuid4())

    # Initialize the session with the system message
    sessions[session_id] = {
        "system_message": SystemMessage(content=system_message_content),
        "dialog": [SystemMessage(content=system_message_content)],
    }

    return jsonify({"session_id": session_id})

@app.route('/chat_completion', methods=['POST'])
def chat_completion():
    data = request.json
    session_id = data.get("session_id")
    user_message_content = data.get("user_message")

    if not session_id or not user_message_content:
        return jsonify({"error": "Session ID and user message are required"}), 400

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    # Retrieve the session dialog
    session = sessions[session_id]
    dialog = session["dialog"]

    # Add the user's message to the dialog
    user_message = UserMessage(content=user_message_content)
    dialog.append(user_message)

    # Generate the model's response
    result = generator.chat_completion(
        dialog,
        max_gen_len=None,
        temperature=0.6,
        top_p=0.9,
    )

    # Add the model's response to the dialog
    out_message = result.generation
    dialog.append(out_message)

    return jsonify({
        "role": out_message.role,
        "content": out_message.content
    })

@app.route('/get_session', methods=['GET'])
def get_session():
    session_id = request.args.get("session_id")
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    # Return the full dialog history for the session
    session = sessions[session_id]
    dialog = [{"role": msg.role, "content": msg.content} for msg in session["dialog"]]

    return jsonify({
        "session_id": session_id,
        "dialog": dialog
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
