import json
from flask import request, jsonify
from python.helpers.api import api_bp
from python.helpers.distributed_orchestrator import get_orchestrator, TaskPriority
from agent import AgentContext


@api_bp.route("/orchestration/register", methods=["POST"])
def register_agent():
    """Register an agent with the orchestrator"""
    try:
        data = request.get_json()
        context_id = data.get("context_id")
        skills = data.get("skills", [])
        
        if not context_id:
            return jsonify({"error": "context_id is required"}), 400
        
        # Get the agent context
        context = AgentContext.get(context_id)
        if not context:
            return jsonify({"error": f"Agent context {context_id} not found"}), 404
        
        orchestrator = get_orchestrator()
        agent_id = orchestrator.register_agent(context, skills)
        
        return jsonify({
            "success": True,
            "agent_id": agent_id,
            "skills": skills
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/register", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/decompose", methods=["POST"])
def decompose_goal():
    """Decompose a goal into atomic subtasks"""
    try:
        data = request.get_json()
        goal = data.get("goal")
        context = data.get("context", "")
        
        if not goal:
            return jsonify({"error": "goal is required"}), 400
        
        orchestrator = get_orchestrator()
        subtasks = orchestrator.decompose_goal(goal, context)
        
        # Convert subtasks to serializable format
        subtask_data = []
        for subtask in subtasks:
            subtask_data.append({
                "uuid": subtask.uuid,
                "name": subtask.name,
                "description": subtask.description,
                "priority": subtask.priority.name,
                "estimated_duration": subtask.estimated_duration,
                "required_skills": subtask.required_skills,
                "dependencies": subtask.dependencies,
                "state": subtask.state.value,
                "created_at": subtask.created_at.isoformat()
            })
        
        return jsonify({
            "success": True,
            "subtasks": subtask_data,
            "count": len(subtasks)
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/decompose", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/assign", methods=["POST"])
def assign_tasks():
    """Assign available subtasks to capable agents"""
    try:
        orchestrator = get_orchestrator()
        assignments = orchestrator.assign_subtasks()
        
        assignment_data = []
        for subtask, agent_id in assignments:
            assignment_data.append({
                "subtask_uuid": subtask.uuid,
                "subtask_name": subtask.name,
                "agent_id": agent_id,
                "assigned_at": subtask.started_at.isoformat() if subtask.started_at else None
            })
        
        return jsonify({
            "success": True,
            "assignments": assignment_data,
            "count": len(assignments)
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/assign", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/status", methods=["GET"])
def get_orchestration_status():
    """Get orchestration system status"""
    try:
        orchestrator = get_orchestrator()
        status = orchestrator.get_orchestration_status()
        
        return jsonify({
            "success": True,
            "status": status
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/status", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/tensor", methods=["GET"])
def get_task_tensor():
    """Get the task tensor representation"""
    try:
        orchestrator = get_orchestrator()
        tensor = orchestrator.get_task_tensor()
        
        return jsonify({
            "success": True,
            "tensor_shape": tensor.shape,
            "tensor_data": tensor.tolist()  # Convert numpy array to list for JSON
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/tensor", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/complete", methods=["POST"])
def mark_task_completed():
    """Mark a subtask as completed"""
    try:
        data = request.get_json()
        subtask_uuid = data.get("subtask_uuid")
        result = data.get("result", "Task completed successfully")
        
        if not subtask_uuid:
            return jsonify({"error": "subtask_uuid is required"}), 400
        
        orchestrator = get_orchestrator()
        orchestrator.mark_subtask_completed(subtask_uuid, result)
        
        return jsonify({
            "success": True,
            "message": f"Subtask {subtask_uuid} marked as completed"
        })
    
    except Exception as e:
        import logging
        logging.error("An error occurred in /orchestration/complete", exc_info=True)
        return jsonify({"error": "An internal error has occurred."}), 500


@api_bp.route("/orchestration/heartbeat", methods=["POST"])
def update_heartbeat():
    """Update agent heartbeat"""
    try:
        data = request.get_json()
        agent_id = data.get("agent_id")
        
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400
        
        orchestrator = get_orchestrator()
        orchestrator.update_agent_heartbeat(agent_id)
        
        return jsonify({
            "success": True,
            "message": f"Heartbeat updated for agent {agent_id}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/orchestration/unregister", methods=["POST"])
def unregister_agent():
    """Unregister an agent from the orchestrator"""
    try:
        data = request.get_json()
        agent_id = data.get("agent_id")
        
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400
        
        orchestrator = get_orchestrator()
        orchestrator.unregister_agent(agent_id)
        
        return jsonify({
            "success": True,
            "message": f"Agent {agent_id} unregistered successfully"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500