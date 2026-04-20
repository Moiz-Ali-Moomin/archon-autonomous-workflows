import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "agentpassword")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def store_goal_generates_file(task_id: str, goal: str, files: list):
    driver = get_driver()
    with driver.session() as session:
        for filename in files:
            session.run(
                """
                MERGE (g:Goal {task_id: $task_id})
                  ON CREATE SET g.text = $goal
                MERGE (f:File {name: $filename})
                MERGE (g)-[:GENERATES]->(f)
                """,
                task_id=task_id,
                goal=goal,
                filename=filename,
            )


def store_error_fixed_by_code(task_id: str, error: str, iteration: int):
    driver = get_driver()
    short_error = error[:300]
    with driver.session() as session:
        session.run(
            """
            MERGE (e:Error {task_id: $task_id, iteration: $iteration})
              ON CREATE SET e.message = $error
            MERGE (g:Goal {task_id: $task_id})
            MERGE (g)-[:ENCOUNTERED]->(e)
            """,
            task_id=task_id,
            iteration=iteration,
            error=short_error,
        )


def store_task_graph(task_id: str, goal: str, result: dict):
    files_written = list(result.get("files_written", ["output/main.py"]))
    store_goal_generates_file(task_id, goal, files_written)

    execution = result.get("execution", {})
    if not result.get("success"):
        error = execution.get("stderr") or execution.get("stdout") or "unknown error"
        store_error_fixed_by_code(task_id, error, result.get("iterations", 1))
