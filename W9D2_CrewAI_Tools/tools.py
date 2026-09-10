from crewai_tools import SerperDevTool
from crewai.tools import BaseTool


# Web Search Tool
search_tool = SerperDevTool()


# Simple Python Code Execution Tool
class CodeExecutionTool(BaseTool):
    name: str = "Code Execution Tool"

    description: str = (
        "Executes Python code for calculations and simple data processing. "
        "Use this tool when a numerical calculation or Python execution is required."
    )

    def _run(self, code: str) -> str:
        try:
            local_vars = {}

            exec(
                code,
                {"__builtins__": __builtins__},
                local_vars
            )

            return str(local_vars)

        except Exception as e:
            return f"Code execution error: {e}"


code_tool = CodeExecutionTool()