\# Baseline vs Web-Enhanced CrewAI Research



\## 1. Baseline Version



The baseline CrewAI system used three agents:



\- Researcher

\- Technical Writer

\- Reviewer



The Researcher used the local Ollama Llama 3.2 model without external web search.



\## 2. Web-Enhanced Version



The Researcher was enhanced with the SerperDevTool web search tool.



The Researcher could retrieve current information from the web before passing the findings to the Technical Writer.



The Reviewer then checked and improved the generated report.



\## 3. Improvements Observed



The web-enhanced approach provided:



\- More current information

\- Better coverage of recent developments

\- More real-world examples

\- Additional factual information

\- Reduced dependence on model memory

\- Better research depth



\## 4. Multi-Agent Workflow



The final workflow is:



Researcher + Web Search

&#x20;       ↓

Technical Writer

&#x20;       ↓

Reviewer

&#x20;       ↓

Final Research Report



\## 5. Conclusion



Adding web search improved the CrewAI research workflow by allowing the Researcher to access current external information. The Writer converted the research findings into a structured report, while the Reviewer checked the final output for clarity, completeness and quality.

