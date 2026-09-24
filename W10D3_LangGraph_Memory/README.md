\# W10D3 - LangGraph + Memory — Persistent Conversations



\## Objective



Build a stateful LangGraph agent with:



\- Three nodes: classify, route, respond

\- Conditional routing

\- Persistent conversation state

\- Human-in-the-loop interruption

\- Resume after human input

\- Testing with five inputs



\## Architecture



```text

User Input

&#x20;   |

&#x20;   v

+-----------+

|  CLASSIFY |

+-----------+

&#x20;   |

&#x20;   v

+-----------+

|   ROUTE   |

+-----------+

&#x20;   |

&#x20;   +--------------------+

&#x20;   |                    |

&#x20;   v                    v

&#x20;GENERAL              IMPORTANT

&#x20;   |                    |

&#x20;   |                    v

&#x20;   |             HUMAN APPROVAL

&#x20;   |                    |

&#x20;   |                    v

&#x20;   +----------> RESPOND <+

&#x20;                  |

&#x20;                  v

&#x20;                 END

