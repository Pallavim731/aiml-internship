\# W10D4 - Human-in-the-Loop with LangGraph



\## Objective



Build and test a LangGraph workflow with three main nodes:



\- Classify

\- Route

\- Respond



The workflow also includes conditional routing and a human-in-the-loop interruption.



\---



\## Workflow



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

&#x20;   +----------------------+

&#x20;   |                      |

&#x20;   v                      v

&#x20;GENERAL                IMPORTANT

&#x20;   |                      |

&#x20;   |                      v

&#x20;   |               HUMAN APPROVAL

&#x20;   |                      |

&#x20;   |                      v

&#x20;   +---------> RESPOND <---+

&#x20;                 |

&#x20;                 v

&#x20;                END

