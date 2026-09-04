from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

documents = {
    "document1.pdf": """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is the field of computer science concerned
with creating systems that can perform tasks that normally require human
intelligence.

Machine Learning (ML) is a subset of AI in which computers learn patterns
from data instead of being explicitly programmed for every task.

Common machine learning applications include classification, regression,
recommendation systems, computer vision and natural language processing.

Supervised learning uses labelled training data. Unsupervised learning
works with data without predefined labels.
""",

    "document2.pdf": """
Cloud Computing

Cloud computing provides computing resources such as servers, storage,
databases and software through the internet.

The three major cloud service models are Infrastructure as a Service (IaaS),
Platform as a Service (PaaS), and Software as a Service (SaaS).

Cloud computing provides scalability, flexibility and reduced infrastructure
management.

Deployment models include public cloud, private cloud and hybrid cloud.
""",

    "document3.pdf": """
Cybersecurity

Cybersecurity protects computers, networks, applications and data from
unauthorized access and malicious attacks.

Important security principles include confidentiality, integrity and
availability, commonly known as the CIA triad.

Authentication verifies the identity of a user, while authorization
determines what resources that user is allowed to access.

Common cyber threats include phishing, malware, ransomware and denial of
service attacks.
""",

    "document4.pdf": """
Internet of Things

The Internet of Things (IoT) refers to connected physical devices that
collect, exchange and process data through communication networks.

IoT systems commonly contain sensors, communication modules, processing
components and applications.

Examples include smart homes, industrial monitoring systems, connected
vehicles and smart agriculture.

IoT devices can send sensor information to cloud platforms for storage,
analysis and visualization.
""",

    "document5.pdf": """
Digital Twins

A digital twin is a virtual representation of a physical object, process
or system.

Digital twins can use real-time or simulated data to monitor system
behavior, analyze performance and support decision making.

A digital twin may contain a physical system, a virtual model, a data
communication layer and analytics.

Applications include smart buildings, manufacturing, healthcare,
transportation and energy management.

Digital twins can help predict failures, optimize operations and improve
system efficiency.
"""
}

for filename, text in documents.items():

    path = DATA_DIR / filename

    pdf = canvas.Canvas(str(path), pagesize=A4)

    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica", 11)

    for line in text.strip().splitlines():

        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 50

        pdf.drawString(50, y, line[:100])
        y -= 18

    pdf.save()

    print(f"Created: {path}")

print("\n5 PDF documents created successfully.")