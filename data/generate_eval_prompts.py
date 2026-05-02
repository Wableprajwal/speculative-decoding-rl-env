"""Generates 100 diverse eval prompts and saves to data/eval_prompts.txt"""

prompts = [
    # Geography
    "The capital of France is", "The longest river in Africa is",
    "The highest mountain in the world is", "The capital of Japan is",
    "The Amazon rainforest is located in", "The Sahara Desert spans across",
    "The capital of Australia is", "The Pacific Ocean is the",
    "The capital of Brazil is", "The Nile River flows through",
    # Science
    "The speed of light is approximately", "Photosynthesis is the process by which",
    "The human body contains approximately", "The periodic table was invented by",
    "DNA stands for", "The theory of evolution was proposed by",
    "The boiling point of water at sea level is", "Gravity was described by",
    "The smallest unit of matter is", "The Big Bang theory states that",
    # History
    "The French Revolution began in", "World War II ended in",
    "The first moon landing occurred in", "The Roman Empire fell in",
    "The printing press was invented by", "The Declaration of Independence was signed in",
    "The Renaissance period began in", "The Cold War lasted from",
    "The Berlin Wall fell in", "The first computer was built in",
    # Technology
    "The first programming language was", "The internet was invented in",
    "Artificial intelligence refers to", "Machine learning is a subset of",
    "The Python programming language was created by", "The first iPhone was released in",
    "Cloud computing refers to", "The Linux kernel was created by",
    "Blockchain technology was first described in", "The transistor was invented in",
    # Math
    "The square root of 144 is", "Pi is approximately equal to",
    "The Fibonacci sequence begins with", "A prime number is defined as",
    "The Pythagorean theorem states that", "Calculus was invented by",
    "The value of Euler's number e is approximately", "A quadratic equation has the form",
    "The sum of angles in a triangle is", "Binary number system uses only",
    # Story starters
    "On a cold winter morning,", "The old lighthouse stood at the edge of",
    "She had never seen anything like it before,", "The last train left the station at",
    "Deep in the forest, there was a",  "The scientist stared at the results and",
    "After ten years away,", "The message arrived at midnight,",
    "No one expected the discovery of", "The robot looked up and said,",
    # Nature
    "The migration of birds is triggered by", "Coral reefs are important because",
    "The water cycle consists of", "Volcanoes form when",
    "Earthquakes are caused by", "The Amazon produces approximately",
    "Polar ice caps are melting because", "The ozone layer protects Earth from",
    "Hurricanes form over", "Bioluminescence is the ability of",
    # Economics
    "Inflation refers to the", "The stock market is a place where",
    "Gross domestic product measures", "Supply and demand determines",
    "A recession is defined as", "Central banks control",
    "Cryptocurrency is a form of", "The gold standard refers to",
    "Free trade agreements allow", "Microeconomics focuses on",
    # Medicine
    "The human immune system protects", "Antibiotics are used to treat",
    "The heart pumps blood through", "Vaccines work by",
    "The nervous system is responsible for", "Cancer occurs when",
    "Mental health refers to", "The digestive system breaks down",
    "Genetics is the study of", "The placebo effect occurs when",
    # Space
    "The Milky Way galaxy contains", "Black holes are formed when",
    "The International Space Station orbits", "Mars is known as",
    "The speed required to escape Earth's gravity is", "Neutron stars are created when",
    "The James Webb Space Telescope can observe", "Solar flares are caused by",
    "The nearest star to Earth is", "Dark matter makes up approximately",
]

import os
os.makedirs("data", exist_ok=True)
with open("data/eval_prompts.txt", "w") as f:
    for p in prompts:
        f.write(p + "\n")

print(f"Generated {len(prompts)} eval prompts -> data/eval_prompts.txt")
